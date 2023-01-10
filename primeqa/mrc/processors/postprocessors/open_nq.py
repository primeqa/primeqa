from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from typing import List, Dict, Any, Tuple
import os

import sklearn
from sklearn.neural_network import MLPClassifier
import joblib

from datasets import Dataset
from tqdm import tqdm
import numpy as np
from datetime import datetime
import torch
import logging
from transformers import EvalPrediction

from primeqa.mrc.processors.postprocessors.abstract import AbstractPostProcessor
from primeqa.mrc.processors.postprocessors.scorers import initialize_scorer
from primeqa.mrc.data_models.target_type import TargetType
from primeqa.mrc.data_models.eval_prediction_with_processing import EvalPredictionWithProcessing
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from primeqa.calibration.confidence_scorer import ConfidenceScorer

logger = logging.getLogger(__name__)


class OpenNQPostProcessor(AbstractPostProcessor):
    """
    Post processor for OpenNQ QA (use with `ExtractiveQAHead`).
    """
    def __init__(self,
                 *args,
                 n_best_size: int,
                 scorer_type=SupportedSpanScorers.WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF,
                 output_confidence_feature: bool = False,
                 confidence_model_path: str = None,
                 **kwargs):
        """
        Args:
            *args: Arguments for super class constructor.
            n_best_size: Max number of start/end logits to consider (max values).
            scorer_type: Scoring algorithm to use.
            **kwargs: Keyword Arguments for super class constructor.
        """
        super().__init__(*args, **kwargs)
        self._n_best_size = n_best_size
        self._score_calculator = initialize_scorer(scorer_type)
        self._output_confidence_feature = output_confidence_feature
        if confidence_model_path:
            self._confidence_scorer = ConfidenceScorer(confidence_model_path)
        else:
            self._confidence_scorer = None

    def process(self, examples: Dataset, features: Dataset, predictions: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        if len(features) != predictions[0].shape[0] and all(
                p.shape[0] == predictions[0].shape[0] for p in predictions[1:]):
            raise ValueError(f"Size mismatch withing {len(features)} features and predictions "
                             f"of first dim {[p.shape[0] for p in predictions]}")

        if self._output_confidence_feature:
            all_start_logits, all_end_logits, all_targettype_logits, \
            all_start_stdev, all_end_stdev, all_query_passage_similarity = predictions
        else:
            all_start_logits, all_end_logits, all_targettype_logits = predictions
            all_start_stdev = None
            all_end_stdev = None
            all_query_passage_similarity = None

        # The dictionaries we have to fill.
        all_predictions = {}
        for i, feature in enumerate(features):
            idx = feature['example_idx']
            example = examples[idx]
            if feature['example_id'] != example['id'][0]:
                raise ValueError(f"Example id mismatch between example ({example['id']}) "
                                 f"and feature ({feature['example_id']})")
            all_predictions[str(feature['example_id'])] = []

            start_logits = all_start_logits[i].tolist()
            end_logits = all_end_logits[i].tolist()
            target_type_logits = all_targettype_logits[i].tolist()

            if all_start_stdev is not None and all_end_stdev is not None \
                and all_query_passage_similarity is not None:
                start_stdev = all_start_stdev[i].tolist()
                end_stdev = all_end_stdev[i].tolist()
                query_passage_similarity = all_query_passage_similarity[i].tolist()
            else:
                start_stdev = [[0.0] * len(start_logits[0])] * len(start_logits)
                end_stdev = [[0.0] * len(end_logits[0])] * len(end_logits)
                query_passage_similarity = [0.0] * len(start_logits)

            for k in range(len(start_logits)):
                offset_mapping = feature["offset_mapping"][k]

                start_indexes = np.argsort(start_logits[k][:len(offset_mapping)])[-1 : -self._n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits[k][:len(offset_mapping)])[-1 : -self._n_best_size - 1 : -1].tolist()
                prelim_predictions = []
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                        # to part of the input_ids that are not in the context.
                        if (
                            start_index >= len(offset_mapping)
                            or end_index >= len(offset_mapping)
                            or offset_mapping[start_index] is None
                            or len(offset_mapping[start_index]) < 2
                            or offset_mapping[end_index] is None
                            or len(offset_mapping[end_index]) < 2
                        ):
                            continue
                        # Don't consider answers with a length that is either < 0 or > max_answer_length.
                        if end_index < start_index or end_index - start_index + 1 > self._max_answer_length:
                            continue

                        start_position = offset_mapping[start_index][0]
                        end_position = offset_mapping[end_index][1]
                        context = example['passages'][0][k]['text']
                        ir_score = example['passages'][0][k]['score']
                        normalized_ir_score = example['passages'][0][k]['normalized_score']
                        span_answer_text = context[offset_mapping[start_index][0]:offset_mapping[end_index][1]]
                        feature_null_score = start_logits[k][0] + end_logits[k][0]
                        span_answer_score = self._score_calculator(start_logits[k][start_index] + end_logits[k][end_index],
                                                               feature_null_score, target_type_logits[k])
                        prelim_predictions.append({
                            'example_id': str(feature['example_id']) + "_" + str(k + 1),
                            'cls_score': feature_null_score,
                            'start_logit': start_logits[k][start_index],
                            'end_logit': end_logits[k][end_index],
                            'span_answer': {
                                "start_position": start_position,
                                "end_position": end_position,
                            },
                            'span_answer_score' : span_answer_score,
                            'start_index': start_index,
                            'end_index':   end_index,
                            'passage_index' : 0,
                            'target_type_logits': target_type_logits[k],
                            'span_answer_text': span_answer_text,
                            'yes_no_answer': 0,
                            'start_stdev': start_stdev[k][start_index],
                            'end_stdev': end_stdev[k][end_index],
                            'query_passage_similarity': query_passage_similarity[k],
                            'ir_score': ir_score,
                            'normalized_ir_score': normalized_ir_score,
                        })
                example_predictions = sorted(prelim_predictions, key=itemgetter('span_answer_score'), reverse=True)[:self._k]

                # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
                # failure.
                if len(example_predictions) == 0:
                    logger.info(f'We do not have any non-null predictions for example {feature["example_id"]}')
                    for n in range(self._k):
                        example_predictions.append(
                            {
                                'example_id': str(feature['example_id']) + "_" + str(k + 1),
                                'cls_score': 0.0,
                                'start_logit': 0.0,
                                'end_logit': 0.0,
                                'span_answer': {'start_position': -1, 'end_position': -1,},
                                'span_answer_score': 0.0,
                                'span_answer_text': "empty",
                                'start_index': -1,
                                'end_index': -1,
                                'passage_index' : -1,
                                'target_type_logits' : [0, 0, 0, 0, 0],
                                'yes_no_answer': int(TargetType.NO_ANSWER),
                                'start_stdev': 0,
                                'end_stdev': 0,
                               'query_passage_similarity': 0,
                                'ir_score': 0,
                                'normalized_ir_score': 0,
                            })

                # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
                # the LogSumExp trick).
                scores = np.array([pred["span_answer_score"] for pred in example_predictions])
                exp_scores = np.exp(scores - np.max(scores))
                probs = exp_scores / exp_scores.sum()

                # Normalized score over top-k predictions.
                for prob, pred in zip(probs, example_predictions):
                    pred["normalized_span_answer_score"] = prob

                all_predictions[str(feature['example_id'])].append(example_predictions)

        for example_id in all_predictions:
            for i, preds in enumerate(all_predictions[example_id]):
                for j in range(len(preds)):
                    all_predictions[example_id][i][j]['normalized_span_answer_score_by_passage'] = 0.0

            scores = np.array([p["span_answer_score"] for p in preds for preds in all_predictions[example_id]])
            max_score = np.max(scores)
            sum_exp_score = 0.0
            for i, preds in enumerate(all_predictions[example_id]):
                for j in range(len(preds)):
                    sum_exp_score += np.exp(all_predictions[example_id][i][j]['span_answer_score'] - max_score)
            span_scores = {}
            for i, preds in enumerate(all_predictions[example_id]):
                for j in range(len(preds)):
                    all_predictions[example_id][i][j]['normalized_span_answer_score_by_passage'] = \
                        np.exp(all_predictions[example_id][i][j]['span_answer_score'] - max_score) / sum_exp_score
                    if all_predictions[example_id][i][j]['span_answer_text'].lower() not in span_scores:
                        span_scores[all_predictions[example_id][i][j]['span_answer_text'].lower()] = 0.0
                    span_scores[all_predictions[example_id][i][j]['span_answer_text'].lower()] += \
                        all_predictions[example_id][i][j]['normalized_span_answer_score_by_passage']
            for i, preds in enumerate(all_predictions[example_id]):
                for j in range(len(preds)):
                    all_predictions[example_id][i][j]['normalized_span_answer_score_by_passage'] = \
                        span_scores[all_predictions[example_id][i][j]['span_answer_text'].lower()]

            # Confidence score
            for i, preds in enumerate(all_predictions[example_id]):
                if self._confidence_scorer is not None and self._confidence_scorer.model_exists():
                    scores = self._confidence_scorer.predict_scores(preds)
                    for j in range(len(preds)):
                        all_predictions[example_id][i][j]["confidence_score"] = scores[j]
                else:
                    for j in range(len(preds)):
                        all_predictions[example_id][i][j]["confidence_score"] = \
                            all_predictions[example_id][i][j]["normalized_span_answer_score"]

        new_all_predictions = {}
        for example_id in all_predictions:
            for preds in all_predictions[example_id]:
                new_all_predictions[preds[0]['example_id']] = preds

        return new_all_predictions
        
    def prepare_examples_as_references(self, examples: Dataset) -> List[Dict[str, Any]]:
        references = []
        for example_idx in range(examples.num_rows):
            example = examples[example_idx]
            for passage in example["passages"]:
                passage_index = 0 if passage['start'] >= 0 and passage['end'] >= 0 else -1
                label = {
                    'start_position': [passage['start']],
                    'end_position': [passage['end']],
                    'passage_index': [passage_index],
                    'yes_no_answer': [0],
                    'example_id': [example['id'] + "_" + str(passage['rank'])],
                    'language': ['english'],
                    'document_plaintext': [''],
                    'question': [example['question']]
                }
                references.append(label)
        return references

    def process_references_and_predictions(self, examples, features, predictions) -> EvalPredictionWithProcessing:
        references = self.prepare_examples_as_references(examples)
        predictions = self.process(examples, features, predictions)
        predictions_for_metric = []

        for example_id, preds in predictions.items():
            top_pred = preds[0]
            prediction_for_metric = {
                'example_id': example_id,
                'start_position': top_pred['span_answer']['start_position'],
                'end_position': top_pred['span_answer']['end_position'],
                'passage_index': top_pred['passage_index'],
                'yes_no_answer': top_pred['yes_no_answer'],
                'confidence_score': top_pred['span_answer_score']
            }
            predictions_for_metric.append(prediction_for_metric)

        # noinspection PyTypeChecker
        return EvalPredictionWithProcessing(
            label_ids=references,
            predictions=predictions,
            processed_predictions=predictions_for_metric
        )
