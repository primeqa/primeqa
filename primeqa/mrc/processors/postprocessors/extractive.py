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


class ExtractivePostProcessor(AbstractPostProcessor):
    """
    Post processor for extractive QA (use with `ExtractiveQAHead`).
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
        features_itr = groupby(features, key=itemgetter('example_idx'))
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

        start_idx = 0
        for example_idx, example in enumerate(tqdm(examples)):
            feat_example_idx, example_features = next(features_itr)
            if feat_example_idx != example_idx:
                raise ValueError(f"Example id mismatch between example ({example['example_id']}) "
                                 f"and feature ({feat_example_idx})")
            example_features = list(example_features)
            example_id = example_features[0]['example_id']
            contexts = example["context"]
            example_start_logits = all_start_logits[start_idx:start_idx+len(example_features)]
            example_end_logits = all_end_logits[start_idx:start_idx+len(example_features)]
            example_targettype_preds = all_targettype_logits[start_idx:start_idx+len(example_features)]

            if all_start_stdev is not None and all_end_stdev is not None and all_query_passage_similarity is not None:
                example_start_stdev = all_start_stdev[start_idx:start_idx+len(example_features)]
                example_end_stdev = all_end_stdev[start_idx:start_idx+len(example_features)]
                example_query_passage_similarity = all_query_passage_similarity[start_idx:start_idx+len(example_features)]
            else:
                example_start_stdev = None
                example_end_stdev = None
                example_query_passage_similarity = None
            start_idx += len(example_features)

            min_null_prediction = None
            prelim_predictions = []

            for i, input_feature in enumerate(example_features):
                if input_feature['example_id'] != example_id:
                    raise ValueError(f"Example id mismatch between example ({example_id}) "
                                 f"and feature ({input_feature['example_id']})")
                start_logits = example_start_logits[i].tolist()
                end_logits = example_end_logits[i].tolist()
                target_type_logits = example_targettype_preds[i].tolist()

                if example_start_stdev is not None and example_end_stdev is not None \
                        and example_query_passage_similarity is not None:
                    start_stdev = example_start_stdev[i].tolist()
                    end_stdev = example_end_stdev[i].tolist()
                    query_passage_similarity = float(example_query_passage_similarity[i])
                else:
                    start_stdev = [0.0] * len(start_logits)
                    end_stdev = [0.0] * len(end_logits)
                    query_passage_similarity = 0.0
                offset_mapping = input_feature["offset_mapping"]

                token_is_max_context = input_feature.get("token_is_max_context", None)
                # Update minimum null prediction.
                feature_null_score = start_logits[0] + end_logits[0]
                if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                    min_null_prediction = {
                        "offsets": (0, 0),
                        "score": feature_null_score,
                        "start_logit": start_logits[0],
                        "end_logit": end_logits[0],
                    }

                start_indexes = np.argsort(start_logits[:len(offset_mapping)])[-1 : -self._n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits[:len(offset_mapping)])[-1 : -self._n_best_size - 1 : -1].tolist()
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
                        # Don't consider answer that don't have the maximum context available (if such information is
                        # provided).
                        if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                            continue

                        start_position = offset_mapping[start_index][0]
                        end_position = offset_mapping[end_index][1]

                        if self._single_context_multiple_passages:
                            passage_candidates = example['passage_candidates']
                            for context_idx in range(len(passage_candidates['start_positions'])):
                                passage_start_position = passage_candidates['start_positions'][context_idx]
                                passage_end_position = passage_candidates['end_positions'][context_idx]
                                if passage_start_position <= start_position <= end_position <= passage_end_position:
                                    break
                            else:
                                context_idx = -1
                            passage_text = contexts[0]
                        else:
                            context_idx = input_feature['context_idx']
                            passage_text = contexts[context_idx]

                        span_answer_text = passage_text[offset_mapping[start_index][0]:offset_mapping[end_index][1]]
                        span_answer_score = self._score_calculator(start_logits[start_index] + end_logits[end_index],
                                                feature_null_score, target_type_logits)
                        prelim_predictions.append(
                        {
                            'example_id': input_feature['example_id'],
                            'cls_score': feature_null_score,
                            'start_logit': start_logits[start_index],
                            'end_logit': end_logits[end_index],
                            'span_answer': {
                                "start_position": start_position,
                                "end_position": end_position,
                            },
                            'span_answer_score' : span_answer_score,
                            'start_index': start_index,
                            'end_index':   end_index,
                            'passage_index' : context_idx,
                            'target_type_logits': target_type_logits,
                            'span_answer_text': span_answer_text,
                            'yes_no_answer': int(TargetType.NO_ANSWER),
                            'start_stdev': start_stdev[start_index],
                            'end_stdev': end_stdev[end_index],
                            'query_passage_similarity': query_passage_similarity
                        }
                    )
            example_predictions = sorted(prelim_predictions, key=itemgetter('span_answer_score'), reverse=True)[:self._k]
            all_predictions[example_id] = example_predictions

            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            if len(example_predictions) == 0: 
                logger.info(f'We do not have any non-null predictions for example {example_id}')
                example_predictions.append( 
                    {
                        'example_id': example_id,
                        'cls_score': 0.0,
                        'start_logit': 0.0, 
                        'end_logit': 0.0, 
                        'span_answer': {'start_position': -1, 'end_position': -1,},
                        'span_answer_score': 0.0, 
                        'span_answer_text': "empty", 
                        'start_index': -1,
                        'end_index':   -1,
                        'passage_index' : -1,
                        'target_type_logits' : [0, 0, 0, 0, 0],
                        'yes_no_answer': int(TargetType.NO_ANSWER),
                        'start_stdev': 0,
                        'end_stdev': 0,
                        'query_passage_similarity': 0
                    })

            # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
            # the LogSumExp trick).
            scores = np.array([pred["span_answer_score"] for pred in example_predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, example_predictions):
                pred["normalized_span_answer_score"] = prob

            # Confidence score
            if self._confidence_scorer is not None and self._confidence_scorer.model_exists():
                scores = self._confidence_scorer.predict_scores(example_predictions)
                for i in range(len(example_predictions)):
                    example_predictions[i]["confidence_score"] = scores[i]
            else:
                for i in range(len(example_predictions)):
                    example_predictions[i]["confidence_score"] = example_predictions[i]["normalized_span_answer_score"]

        return all_predictions
        
    def prepare_examples_as_references(self, examples: Dataset) -> List[Dict[str, Any]]:
        references = []
        for example_idx in range(examples.num_rows):
            example = examples[example_idx]
            n_annotators = len(example['target']['start_positions'])
            label = {
                'start_position': example['target']['start_positions'],
                'end_position': example['target']['end_positions'],
                'passage_index': example['target']['passage_indices'],
                'yes_no_answer': list(map(TargetType.from_bool_label, example['target']['yes_no_answer'])),  # TODO: decide on schema type for bool ans
                'example_id': [example['example_id']] * n_annotators,
                'language': [example['language']] * n_annotators,
                'document_plaintext': [example['document_plaintext']] * n_annotators,
                'question': [example['question']]  * n_annotators
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
