from collections import defaultdict
from itertools import groupby
from operator import itemgetter
from typing import List, Dict, Any

from datasets import Dataset
from tqdm import tqdm
import numpy as np
from datetime import datetime
import torch
import logging
from transformers import EvalPrediction

from oneqa.mrc.processors.postprocessors.abstract import AbstractPostProcessor
from oneqa.mrc.processors.postprocessors.scorers import initialize_scorer
from oneqa.mrc.data_models.target_type import TargetType

logger = logging.getLogger(__name__)

class ExtractivePostProcessor(AbstractPostProcessor):
    def __init__(self, k: int, n_best_size: int, max_answer_length: int, scorer_type: str='weighted_sum_target_type_and_score_diff'):
        super().__init__(k)
        self._n_best_size = n_best_size
        self._max_answer_length = max_answer_length
        # self._span_trackers = defaultdict(span_tracker_factory)  # TODO factory type?
        self._score_calculator = initialize_scorer(scorer_type)

    def process(self, examples, features, predictions):
        features_itr = groupby(features, key=itemgetter('example_idx'))  # TODO: may need to change this to example_id when getting rid of example_idx column.
        predictions_i = 0
        if len(features) != predictions[0].shape[0] and all(
                p.shape[0] == predictions[0].shape[0] for p in predictions[1:]):
            raise ValueError(f"Size mismatch withing {len(features)} features and predictions "
                             f"of first dim {[p.shape[0] for p in predictions]}")
        
        all_start_logits, all_end_logits, all_targettype_logits = predictions

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
            # example['example_id'] = example_id  # TODO: assign example id to examples before featurization
            contexts = example["context"]
            example_start_logits = all_start_logits[start_idx:start_idx+len(example_features)]
            example_end_logits = all_end_logits[start_idx:start_idx+len(example_features)]
            example_targettype_preds = all_targettype_logits[start_idx:start_idx+len(example_features)]  
            start_idx += len(example_features)

            # # TODO: move this to the preprocessor
            # label = {
            #     'start_position': example['target']['start_positions'],
            #     'end_position': example['target']['end_positions'],
            #     'passage_index': example['target']['passage_indices'],
            #     'yes_no_answer': list(map(TargetType.from_bool_label, example['target']['yes_no_answer'])),
            #     'example_id': example_id
            # }

            min_null_prediction = None
            prelim_predictions = []

            for i, input_feature in enumerate(example_features):
                if input_feature['example_id'] != example_id:
                    raise ValueError(f"Example id mismatch between example ({example_id}) "
                                 f"and feature ({input_feature['example_id']})")
                start_logits = example_start_logits[i].tolist()
                end_logits = example_end_logits[i].tolist()
                target_type_logits = example_targettype_preds[i].tolist()
                offset_mapping = input_feature["offset_mapping"]
                context_idx = input_feature['context_idx']

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

                start_indexes = np.argsort(start_logits)[-1 : -self._n_best_size - 1 : -1].tolist()
                end_indexes = np.argsort(end_logits)[-1 : -self._n_best_size - 1 : -1].tolist()
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
                        
                        span_answer_score = self._score_calculator(start_logits[start_index] + end_logits[end_index],
                                                feature_null_score, target_type_logits)
                        prelim_predictions.append(
                        {
                            'example_id': input_feature['example_id'],
                            'cls_score': feature_null_score,
                            'start_logit': start_logits[start_index],
                            'end_logit': end_logits[end_index],
                            'span_answer': {
                                "start_position": offset_mapping[start_index][0], 
                                "end_position": offset_mapping[end_index][1],
                            },
                            'span_answer_score' : span_answer_score,
                            'start_index': start_index,
                            'end_index':   end_index,
                            'passage_index' : context_idx,
                            'target_type_logits': target_type_logits,
                            'span_answer_text': contexts[context_idx][offset_mapping[start_index][0]:offset_mapping[end_index][1]],
                            'yes_no_answer': int(TargetType.NO_ANSWER)
                        }
                    )
            example_predictions = sorted(prelim_predictions, key=itemgetter('span_answer_score'), reverse=True)[:self._n_best_size]
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
                        'yes_no_answer': int(TargetType.NO_ANSWER)
                    })

            # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
            # the LogSumExp trick).
            scores = np.array([pred["span_answer_score"] for pred in example_predictions])
            exp_scores = np.exp(scores - np.max(scores))
            probs = exp_scores / exp_scores.sum()

            # Include the probabilities in our predictions.
            for prob, pred in zip(probs, example_predictions):
                pred["normalized_span_answer_score"] = prob

            # We assume null answer is not possible
            # Otherwise we first need to find the best non-empty prediction.
            # i = 0
            # while predictions[i]["text"] == "":
            #     i += 1
            # best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            # score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            # scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            # if score_diff > null_score_diff_threshold:
            #     all_predictions[example["id"]] = ""
            # else:
            #     all_predictions[example["id"]] = best_non_null_pred["text"]

        return all_predictions
        
    def prepare_examples_as_references(self, examples: Dataset) -> List[Dict[str, Any]]:
        references = []
        n_annotators = len(examples[0]['target']['start_positions'])
        for example_idx in range(examples.num_rows):
            example = examples[example_idx]
            # label = {
            #     'span_answer': {'start_position': example['target']['start_positions'],
            #                     'end_position': example['target']['end_positions'], },
            #     'passage_index': example['target']['passage_indices'],
            #     'yes_no_answer': example['target']['yes_no_answer'],
            #     'example_id': example['example_id']
            # }
            label = {
                'start_position': example['target']['start_positions'],
                'end_position': example['target']['end_positions'],
                'passage_index': example['target']['passage_indices'],
                'yes_no_answer': list(map(TargetType.from_bool_label, example['target']['yes_no_answer'])),  # TODO: decide on schema type for bool ans
                'example_id': [example['example_id']] * n_annotators,
                'language': [example['language']] * n_annotators  # TODO: schema does not have language by default
            }
            references.append(label)
        return references

    def process_references_and_predictions(self, examples, features, predictions) -> EvalPrediction:
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
                'confidence_score': top_pred['normalized_span_answer_score']
            }
            predictions_for_metric.append(prediction_for_metric)

        # noinspection PyTypeChecker
        return EvalPrediction(label_ids=references, predictions=predictions_for_metric)
