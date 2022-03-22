from collections import defaultdict, OrderedDict
from itertools import groupby
from operator import itemgetter

from tqdm import tqdm
import numpy as np
from datetime import datetime
import torch

from oneqa.mrc.processors.postprocessors.abstract import AbstractPostProcessor
from oneqa.mrc.processors.postprocessors.scorers import initialize_scorer
from oneqa.mrc.data_models.target_type import TargetType

def to_list(tensor: torch.Tensor) -> list:
    """
    :param torch.Tensor tensor: tensor to process
    :return: detached cpu tensor list
    :rtype: list
    """
    return tensor.detach().cpu().tolist()

class ExtractivePostProcessor(AbstractPostProcessor):
    def __init__(self, k: int, n_best_size: int, max_answer_length: int, scorer_type='weighted_sum_target_type_and_score_diff'):
        super().__init__(k)
        self._n_best_size = n_best_size
        self._max_answer_length = max_answer_length
        # self._span_trackers = defaultdict(span_tracker_factory)  # TODO factory type?
        self._score_calculator = initialize_scorer(scorer_type)

    def process(self, examples, features, predictions):
        features_itr = groupby(features, key=itemgetter('example_idx'))
        predictions_i = 0
        if len(features) != predictions[0].shape[0] and all(
                p.shape[0] == predictions[0].shape[0] for p in predictions[1:]):
            raise ValueError(f"Size mismatch withing {len(features)} features and predictions "
                             f"of first dim {[p.shape[0] for p in predictions]}")
        
        all_start_logits = predictions[0]
        all_end_logits = predictions[1]
        all_targettype_logits = predictions[2]

        # The dictionaries we have to fill.
        all_predictions = OrderedDict()
        all_target_types = []

        start_idx = 0
        for example_idx, example in enumerate(tqdm(examples)):
            example_id, example_features = next(features_itr)
            if example_id != example_idx:
                raise ValueError(f"Example id mismatch between example ({example['example_id']}) "
                                 f"and feature ({example_id})")
            example_features = list(example_features)
            context = example["context"]
            example_start_logits = all_start_logits[start_idx:start_idx+len(example_features)]
            example_end_logits = all_end_logits[start_idx:start_idx+len(example_features)]
            example_targettype_preds = all_targettype_logits[start_idx:start_idx+len(example_features)]  
            start_idx += len(example_features)

            min_null_prediction = None
            prelim_predictions = []
            # target_type = None

            for i, input_feature in enumerate(example_features):
                start_logits = example_start_logits[i].tolist()
                end_logits = example_end_logits[i].tolist()
                target_type_logits = example_targettype_preds[i].tolist()
                offset_mapping = input_feature["offset_mapping"]
                # input_feature = feature_index.item()

                # target_type_idx = np.argmax(target_type_logits)
                # target_type_score = target_type_logits[target_type_idx]
                # if target_type == None or target_type_score > target_type[1]:
                #     target_type = (target_type_idx,target_type_score)

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
                        
                        short_answer_score = self._score_calculator(start_logits[start_index] + end_logits[end_index],
                                                feature_null_score, target_type_logits)
                        prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "cls_score": feature_null_score,
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                            "span_answer_score" : short_answer_score,
                            "start_index": start_index,
                            "end_index":end_index,
                            "target_type_logits": target_type_logits,
                            "span_answer_text": context[i][offset_mapping[start_index][0]:offset_mapping[end_index][1]]
                        }

                    )
            example_predictions = sorted(prelim_predictions, key=lambda x: x["span_answer_score"], reverse=True)[:self._n_best_size]
            # all_target_types.append(target_type)
            all_predictions[example_idx] = example_predictions
            # Use the offsets to gather the answer text in the original context.
            # context = example["context"]
            # for pred in example_predictions:
            #     offsets = pred.pop("offsets")
            #     pred["text"] = context[offsets[0] : offsets[1]]

            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
            # failure.
            if len(example_predictions) == 0:
                example_predictions.insert(0, {"span_answer_text": "empty", "start_logit": 0.0, "end_logit": 0.0, "span_answer_score": 0.0})

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

        # Make `predictions` JSON-serializable by casting np.float back to float.
        # all_nbest_json[example["id"]] = [
        #     {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
        #     for pred in predictions
        # ]

        #format for metrics

        return all_predictions
        


                









