from datasets import Dataset
import pytest
import sys
from pytest import raises
from transformers import AutoTokenizer
from itertools import groupby
from operator import itemgetter


import numpy as np

from primeqa.calibration.confidence_scorer import ConfidenceScorer
from primeqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from primeqa.mrc.data_models.target_type import TargetType
from tests.primeqa.mrc.common.base import UnitTest
from tests.primeqa.mrc.common.parameterization import PARAMETERIZE_INVALID_SUBSAMPLING_PROBABILITIES

class TestConfidenceScorer(UnitTest):

    _start_score_of_cls_token = 0
    _end_score_of_cls_token = 0
    _start_stdev_of_cls_token = 1.0
    _end_stdev_of_cls_token = 1.0
    _score_none_token = -10
    _stdev_none_token = 2.0
    
    example1_start_scores = [ [0.5, 0.3, 0.2, 0.1, 0.1], 
                              [1.5, 0.7, 0.2, 0.1, 0.1],
                ]
    example1_end_scores = [ [0.5, 0.3, 0, 0, 0], 
                            [1.5, 0.7, 0, 0, 0],
                ]
    example1_start_stdevs = [ [0.1, 0.2, 0.3, 0.4, 0.5], 
                              [0.1, 0.2, 0.3, 0.4, 0.5],
                ]
    example1_end_stdevs = [ [0.5, 0.6, 0.7, 0.8, 0.9], 
                            [0.5, 0.6, 0.7, 0.8, 0.9],
                ]
    example1_query_passage_similarities = [ [1.5], [1.9],
                ]

    example2_start_scores = [ [-1, 2, 1, -3, -2.5, -5, -4, -2, -2, -1, -1.5, -2.5],  
                              [-3, -2, -5, -2,-1, -1, -3, -2, -4, -5, -1.5, -2.5], 
                              [-3, -2, -5, -2,-1, -1, -3, -2, -4, -5, -1.5, -2.5], 
                ]
    example2_end_scores = [   [-1, 1, 1.5, -3, -2.5, -5, -4, -2, -2, -1, -1.5, -2.5], 
                              [-3, -2, -5, -2,-1, -1, -3, -2, -4, -5, -1.5, -2.5],
                              [-3, -2, -5, -2,-1, -1, -3, -2, -4, -5, -1.5, -2.5],
                ]
    example2_start_stdevs = [ [0.5, 0.5, 0.6, 0.8, 0.7, 0.3, 0.2, 0.9, 0.5, 0.1, 0.9, 0.9],  
                              [0.8, 1.1, 1.6, 1.3, 1.2, 1.9, 1.7, 1.5, 1.8, 1.5, 1.4, 1.9], 
                              [0.6, 1.8, 0.3, 1.6, 0.4, 1.3, 0.5, 1.4, 0.9, 1.2, 0.7, 1.5], 
                ]
    example2_end_stdevs = [ [0.5, 0.5, 0.6, 0.8, 0.7, 0.3, 0.2, 0.9, 0.5, 0.1, 0.9, 0.9],  
                              [0.8, 1.1, 1.6, 1.3, 1.2, 1.9, 1.7, 1.5, 1.8, 1.5, 1.4, 1.9], 
                              [0.6, 1.8, 0.3, 1.6, 0.4, 1.3, 0.5, 1.4, 0.9, 1.2, 0.7, 1.5], 
                ]
    example2_query_passage_similarities = [ [1.1], [1.6], [1.3],
                ]


    _start_scores = [example1_start_scores, example2_start_scores]
    _end_scores = [ example1_end_scores, example2_end_scores]

    _start_stdevs = [example1_start_stdevs, example2_start_stdevs]
    _end_stdevs = [ example1_end_stdevs, example2_end_stdevs]

    _query_passage_similarities = [example1_query_passage_similarities, example2_query_passage_similarities]
    
    _target_type_scores = [
        np.array([ -4.2624542e-04, 1.2508204e+00, -1.8462906e-02, -4.0960628e-01, -3.8014248e-01,  -4.2624542e-04], dtype=np.float32),
        np.array([  1.0599241 , 0.01550296, -0.40537557, -0.3395432 ,  0.12297338],dtype=np.float32),
    ]

    _expected_predictions = {
        'foo-abc' : {
            'start_index': 0, #8,
            'end_index': 0, #8,
            'passage_index': 1,
            'target_type' : int(TargetType.SPAN_ANSWER),
            'span_answer_text': 'Bob'
        },
        'bar-123': {
            'start_index': 1, # 9,
            'end_index': 2, #10,
            'passage_index': 0,
            'target_type' : int(TargetType.NO_ANSWER),
            'span_answer_text': 'quick brown'
        }
    }

    def _start_end_target_type_logits_stdevs(self, examples, features):

        all_start_logits = []
        all_end_logits = []
        all_target_type_logits = []
        all_start_stdevs = []
        all_end_stdevs = []
        all_query_passage_similarities = []
        
        adjusted_start_end_index = {}

        features_itr = groupby(features, key=itemgetter('example_idx'))
        for example in examples:
            example_id = example['example_id']
            expected = self._expected_predictions[example_id]

            example_idx, example_features = next(features_itr)
            example_features = list(example_features)

            for feat_idx, feature in enumerate(example_features):
                offset_mapping = feature["offset_mapping"]

                if feat_idx == expected['passage_index']:
                    for i, offset in enumerate(offset_mapping):
                        if offset != None:
                            adjusted_start_end_index[example_id] = (expected['start_index']+i,
                                                    expected['end_index'] + i)
                            break

                start_logits = []
                end_logits = []
                start_stdevs = []
                end_stdevs = []
                query_passage_similarities = []
                
                start_scores_iter = iter(self._start_scores[example_idx][feat_idx])
                end_scores_iter = iter(self._end_scores[example_idx][feat_idx])
                start_stdevs_iter = iter(self._start_stdevs[example_idx][feat_idx])
                end_stdevs_iter = iter(self._end_stdevs[example_idx][feat_idx])
#                similarity_iter = iter(self._query_passage_similarities[example_idx][feat_idx])
                
                for i, offset in enumerate(offset_mapping):
                    if i == 0:
                        start_logits.append(self._start_score_of_cls_token)
                        end_logits.append(self._end_score_of_cls_token)
                        start_stdevs.append(self._start_stdev_of_cls_token)
                        end_stdevs.append(self._end_stdev_of_cls_token)
                    elif offset == None:
                        start_logits.append(self._score_none_token)
                        end_logits.append(self._score_none_token)
                        start_stdevs.append(self._stdev_none_token)
                        end_stdevs.append(self._stdev_none_token)
                    else:
                        start_logits.append(next(start_scores_iter))
                        end_logits.append(next(end_scores_iter))
                        start_stdevs.append(next(start_stdevs_iter))
                        end_stdevs.append(next(end_stdevs_iter))

                start_logits +=  [sys.float_info.min]*(128-len(start_logits))
                all_start_logits.append(np.array( start_logits))
                end_logits += [sys.float_info.min]*(128-len(end_logits))
                all_end_logits.append(np.array( end_logits))
                all_target_type_logits.append(self._target_type_scores[example_idx])

                start_stdevs +=  [sys.float_info.min]*(128-len(start_stdevs))
                all_start_stdevs.append(np.array( start_stdevs))
                end_stdevs += [sys.float_info.min]*(128-len(end_stdevs))
                all_end_stdevs.append(np.array( end_stdevs))
                all_query_passage_similarities.append(self._query_passage_similarities[example_idx][feat_idx])
                
        return adjusted_start_end_index, ( np.array(all_start_logits), np.array(all_end_logits), np.array(all_target_type_logits), np.array(all_start_stdevs), np.array(all_end_stdevs), np.array(all_query_passage_similarities))

    def test_confidence_feature_generation(self, eval_examples_and_features):
        eval_examples, eval_features = eval_examples_and_features
        postprocessor_class = ExtractivePostProcessor  
        scorer_type='weighted_sum_target_type_and_score_diff'

        expected_start_end_index, predictions = self._start_end_target_type_logits_stdevs(eval_examples, eval_features)
        
        postprocessor = postprocessor_class(k=5, n_best_size=3, max_answer_length=30,
                                            scorer_type=SupportedSpanScorers(scorer_type),
                                            single_context_multiple_passages=False,
                                            output_confidence_feature=True)        
        example_predictions = postprocessor.process(eval_examples, eval_features, predictions)
        
        for example_id, preds in  example_predictions.items():
            predicted = preds[0]
            expected = self._expected_predictions[example_id]
            expected_start_end = expected_start_end_index[example_id]
            ptargettype = int(np.argmax(predicted['target_type_logits']))
            assert (predicted['start_index'], predicted['end_index']) == expected_start_end
            assert predicted['passage_index']  == expected['passage_index'] 
            assert ptargettype  == expected['target_type'] 

            X = ConfidenceScorer.make_features(preds)
            assert len(X) == len(preds)
            assert preds[0]["confidence_score"] == preds[0]["normalized_span_answer_score"]

            
    def test_reference_prediction_overlap(self, eval_examples_and_features):
        eval_examples, eval_features = eval_examples_and_features
        postprocessor_class = ExtractivePostProcessor  
        scorer_type='weighted_sum_target_type_and_score_diff'

        expected_start_end_index, predictions = self._start_end_target_type_logits_stdevs(eval_examples, eval_features)
        
        postprocessor = postprocessor_class(k=5, n_best_size=3, max_answer_length=30,
                                            scorer_type=SupportedSpanScorers(scorer_type),
                                            single_context_multiple_passages=False,
                                            output_confidence_feature=True)        
        example_predictions = postprocessor.process(eval_examples, eval_features, predictions)
        
        for example_id, preds in  example_predictions.items():
            expected_start_end = expected_start_end_index[example_id]
            truth = [{"start_position":expected_start_end[0], "end_position":expected_start_end[1]}]
            prediction = {"start_position":preds[0]["start_index"], "end_position":preds[0]["end_index"]}
            assert ConfidenceScorer.reference_prediction_overlap(truth, prediction) == 1.0

        truth = [{"start_position":-1, "end_position":-1}]
        prediction = {"start_position":-1, "end_position":-1}
        assert ConfidenceScorer.reference_prediction_overlap(truth, prediction) == 1.0

        truth = [{"start_position":11, "end_position":20}]
        prediction = {"start_position":25, "end_position":30}
        assert ConfidenceScorer.reference_prediction_overlap(truth, prediction) == 0.0

        truth = [{"start_position":11, "end_position":20}]
        prediction = {"start_position":16, "end_position":30}
        assert ConfidenceScorer.reference_prediction_overlap(truth, prediction) == 0.4

        truth = [{"start_position":5, "end_position":10}, {"start_position":11, "end_position":20}]
        prediction = {"start_position":16, "end_position":30}
        assert ConfidenceScorer.reference_prediction_overlap(truth, prediction) == 0.4

        
    def test_instantiation(self, eval_examples_and_features):
        confidence_scorer = ConfidenceScorer()

        eval_examples, eval_features = eval_examples_and_features
        postprocessor_class = ExtractivePostProcessor  
        scorer_type='weighted_sum_target_type_and_score_diff'

        expected_start_end_index, predictions = self._start_end_target_type_logits_stdevs(eval_examples, eval_features)
        
        postprocessor = postprocessor_class(k=5, n_best_size=3, max_answer_length=30,
                                            scorer_type=SupportedSpanScorers(scorer_type),
                                            single_context_multiple_passages=False,
                                            output_confidence_feature=True)        
        example_predictions = postprocessor.process(eval_examples, eval_features, predictions)        
        for example_id, preds in  example_predictions.items():
            scores = confidence_scorer.predict_scores(preds)
            assert len(scores) == len(preds)
