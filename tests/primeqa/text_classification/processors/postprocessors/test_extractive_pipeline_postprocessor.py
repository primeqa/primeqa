from datasets import Dataset
import pytest
import sys
from pytest import raises
from itertools import groupby
from operator import itemgetter
import numpy as np
from primeqa.text_classification.processors.postprocessors.extractive import ExtractivePipelinePostProcessor
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from primeqa.mrc.data_models.target_type import TargetType
from tests.primeqa.mrc.common.base import UnitTest
from tests.primeqa.mrc.common.parameterization import PARAMETERIZE_INVALID_SUBSAMPLING_PROBABILITIES


class TestExtractivePipelinePostProcessor(UnitTest):

    @pytest.fixture(scope='session')
    def mock_logits_expected_predictions(self):
        _start_score_of_cls_token = 0
        _end_score_of_cls_token = 0
        _score_none_token = -10
        example1_start_scores = [ [0.5, 0.3, 0.2, 0.1, 0.1], 
                                [1.5, 0.7, 0.2, 0.1, 0.1],
                    ]
        example1_end_scores = [ [0.5, 0.3, 0, 0, 0], 
                                [1.5, 0.7, 0, 0, 0],
                    ]
        example2_start_scores = [ [-1, 2, 1, -3, -2.5, -5, -4, -2, -2, -1, -1.5, -2.5],  
                                [-3, -2, -5, -2,-1, -1, -3, -2, -4, -5, -1.5, -2.5], 
                                [-3, -2, -5, -2,-1, -1, -3, -2, -4, -5, -1.5, -2.5], 
                    ]
        example2_end_scores = [   [-1, 1, 1.5, -3, -2.5, -5, -4, -2, -2, -1, -1.5, -2.5], 
                                [-3, -2, -5, -2,-1, -1, -3, -2, -4, -5, -1.5, -2.5],
                                [-3, -2, -5, -2,-1, -1, -3, -2, -4, -5, -1.5, -2.5],
                    ]

        _start_scores = [example1_start_scores, example2_start_scores]
        _end_scores = [ example1_end_scores, example2_end_scores]

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
            'span_answer_text': 'Bob',
            'question': 'Who walked the dog?',
            'language': 'UNKNOWN'
            },
            'bar-123': {
            'start_index': 1, # 9,
            'end_index': 2, #10,
            'passage_index': 0,
            'target_type' : int(TargetType.NO_ANSWER),
            'span_answer_text': 'quick brown',
            'question': 'What time is it?',
            'language': 'UNKNOWN'
            }
        }

        return {
            'start_score_of_cls_token': _start_score_of_cls_token,
            'end_score_of_cls_token': _end_score_of_cls_token,
            'score_none_token': _score_none_token,
            'start_scores': _start_scores,
            'end_scores': _end_scores,
            'target_type_scores': _target_type_scores,
            'expected_predictions': _expected_predictions
        }
        
    
    # TODO this is the same code as 
    # tests.primeqa.mrc.processors.postprocessors.test_extractive_postprocessor.TestExtractivePostProcessor    
    # but pytest doesn't seem to like using inheritance to de-duplicate code - it gives extra logging messages
    # and appararently runs everything twice!?
    def _start_end_target_type_logits(self, examples, features, mock_logits_expected_predictions):

        all_start_logits = []
        all_end_logits = []
        all_target_type_logits = []
        adjusted_start_end_index = {}

        features_itr = groupby(features, key=itemgetter('example_idx'))
        for example in examples:
            example_id = example['example_id']
            expected = mock_logits_expected_predictions['expected_predictions'][example_id]

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
                start_scores_iter = iter(mock_logits_expected_predictions['start_scores'][example_idx][feat_idx])

                end_scores_iter = iter(mock_logits_expected_predictions['end_scores'][example_idx][feat_idx])
                for i, offset in enumerate(offset_mapping):
                    if i == 0:
                        start_logits.append(mock_logits_expected_predictions['start_score_of_cls_token'])
                        end_logits.append(mock_logits_expected_predictions['end_score_of_cls_token'])
                    elif offset == None:
                        start_logits.append(mock_logits_expected_predictions['score_none_token'])
                        end_logits.append(mock_logits_expected_predictions['score_none_token'])
                    else:
                        start_logits.append(next(start_scores_iter))
                        end_logits.append(next(end_scores_iter))

                start_logits +=  [sys.float_info.min]*(128-len(start_logits))
                all_start_logits.append(np.array( start_logits))
                end_logits += [sys.float_info.min]*(128-len(end_logits))
                all_end_logits.append(np.array( end_logits))
                all_target_type_logits.append(mock_logits_expected_predictions['target_type_scores'][example_idx])

        return adjusted_start_end_index, ( np.array(all_start_logits), np.array(all_end_logits), np.array(all_target_type_logits) )

    def test_post_processor_has_examples_and_features(self, eval_examples_and_features, mock_logits_expected_predictions):
        eval_examples, eval_features = eval_examples_and_features
        postprocessor_class = ExtractivePipelinePostProcessor  
        scorer_type='weighted_sum_target_type_and_score_diff'

        expected_start_end_index, predictions = self._start_end_target_type_logits(eval_examples, eval_features, mock_logits_expected_predictions)
        
        postprocessor = postprocessor_class(k=5, n_best_size=3, max_answer_length=30,
                                            scorer_type=SupportedSpanScorers(scorer_type),
                                            single_context_multiple_passages=False)
        example_predictions = postprocessor.process(eval_examples, eval_features, predictions)
        print('example_predictions='+str(example_predictions))
        for example_id, preds in  example_predictions.items():
            predicted = preds[0]
            expected = mock_logits_expected_predictions['expected_predictions'][example_id]
            expected_start_end = expected_start_end_index[example_id]
            ptargettype = int(np.argmax(predicted['target_type_logits']))
            assert (predicted['question'] == expected['question'] )
            assert (predicted['language'] == expected['language'] )
            assert (predicted['start_index'], predicted['end_index']) == expected_start_end
            assert predicted['passage_index']  == expected['passage_index'] 
            assert ptargettype  == expected['target_type'] 




    
