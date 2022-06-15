import pytest
from pytest import raises
from tests.primeqa.mrc.common.base import UnitTest
from primeqa.boolqa.score_normalizer.score_normalizer import ScoreNormalizer
import numpy as np

class TestScoreNormalizer(UnitTest):
    
    @pytest.fixture(scope='session')
    def score_normalizer(self):
        return ScoreNormalizer()
    
    @pytest.fixture(scope='session')
    def qa_unnormalized_predictions(self):
            prediction = dict(
                example_id = "308f64d3-2794-410c-b2d5-10472b7e6661",
                cls_score = 8.59765625,
                start_logit = 4.26171875,
                end_logit = 3.48828125,
                span_answer_score = 1.4951171875,
                start_index = 304,
                end_index = 306,
                span_answer_start_position = 6703,
                span_answer_end_position = 6715,
                span_answer_text = "1200-luvulla",
                passage_index = 13,
                target_type_logits = [
                    2.255859375,
                    3.837890625,
                    2.96875,
                    -3.71875,
                    -4.203125
                ],
                yes_no_answer = 0,
                confidence_score = 0.1801623883528011,
                language = "finnish",
                rank = 0,
                question_type_pred = "short_answer",
                question_type_scores = dict(
                    boolean = -2.9972593784332275, 
                    short_answer = 3.7809150218963623),
                question_type_conf = 3.780915,
                boolean_answer_pred = "yes",
                boolean_answer_scores = dict(
                    no = -4.663113117218018,
                    no_answer = 6.284051895141602,
                    yes = -0.9193163514137268),
                boolean_answer_conf = -0.91931635
            )
            return prediction
     
    @pytest.fixture(scope='session')
    def qtc_is_boolean_label(self):
        return "boolean"
    
    _expected_features = np.array([[0.0, 4.26171875, 3.48828125, 2.255859375]])
    
    _expected_prediction = dict(example_id = "308f64d3-2794-410c-b2d5-10472b7e6661",
                                 start_position = 6703, 
                                 end_position = 6715, 
                                 passage_index = 13,
                                 yes_no_answer = 0)
    
    def test_create_features(self, score_normalizer,qa_unnormalized_predictions,qtc_is_boolean_label):
        qtc_boolean_label = qtc_is_boolean_label
        qa_pred = qa_unnormalized_predictions
        features = score_normalizer.create_features(qtc_boolean_label, qa_pred)
        assert np.array_equal(features, self._expected_features)
        
    
    def test_create_prediction(self,score_normalizer,qa_unnormalized_predictions,qtc_is_boolean_label):
        qtc_boolean_label = qtc_is_boolean_label
        qa_pred = qa_unnormalized_predictions
        prediction = score_normalizer.create_prediction(qtc_boolean_label, qa_pred)
        assert prediction == self._expected_prediction