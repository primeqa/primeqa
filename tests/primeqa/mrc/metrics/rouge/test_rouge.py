import datasets
import pytest

from tests.primeqa.mrc.common.base import UnitTest
from primeqa.mrc.metrics.rouge.rouge import ROUGE

class TestROUGE(UnitTest):
    
    @pytest.fixture(scope='session')
    def metric(self):
        return ROUGE()
    
    @pytest.fixture(scope='session')
    def references_and_predictions(self):
        references = [
            dict(id='1', answers=["hello there"]),
            dict(id='2', answers=["general kenobi"])
        ]
        predictions = [
            dict(id='1', prediction_text='hello there'),
            dict(id='2', prediction_text='general kenobi')
        ]
        return dict(references=references, predictions=predictions)
    
    def test_instantiate_metric_from_class(self, metric):
            _ = metric

    def test_instantiate_metric_from_load_metric(self):
        from primeqa.mrc.metrics.rouge import rouge
        _ = datasets.load_metric(rouge.__file__)
        
    def test_compute_metric(self, metric, references_and_predictions):
        metric.add_batch(**references_and_predictions)
        actual_metric_values = metric.compute()
        expected_metric_values = {'kilt_rougeL': 100.00, 'google_rougeL': 100.00, 'gen_len': 1.0}
        assert actual_metric_values == expected_metric_values