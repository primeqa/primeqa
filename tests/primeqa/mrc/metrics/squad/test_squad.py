import datasets
import pytest

from tests.primeqa.mrc.common.base import UnitTest
from primeqa.mrc.metrics.squad.squad import SQUAD


class TestSQUAD(UnitTest):
    @pytest.fixture(scope='session')
    def metric(self):
        return SQUAD()

    @pytest.fixture(scope='session')
    def references_and_predictions(self):
        references = [
            dict(id='56be4db0acb8001400a502ec', answers=dict(text=['Denver Broncos'], answer_start=[177])),
            dict(id='56be4db0acb8001400a502ed', answers=dict(text=['Carolina Panthers'], answer_start=[249])),
            dict(id='56be4db0acb8001400a502ee', answers=dict(text=["Levi's Stadium"], answer_start=[355])), 
        ]
        predictions = [
            dict(id='56be4db0acb8001400a502ec', prediction_text='Denver Broncos'),
            dict(id='56be4db0acb8001400a502ed', prediction_text='Carolina Panthers'),
            dict(id='56be4db0acb8001400a502ee', prediction_text="Levi's Stadium in the San Francisco Bay Area"),
        ]
        return dict(references=references, predictions=predictions)

    def test_instantiate_metric_from_class(self, metric):
        _ = metric

    def test_instantiate_metric_from_load_metric(self):
        from primeqa.mrc.metrics.squad import squad
        _ = datasets.load_metric(squad.__file__)

    def test_compute_metric(self, metric, references_and_predictions):
        metric.add_batch(**references_and_predictions)
        actual_metric_values = metric.compute()
        expected_metric_values = {'exact_match': 66.66666666666667, 'f1': 81.48148148148148}
        assert actual_metric_values == expected_metric_values
