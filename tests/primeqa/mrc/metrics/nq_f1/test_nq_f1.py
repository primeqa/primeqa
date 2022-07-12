import datasets
import pytest

from primeqa.mrc.metrics.nq_f1.nq_f1 import NQF1
from primeqa.mrc.metrics.nq_f1.eval_utils import NQLabel, NQSpan
from primeqa.mrc.data_models.target_type import TargetType
from tests.primeqa.mrc.common.base import UnitTest


class TestNQF1(UnitTest):
    @pytest.fixture(scope='session')
    def metric(self):
        return NQF1()

    @pytest.fixture(scope='session')
    def n_annotators(self):
        return 5

    @pytest.fixture(scope='session')
    def references_and_predictions(self, n_annotators):
        references = [
            [dict(start_position=-1, end_position=-1, passage_index=-1,
                  yes_no_answer=TargetType.NO_ANSWER, example_id='6092', language='english')] * 4 +
            [dict(start_position=44318, end_position=44657, passage_index=0,
                  yes_no_answer=TargetType.NO_ANSWER, example_id='6092', language='english')],

            [dict(start_position=-1, end_position=-1, passage_index=-1,
                  yes_no_answer=TargetType.NO, example_id='7647', language='english')] * n_annotators,

            [dict(start_position=-1, end_position=-1, passage_index=-1,
                  yes_no_answer=TargetType.YES, example_id='9484', language='english')] +
            [dict(start_position=37907, end_position=37946, passage_index=1,
                  yes_no_answer=TargetType.NO_ANSWER, example_id='9484', language='english')] * 2 +
            [dict(start_position=37907, end_position=37930, passage_index=1,
                  yes_no_answer=TargetType.NO_ANSWER, example_id='9484', language='english')] * 2
        ]
        predictions = [
            dict(start_position=50425, end_position=50485, passage_index=4,
                 yes_no_answer=TargetType.YES, example_id='6092', confidence_score=1.4763),
            dict(start_position=59297, end_position=59331, passage_index=23,
                 yes_no_answer=TargetType.NO, example_id='7647', confidence_score=3.4773),
            dict(start_position=37907, end_position=37930, passage_index=1,
                 yes_no_answer=TargetType.NO_ANSWER, example_id='9484', confidence_score=6.5318),
        ]
        return dict(references=references, predictions=predictions)

    def test_instantiate_metric_from_class(self, metric):
        _ = metric

    def test_instantiate_metric_from_load_metric(self):
        from primeqa.mrc.metrics.nq_f1 import nq_f1
        _ = datasets.load_metric(nq_f1.__file__)

    def test_passage_index_to_long_span(self, references_and_predictions):
        for reference in references_and_predictions["references"]:
            for ref in reference:
                if ref['passage_index'] != -1:
                    assert isinstance(NQF1._passage_index_to_long_span(ref['passage_index']), NQSpan)
                else:
                    assert NQF1._passage_index_to_long_span(ref['passage_index']) == NQSpan.null_span()

    def test_compute_metric(self, metric, references_and_predictions):
        metric.add_batch(**references_and_predictions)
        actual_metric_values = metric.compute()
        expected_metric_values = {
            "long_answer_f1": 0.5, "long_answer_n": 3, "long_answer_precision": 1.0/3, "long_answer_recall": 1.0,
            "short_answer_f1": 0.8, "short_answer_n": 3, "short_answer_precision": 2.0/3, "short_answer_recall": 1.0
        }
        assert actual_metric_values == expected_metric_values

