import datasets
import pytest

from primeqa.mrc.metrics.tydi_f1.tydi_f1 import TyDiF1
from primeqa.mrc.data_models.target_type import TargetType
from tests.primeqa.mrc.common.base import UnitTest


class TestTyDiF1(UnitTest):
    @pytest.fixture(scope='session')
    def metric(self):
        return TyDiF1()

    @pytest.fixture(scope='session')
    def n_annotators(self):
        return 5

    @pytest.fixture(scope='session')
    def references_and_predictions(self, n_annotators):
        references = [
            [dict(start_position=17, end_position=42, passage_index=1,
                  yes_no_answer=TargetType.NO_ANSWER, example_id='a', language='finnish', document_plaintext='', question='')] * n_annotators,
            [dict(start_position=-1, end_position=-1, passage_index=2,
                  yes_no_answer=TargetType.YES, example_id='b', language='finnish', document_plaintext='', question='')] * n_annotators,
            [dict(start_position=17, end_position=42, passage_index=1,
                  yes_no_answer=TargetType.NO_ANSWER, example_id='c', language='swahili', document_plaintext='', question='')] * 2 +
            [dict(start_position=-1, end_position=-1, passage_index=-1,
                  yes_no_answer=TargetType.NO_ANSWER, example_id='c', language='swahili', document_plaintext='', question='')] * (n_annotators - 2),
            [dict(start_position=-1, end_position=-1, passage_index=-1,
                  yes_no_answer=TargetType.NO_ANSWER, example_id='d', language='swahili', document_plaintext='', question='')] * n_annotators,
            [dict(start_position=-1, end_position=-1, passage_index=1,
                  yes_no_answer=TargetType.NO_ANSWER, example_id='e', language='swahili', document_plaintext='', question='')] * n_annotators,
            [dict(start_position=17, end_position=42, passage_index=1,
                  yes_no_answer=TargetType.NO_ANSWER, example_id='f', language='thai', document_plaintext='', question='')] * n_annotators,
            [dict(start_position=17, end_position=42, passage_index=1,
                  yes_no_answer=TargetType.NO_ANSWER, example_id='g', language='thai', document_plaintext='', question='')] * n_annotators,
            [dict(start_position=17, end_position=42, passage_index=1,
                  yes_no_answer=TargetType.NO_ANSWER, example_id='h', language='korean', document_plaintext='', question='')] * n_annotators,
            [dict(start_position=17, end_position=42, passage_index=1,
                  yes_no_answer=TargetType.NO_ANSWER, example_id='i', language='korean', document_plaintext='', question='')] * n_annotators,

        ]
        predictions = [
            dict(start_position=17, end_position=42, passage_index=1,
                 yes_no_answer=TargetType.NO_ANSWER, example_id='a', confidence_score=10.),
            dict(start_position=-1, end_position=-1, passage_index=2,
                 yes_no_answer=TargetType.YES, example_id='b', confidence_score=9.9),
            dict(start_position=17, end_position=42, passage_index=1,
                 yes_no_answer=TargetType.NO_ANSWER, example_id='c', confidence_score=9.8),
            dict(start_position=17, end_position=42, passage_index=1,
                 yes_no_answer=TargetType.NO_ANSWER, example_id='d', confidence_score=-10.),
            dict(start_position=17, end_position=42, passage_index=1,
                 yes_no_answer=TargetType.NO_ANSWER, example_id='e', confidence_score=-5.),
            dict(start_position=24, end_position=44, passage_index=1,
                 yes_no_answer=TargetType.NO_ANSWER, example_id='f', confidence_score=10.1),
            dict(start_position=24, end_position=44, passage_index=4,
                 yes_no_answer=TargetType.NO_ANSWER, example_id='g', confidence_score=10.1),
            dict(start_position=111, end_position=141, passage_index=1,
                 yes_no_answer=TargetType.NO_ANSWER, example_id='h', confidence_score=10.1),
            dict(start_position=111, end_position=141, passage_index=4,
                 yes_no_answer=TargetType.NO_ANSWER, example_id='i', confidence_score=10.1),
        ]
        return dict(references=references, predictions=predictions)

    def test_instantiate_metric_from_class(self, metric):
        _ = metric

    def test_instantiate_metric_from_load_metric(self):
        from primeqa.mrc.metrics.tydi_f1 import tydi_f1
        _ = datasets.load_metric(tydi_f1.__file__)

    def test_compute_metric(self, metric, references_and_predictions):
        metric.add_batch(**references_and_predictions)
        actual_metric_values = metric.compute()
        expected_metric_values = {
            "avg_passage_f1": 0.75, "avg_passage_recall": 0.75, "avg_passage_precision": 0.75,
            "avg_minimal_f1": 0.7, "avg_minimal_recall": 0.7, "avg_minimal_precision": 0.7
        }
        assert actual_metric_values == expected_metric_values
