import copy
import functools

import datasets
import pytest

from oneqa.mrc.metrics.tydi_f1.tydi_f1 import TyDiF1
from oneqa.mrc.types.target_type import TargetType
from tests.oneqa.mrc.common.base import UnitTest


def return_name(f):
    @functools.wraps(f)
    def inner(*args, **kwargs):
        return f.__name__
    return inner


class TestTyDiF1(UnitTest):
    @pytest.fixture(scope='session')
    def metric(self):
        return TyDiF1()

    @pytest.fixture(scope='session')
    def n_annotators(self):
        return 5

    @pytest.fixture(scope='session')
    @return_name
    def example_id_for_no_answer(self):
        pass

    @pytest.fixture(scope='session')
    @return_name
    def example_id_for_span_answer(self):
        pass

    @pytest.fixture(scope='session')
    @return_name
    def example_id_for_passage_answer(self):
        pass

    @pytest.fixture(scope='session')
    @return_name
    def example_id_for_mixed_answer(self):
        pass

    @pytest.fixture(scope='session')
    @return_name
    def example_id_for_bool_answer(self):
        pass

    @pytest.fixture(scope='session')
    def high_confidence_score(self):
        return 10.

    @pytest.fixture(scope='session')
    def low_confidence_score(self):
        return -10.

    @pytest.fixture(scope='session')
    def intermediate_confidence_score(self):
        return 0.

    @pytest.fixture(scope='session')
    def gold_all_annotators_no_answer(self, n_annotators, example_id_for_no_answer):
        return dict(start_position=[17] * n_annotators,
                    end_position=[42] * n_annotators,
                    passage_index=[1] * n_annotators,
                    yes_no_answer=[TargetType.NO_ANSWER] * n_annotators,
                    example_id=[example_id_for_no_answer] * n_annotators,
                    language=['english'] * n_annotators)

    @pytest.fixture(scope='session')
    def gold_all_annotators_same_span_answer(self, n_annotators, example_id_for_span_answer):
        return dict(start_position=[17] * n_annotators,
                    end_position=[42] * n_annotators,
                    passage_index=[1] * n_annotators,
                    yes_no_answer=[TargetType.NO_ANSWER] * n_annotators,
                    example_id=[example_id_for_span_answer] * n_annotators,
                    language=['english'] * n_annotators)

    @pytest.fixture(scope='session')
    def gold_all_annotators_same_passage_answer(self, n_annotators, example_id_for_passage_answer):
        return dict(start_position=[-1] * n_annotators,
                    end_position=[-1] * n_annotators,
                    passage_index=[1] * n_annotators,
                    yes_no_answer=[TargetType.NO_ANSWER] * n_annotators,
                    example_id=[example_id_for_passage_answer] * n_annotators,
                    language=['english'] * n_annotators)

    @pytest.fixture(scope='session')
    def gold_mixed_answer(self, n_annotators, example_id_for_mixed_answer):
        return dict(start_position=[17, 128, -1, -1, -1],
                    end_position=[42, 142, -1, -1, -1],
                    passage_index=[1, 0, -1, -1, -1],
                    yes_no_answer=[TargetType.NO_ANSWER] * n_annotators,
                    example_id=[example_id_for_mixed_answer] * n_annotators,
                    language=['english'] * n_annotators)

    @pytest.fixture(scope='session')
    def gold_bool_answer(self, n_annotators, example_id_for_bool_answer):
        return dict(start_position=[-1] * n_annotators,
                    end_position=[-1] * n_annotators,
                    passage_index=[0] * n_annotators,
                    yes_no_answer=[TargetType.YES] * n_annotators,
                    example_id=[example_id_for_bool_answer] * n_annotators,
                    language=['english'] * n_annotators)

    @pytest.fixture(scope='session')
    def pred_no_answer(self, example_id_for_no_answer, low_confidence_score):
        return dict(start_position=11,
                    end_position=14,
                    passage_index=11,
                    yes_no_answer=TargetType.NO_ANSWER,
                    example_id=example_id_for_no_answer,
                    confidence_score=low_confidence_score)

    @pytest.fixture(scope='session')
    def pred_correct_passage_answer(self, example_id_for_passage_answer, high_confidence_score):
        return dict(start_position=-1,
                    end_position=-1,
                    passage_index=1,
                    yes_no_answer=TargetType.NO_ANSWER,
                    example_id=example_id_for_passage_answer,
                    confidence_score=high_confidence_score)

    @pytest.fixture(scope='session')
    def pred_incorrect_passage_answer(self):
        return dict(start_position=-1,
                    end_position=-1,
                    passage_index=11,
                    yes_no_answer=TargetType.NO_ANSWER)

    @pytest.fixture(scope='session')
    def pred_correct_span_correct_passage_answer(self, example_id_for_span_answer, high_confidence_score):
        return dict(start_position=17,
                    end_position=42,
                    passage_index=1,
                    yes_no_answer=TargetType.NO_ANSWER,
                    example_id=example_id_for_span_answer,
                    confidence_score=high_confidence_score)

    @pytest.fixture(scope='session')
    def pred_incorrect_span_correct_passage_answer(self):
        return dict(start_position=64,
                    end_position=86,
                    passage_index=1,
                    yes_no_answer=TargetType.NO_ANSWER)

    @pytest.fixture(scope='session')
    def pred_incorrect_span_incorrect_passage_answer(self):
        return dict(start_position=64,
                    end_position=86,
                    passage_index=2,
                    yes_no_answer=TargetType.NO_ANSWER)

    @pytest.fixture(scope='session')
    def pred_incorrect_span_correct_passage_answer_partial_overlap(self):
        return dict(start_position=24,
                    end_position=42,
                    passage_index=1,
                    yes_no_answer=TargetType.NO_ANSWER)

    @pytest.fixture(scope='session')
    def pred_correct_bool_correct_passage_answer(self, example_id_for_bool_answer, high_confidence_score):
        return dict(start_position=-1,
                    end_position=-1,
                    passage_index=1,
                    yes_no_answer=TargetType.YES,
                    example_id=example_id_for_bool_answer,
                    confidence_score=high_confidence_score)

    @pytest.fixture(scope='session')
    def pred_incorrect_bool_correct_passage_answer(self):
        return dict(start_position=-1,
                    end_position=-1,
                    passage_index=1,
                    yes_no_answer=TargetType.NO)

    @pytest.fixture(scope='session')
    def all_correct_answers(self,
                            gold_all_annotators_no_answer, pred_no_answer,
                            gold_all_annotators_same_span_answer, pred_correct_span_correct_passage_answer,
                            gold_all_annotators_same_passage_answer, pred_correct_passage_answer,
                            gold_bool_answer, pred_correct_bool_correct_passage_answer):
        gold = [
            gold_all_annotators_no_answer,
            gold_all_annotators_same_span_answer,
            gold_all_annotators_same_passage_answer,
            gold_bool_answer
        ]

        pred = [
            pred_no_answer,
            pred_correct_span_correct_passage_answer,
            pred_correct_passage_answer,
            pred_correct_bool_correct_passage_answer
        ]

        return dict(references=gold, predictions=pred)

    def test_instantiate_metric_from_class(self, metric):
        _ = metric

    def test_instantiate_metric_from_load_metric(self):
        _ = datasets.load_metric(TyDiF1.__file__)

    def test_all_correct(self, metric, all_correct_answers):
        metric.add_batch(**all_correct_answers)
        results = metric.compute()
        raise NotImplementedError

    def test_all_incorrect(self):
        raise NotImplementedError

    def test_some_correct(self):
        raise NotImplementedError
