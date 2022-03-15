import copy

import datasets
import pytest

from oneqa.mrc.metrics.nq_f1.nq_f1 import NQF1
from oneqa.mrc.types.target_type import TargetType
from tests.oneqa.mrc.common.base import UnitTest


class TestNQF1(UnitTest):
    @pytest.fixture(scope='session')
    def metric(self):
        return NQF1()

    @pytest.fixture()
    def n_annotators(self):
        return 5

    @pytest.fixture(scope='session')
    def gold_all_annotators_no_answer(self, n_annotators):
        return dict(start_position=[-1] * n_annotators,
                    end_positon=[-1] * n_annotators,
                    passage_index=[-1] * n_annotators,
                    yes_no_answer=[TargetType.NO_ANSWER] * n_annotators)

    @pytest.fixture(scope='session')
    def gold_all_annotators_same_span_answer(self, n_annotators):
        return dict(start_position=[17] * n_annotators,
                    end_positon=[42] * n_annotators,
                    passage_index=[1] * n_annotators,
                    yes_no_answer=[TargetType.NO_ANSWER] * n_annotators)

    @pytest.fixture(scope='session')
    def gold_all_annotators_same_passage_answer(self, n_annotators):
        return dict(start_position=[-1] * n_annotators,
                    end_positon=[-1] * n_annotators,
                    passage_index=[1] * n_annotators,
                    yes_no_answer=[TargetType.NO_ANSWER] * n_annotators)

    @pytest.fixture(scope='session')
    def gold_mixed_answer(self, n_annotators):
        return dict(start_position=[17, 128, -1, -1, -1],
                    end_positon=[42, 142, -1, -1, -1],
                    passage_index=[1, 0, -1, -1, -1],
                    yes_no_answer=[TargetType.NO_ANSWER] * n_annotators)

    @pytest.fixture(scope='session')
    def gold_bool_answer(self, n_annotators):
        return dict(start_position=[-1] * n_annotators,
                    end_positon=[-1] * n_annotators,
                    passage_index=[0] * n_annotators,
                    yes_no_answer=[TargetType.YES] * n_annotators)

    @pytest.fixture(scope='session')
    def pred_no_answer(self):
        return dict(start_position=-1,
                    end_positon=-1,
                    passage_index=-1,
                    yes_no_answer=TargetType.NO_ANSWER)

    @pytest.fixture(scope='session')
    def pred_correct_passage_answer(self):
        return dict(start_position=-1,
                    end_positon=-1,
                    passage_index=1,
                    yes_no_answer=TargetType.NO_ANSWER)

    @pytest.fixture(scope='session')
    def pred_incorrect_passage_answer(self):
        return dict(start_position=-1,
                    end_positon=-1,
                    passage_index=11,
                    yes_no_answer=TargetType.NO_ANSWER)

    @pytest.fixture(scope='session')
    def pred_correct_span_correct_passage_answer(self):
        return dict(start_position=17,
                    end_positon=42,
                    passage_index=1,
                    yes_no_answer=TargetType.NO_ANSWER)

    @pytest.fixture(scope='session')
    def pred_incorrect_span_correct_passage_answer(self):
        return dict(start_position=64,
                    end_positon=86,
                    passage_index=1,
                    yes_no_answer=TargetType.NO_ANSWER)

    @pytest.fixture(scope='session')
    def pred_incorrect_span_incorrect_passage_answer(self):
        return dict(start_position=64,
                    end_positon=86,
                    passage_index=2,
                    yes_no_answer=TargetType.NO_ANSWER)

    @pytest.fixture(scope='session')
    def pred_incorrect_span_correct_passage_answer_partial_overlap(self):
        return dict(start_position=24,
                    end_positon=42,
                    passage_index=1,
                    yes_no_answer=TargetType.NO_ANSWER)

    @pytest.fixture(scope='session')
    def pred_correct_bool_correct_passage_answer(self):
        return dict(start_position=-1,
                    end_positon=-1,
                    passage_index=1,
                    yes_no_answer=TargetType.YES)

    @pytest.fixture(scope='session')
    def pred_incorrect_bool_correct_passage_answer(self):
        return dict(start_position=-1,
                    end_positon=-1,
                    passage_index=1,
                    yes_no_answer=TargetType.NO)

    @pytest.fixture(scope='session')
    def all_correct_answers(self, gold_all_annotators_no_answer, pred_no_answer,
                            gold_all_annotators_same_span_answer, pred_correct_span_correct_passage_answer,
                            gold_all_annotators_same_passage_answer, pred_correct_passage_answer,
                            gold_mixed_answer,
                            gold_bool_answer, pred_correct_bool_correct_passage_answer):
        gold = [
            gold_all_annotators_no_answer,
            gold_all_annotators_same_span_answer,
            gold_all_annotators_same_passage_answer,
            gold_mixed_answer,
            gold_bool_answer
        ]

        pred = [
            pred_no_answer,
            pred_correct_span_correct_passage_answer,
            pred
        ]

    def test_instantiate_metric_from_class(self, metric):
        _ = metric

    def test_instantiate_metric_from_load_metric(self):
        _ = datasets.load_metric(NQF1.__file__)

    def test_all_correct(self):
        raise NotImplementedError

    def test_all_incorrect(self):
        raise NotImplementedError

    def test_some_correct(self):
        raise NotImplementedError
