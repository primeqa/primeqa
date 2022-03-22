from datasets import Dataset
import pytest
from pytest import raises
from transformers import AutoTokenizer

from oneqa.mrc.processors.preprocessors.default import DefaultPreProcessor
from oneqa.mrc.types.target_type import TargetType
from tests.oneqa.mrc.common.base import UnitTest
from tests.oneqa.mrc.common.parameterization import PARAMETERIZE_INVALID_SUBSAMPLING_PROBABILITIES


class TestDefaultPreProcessor(UnitTest):
    @pytest.fixture(scope='session')
    def train_examples_has_answer(self, train_examples):
        example_indices_has_answer = [i for i, t in enumerate(train_examples['target']) if t['passage_indices'][0] != -1]
        return train_examples.select(example_indices_has_answer)

    @pytest.fixture(scope='session')
    def train_examples_no_answer(self, train_examples):
        example_indices_no_answer = [i for i, t in enumerate(train_examples['target']) if t['passage_indices'][0] == -1]
        return train_examples.select(example_indices_no_answer)

    @pytest.fixture(scope='session')
    def invalid_type_train_examples(self):
        train_examples = Dataset.from_dict({'question': ['Who?'], 'context': [[-1]], 'target': [
            {'start_positions': [-1], 'end_positions': [-1], 'passage_indices': [-1], 'yes_no_answer': ['NONE']}]})
        return train_examples

    @pytest.fixture(scope='session')
    def invalid_name_train_examples(self, train_examples):
        return train_examples.rename_columns(dict(context='contextssss'))

    @pytest.fixture(scope='session')
    def invalid_name_eval_examples(self, eval_examples):
        return eval_examples.rename_columns(dict(context='contextssss'))

    @pytest.fixture(scope='session')
    def invalid_type_eval_examples(self, invalid_type_train_examples):
        return invalid_type_train_examples.remove_columns('target')

    @pytest.fixture(scope='session')
    def no_examples_train(self):
        return Dataset.from_dict({'question': [], 'context': [], 'target': [], 'example_id': []})

    @pytest.fixture(scope='session')
    def preprocessor_subsample_keep_all(self, tokenizer):
        return DefaultPreProcessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
            negative_sampling_prob_when_has_answer=1.,
            negative_sampling_prob_when_no_answer=1.,
        )

    @pytest.fixture(scope='session')
    def preprocessor_subsample_keep_none(self, tokenizer):
        return DefaultPreProcessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
            negative_sampling_prob_when_has_answer=0.,
            negative_sampling_prob_when_no_answer=0.,
        )

    @pytest.fixture(scope='session')
    def preprocessor_subsample_keep_no_answer(self, tokenizer):
        return DefaultPreProcessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
            negative_sampling_prob_when_has_answer=0.,
            negative_sampling_prob_when_no_answer=1.,
        )

    @pytest.fixture(scope='session')
    def preprocessor_subsample_keep_has_answer(self, tokenizer):
        return DefaultPreProcessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
            negative_sampling_prob_when_has_answer=1.,
            negative_sampling_prob_when_no_answer=0.,
        )

    @PARAMETERIZE_INVALID_SUBSAMPLING_PROBABILITIES
    def test_preprocessor_raises_value_error_on_invalid_subsampling_prob(
            self,
            tokenizer,
            negative_sampling_prob_when_has_answer,
            negative_sampling_prob_when_no_answer
    ):
        with raises(ValueError):
            _ = DefaultPreProcessor(
                tokenizer,
                stride=128,
                load_from_cache_file=False,
                negative_sampling_prob_when_has_answer=negative_sampling_prob_when_has_answer,
                negative_sampling_prob_when_no_answer=negative_sampling_prob_when_no_answer,
            )

    def test_adapt_dataset_with_train_examples(self, train_examples, preprocessor):
        assert preprocessor.adapt_dataset(train_examples, is_train=True) is train_examples

    def test_adapt_dataset_with_eval_examples(self, eval_examples, preprocessor):
        assert preprocessor.adapt_dataset(eval_examples, is_train=False) is eval_examples

    def test_train_preprocessing_runs_without_errors(self, train_examples, preprocessor):
        train_examples, train_features = preprocessor.process_train(train_examples)
        assert isinstance(train_examples, Dataset)
        assert isinstance(train_features, Dataset)
        expected_feature_columns = {'input_ids', 'attention_mask', 'example_idx', 'context_idx',
                                    'example_id', 'start_positions', 'end_positions', 'target_type'}
        expected_feature_columns.update(preprocessor._tokenizer.model_input_names)
        actual_feature_columns = set(train_features.column_names)
        assert actual_feature_columns == expected_feature_columns

    def test_process_train_feature_target_type_matches_position_labels(self, train_examples, preprocessor):
        _, train_features = preprocessor.process_train(train_examples)
        for i in range(train_features.num_rows):
            tt = train_features['target_type'][i]
            cls_index = train_features['input_ids'][i].index(preprocessor._tokenizer.cls_token_id)
            if tt in (TargetType.PASSAGE_ANSWER, TargetType.YES, TargetType.NO):
                # assert train_features['passage_indices'][i] != -1
                assert train_features['start_positions'][i] == cls_index
                assert train_features['end_positions'][i] == cls_index
            elif tt == TargetType.SPAN_ANSWER:
                # assert train_features['passage_indices'][i] != -1
                assert train_features['start_positions'][i] != cls_index
                assert train_features['end_positions'][i] != cls_index
            elif tt == TargetType.NO_ANSWER:
                # assert train_features['passage_indices'][i] == -1
                assert train_features['start_positions'][i] == cls_index
                assert train_features['end_positions'][i] == cls_index
            else:
                raise ValueError(f"Unexpected target type {tt}")

    def test_all_features_positive_when_subsample_keep_none(self, train_examples, preprocessor_subsample_keep_none):
        train_examples, train_features = preprocessor_subsample_keep_none.process_train(train_examples)
        assert all(tt != TargetType.NO_ANSWER for tt in train_features['target_type'])
        assert train_examples.num_rows > train_features.num_rows

    def test_some_features_negative_when_subsample_keep_all(self, train_examples, preprocessor_subsample_keep_all):
        train_examples, train_features = preprocessor_subsample_keep_all.process_train(train_examples)
        assert any(tt == TargetType.NO_ANSWER for tt in train_features['target_type'])
        assert train_examples.num_rows < train_features.num_rows

    def test_raises_value_error_when_subsampling_removes_all_features(
            self, train_examples_no_answer, preprocessor_subsample_keep_none):
        with raises(ValueError):
            _, _ = preprocessor_subsample_keep_none.process_train(train_examples_no_answer)

    def test_raises_value_error_when_no_examples(self, no_examples_train, preprocessor):
        with raises(ValueError):
            _, _ = preprocessor.process_train(no_examples_train)

    def test_subsample_keeps_features_from_has_answer_when_only_keeping_has_answer_negatives(
            self, train_examples_has_answer, preprocessor_subsample_keep_has_answer):
        train_examples_has_answer, train_features = preprocessor_subsample_keep_has_answer.process_train(
            train_examples_has_answer)
        for fi in range(train_features.num_rows):
            ei = train_features['example_idx'][fi]
            example_has_answer = train_examples_has_answer['target'][ei]['passage_indices'][0] != -1
            tt = train_features['target_type'][fi]
            negative_feature = tt == TargetType.NO_ANSWER
            if negative_feature:
                assert example_has_answer

    def test_subsample_keeps_features_from_no_answer_when_only_keeping_no_answer_negatives(
            self, train_examples_no_answer, preprocessor_subsample_keep_no_answer):
        train_examples_no_answer, train_features = preprocessor_subsample_keep_no_answer.process_train(
            train_examples_no_answer)
        for fi in range(train_features.num_rows):
            ei = train_features['example_idx'][fi]
            example_has_answer = train_examples_no_answer['target'][ei]['passage_indices'][0] != -1
            tt = train_features['target_type'][fi]
            negative_feature = tt == TargetType.NO_ANSWER
            if negative_feature:
                assert not example_has_answer

    def test_eval_preprocessing_runs_without_errors(self, eval_examples, preprocessor):
        eval_examples, eval_features = preprocessor.process_eval(eval_examples)
        assert isinstance(eval_examples, Dataset)
        assert isinstance(eval_features, Dataset)

    def test_cannot_adapt_dataset_with_invalid_train_schema_names(self, preprocessor, invalid_name_train_examples):
        with raises(ValueError):
            _ = preprocessor.adapt_dataset(invalid_name_train_examples, is_train=True)

    def test_cannot_adapt_dataset_with_invalid_train_schema_types(self, preprocessor, invalid_type_train_examples):
        with raises(ValueError):
            _ = preprocessor.adapt_dataset(invalid_type_train_examples, is_train=True)

    def test_cannot_adapt_dataset_with_invalid_train_schema_missing_target(self, preprocessor, eval_examples):
        with raises(ValueError):
            _ = preprocessor.adapt_dataset(eval_examples, is_train=True)

    def test_cannot_adapt_dataset_with_invalid_eval_schema_names(self, preprocessor, invalid_name_eval_examples):
        with raises(ValueError):
            _ = preprocessor.adapt_dataset(invalid_name_eval_examples, is_train=False)

    def test_cannot_adapt_dataset_with_invalid_eval_schema_types(self, preprocessor, invalid_type_eval_examples):
        with raises(ValueError):
            _ = preprocessor.adapt_dataset(invalid_type_eval_examples, is_train=False)
