from datasets import Dataset
import pytest
from pytest import raises
from transformers import AutoTokenizer

from oneqa.mrc.processors.preprocessors.default import DefaultPreProcessor
from oneqa.mrc.types.target_type import TargetType

_INVALID_PROBS = [-0.01, 1.01]


class TestDefaultPreProcessor:

    @pytest.fixture()
    def train_examples(self):
        question = ["Who walked the dog?", "What time is it?"]
        context = [["Alice walks the cat", "Bob walks the dog"],
                   ["The quick brown fox jumps over the lazy dog", "Glenn the otter lives at the aquarium", "Go"]]
        start_positions = [[0], [-1]]
        end_positions = [[2], [-1]]
        passage_indices = [[1], [-1]]
        yes_no_answer = ["NONE", "NONE"]
        examples_dict = dict(question=question, context=context,
                             target=[dict(start_positions=s, end_positions=e, passage_indices=p, yes_no_answer=yn)
                                     for s, e, p, yn in
                                     zip(start_positions, end_positions, passage_indices, yes_no_answer)])
        examples_dataset = Dataset.from_dict(examples_dict)
        return examples_dataset

    # @pytest.fixture()
    # def train_examples_no_answer(self, train_examples):
    #     example_indices_has_answer = [i for i, t in enumerate(train_examples['target']) if t['passage_indices'][0] != -1]
    #     return train_examples.select(example_indices_has_answer)
    #
    # @pytest.fixture()
    # def train_examples_has_answer(self, train_examples):
    #     example_indices_no_answer = [i for i, t in enumerate(train_examples['target']) if t['passage_indices'][0] == -1]
    #     return train_examples.select(example_indices_no_answer)

    @pytest.fixture()
    def eval_examples(self, train_examples):
        return train_examples.remove_columns("target")

    @pytest.fixture()
    def tokenizer(self):
        return AutoTokenizer.from_pretrained('roberta-base')

    @pytest.fixture()
    def preprocessor(self, tokenizer):
        return DefaultPreProcessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
            negative_sampling_prob_when_has_answer=0.5,
            negative_sampling_prob_when_no_answer=0.5,
        )

    @pytest.fixture()
    def preprocessor_subsample_keep_all(self, tokenizer):
        return DefaultPreProcessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
            negative_sampling_prob_when_has_answer=1.,
            negative_sampling_prob_when_no_answer=1.,
        )

    @pytest.fixture()
    def preprocessor_subsample_keep_none(self, tokenizer):
        return DefaultPreProcessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
            negative_sampling_prob_when_has_answer=0.,
            negative_sampling_prob_when_no_answer=0.,
        )

    @pytest.fixture()
    def preprocessor_subsample_keep_no_answer(self, tokenizer):
        return DefaultPreProcessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
            negative_sampling_prob_when_has_answer=0.,
            negative_sampling_prob_when_no_answer=1.,
        )

    @pytest.fixture()
    def preprocessor_subsample_keep_has_answer(self, tokenizer):
        return DefaultPreProcessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
            negative_sampling_prob_when_has_answer=1.,
            negative_sampling_prob_when_no_answer=0.,
        )

    @pytest.mark.parametrize(["negative_sampling_prob_when_has_answer", "negative_sampling_prob_when_no_answer"],
                             [(p1, p2) for p1 in _INVALID_PROBS for p2 in _INVALID_PROBS])
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

    def test_adapt_examples(self, train_examples, preprocessor):
        assert preprocessor.adapt_dataset(train_examples) is train_examples

    def test_train_preprocessing_runs_without_errors(self, train_examples, preprocessor):
        train_examples, train_features = preprocessor.process_train(train_examples)
        assert isinstance(train_examples, Dataset)
        assert isinstance(train_features, Dataset)
        assert set(train_features.column_names) == {'input_ids', 'attention_mask', 'example_idx', 'context_idx',
                                                    'example_id', 'start_positions', 'end_positions', 'target_type'}

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

    # def test_subsample_keep_features_from_no_answer_examples_only(self, train_examples_no_answer,
    #                                                               preprocessor_subsample_keep_no_answer):
    #     train_examples_no_answer, train_features = preprocessor_subsample_keep_no_answer.process_train(train_examples_no_answer)
    #     raise NotImplementedError
    #
    # def test_subsample_keep_features_from_has_answer_examples_only(self, train_examples_has_answer,
    #                                                                preprocessor_subsample_keep_has_answer):
    #     train_examples_has_answer, train_features = preprocessor_subsample_keep_has_answer.process_train(train_examples_has_answer)
    #     raise NotImplementedError

    def test_subsample_keep_features_from_no_answer_examples_only(self, train_examples,
                                                                  preprocessor_subsample_keep_no_answer):
        train_examples, train_features = preprocessor_subsample_keep_no_answer.process_train(train_examples)
        raise NotImplementedError

    def test_subsample_keep_features_from_has_answer_examples_only(self, train_examples,
                                                                   preprocessor_subsample_keep_has_answer):
        train_examples, train_features = preprocessor_subsample_keep_has_answer.process_train(train_examples)
        raise NotImplementedError

    def test_eval_preprocessing_runs_without_errors(self, eval_examples, preprocessor):
        eval_examples, eval_features = preprocessor.process_eval(eval_examples)
        assert isinstance(eval_examples, Dataset)
        assert isinstance(eval_features, Dataset)
