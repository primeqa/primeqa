import datasets
import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from oneqa.mrc.processors.preprocessors.squad import SQUADPreprocessor
from tests.oneqa.mrc.common.base import UnitTest


class TestSQUADQAPreprocessor(UnitTest):

    @pytest.mark.flaky(reruns=10, reruns_delay=30)  # Account for intermittent S3 errors downloading HF data
    @pytest.fixture(scope='session')
    def train_examples(self):
        examples = datasets.load_dataset("squad", "plain_text", split='train[:100]')
        return examples

    @pytest.mark.flaky(reruns=10, reruns_delay=30)  # Account for intermittent S3 errors downloading HF data
    @pytest.fixture(scope='session')
    def eval_examples(self):
        examples = datasets.load_dataset("squad", "plain_text", split='validation[:100]')
        return examples

    @pytest.fixture(scope='class')
    def tokenizer(self):
        return AutoTokenizer.from_pretrained('xlm-roberta-base')

    @pytest.fixture(scope='class')
    def squad_preprocessor(self, tokenizer):
        return SQUADPreprocessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
        )

    def test_train_preprocessing_runs_without_errors(self, train_examples, squad_preprocessor):
        train_examples, train_features = squad_preprocessor.process_train(train_examples)
        assert isinstance(train_examples, Dataset)
        assert isinstance(train_features, Dataset)
        for example in train_examples:
            assert min(example['target']['start_positions']) >= -1
            assert min(example['target']['end_positions']) >= -1
            assert min(example['target']['passage_indices']) >= -1

    def test_eval_preprocessing_runs_without_errors(self, eval_examples, squad_preprocessor):
        eval_examples, eval_features = squad_preprocessor.process_eval(eval_examples)
        assert isinstance(eval_examples, Dataset)
        assert isinstance(eval_features, Dataset)
        for example in eval_examples:
            assert min(example['target']['start_positions']) >= -1
            assert min(example['target']['end_positions']) >= -1
            assert min(example['target']['passage_indices']) >= -1
