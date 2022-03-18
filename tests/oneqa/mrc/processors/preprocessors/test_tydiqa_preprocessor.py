import datasets
import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from oneqa.mrc.processors.preprocessors.tydiqa import TyDiQAPreprocessor
from tests.oneqa.mrc.common.base import UnitTest


class TestTyDiQAPreprocessor(UnitTest):

    @pytest.mark.flaky(reruns=10, reruns_delay=2)  # Account for intermittent S3 errors downloading HF data
    @pytest.fixture(scope='session')
    def train_examples(self):
        examples = datasets.load_dataset("tydiqa", "primary_task", split='train[:100]')
        return examples

    @pytest.mark.flaky(reruns=10, reruns_delay=2)  # Account for intermittent S3 errors downloading HF data
    @pytest.fixture(scope='session')
    def eval_examples(self):
        examples = datasets.load_dataset("tydiqa", "primary_task", split='validation[:100]')
        return examples

    @pytest.fixture(scope='class')
    def tokenizer(self):
        return AutoTokenizer.from_pretrained('xlm-roberta-base')

    @pytest.fixture(scope='class')
    def tydiqa_preprocessor(self, tokenizer):
        return TyDiQAPreprocessor(
            tokenizer,
            stride=128
        )

    def test_train_preprocessing_runs_without_errors(self, train_examples, tydiqa_preprocessor):
        train_examples, train_features = tydiqa_preprocessor.process_train(train_examples)
        assert isinstance(train_examples, Dataset)
        assert isinstance(train_features, Dataset)

    def test_eval_preprocessing_runs_without_errors(self, eval_examples, tydiqa_preprocessor):
        eval_examples, eval_features = tydiqa_preprocessor.process_eval(eval_examples)
        assert isinstance(eval_examples, Dataset)
        assert isinstance(eval_features, Dataset)
