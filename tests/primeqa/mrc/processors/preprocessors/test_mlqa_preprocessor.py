import datasets
import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from primeqa.mrc.processors.preprocessors.mlqa import MLQAPreprocessor
from tests.primeqa.mrc.common.base import UnitTest


class TestMLQAPreprocessor(UnitTest):

    @pytest.mark.flaky(reruns=10, reruns_delay=30)  # Account for intermittent S3 errors downloading HF data
    @pytest.fixture(scope='session')
    def eval_examples(self):
        examples = datasets.load_dataset("mlqa", "mlqa.zh.en", split='validation[:100]')
        return examples

    @pytest.fixture(scope='class')
    def tokenizer(self):
        return AutoTokenizer.from_pretrained('xlm-roberta-base')

    @pytest.fixture(scope='class')
    def mlqa_preprocessor(self, tokenizer):
        return MLQAPreprocessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
        )

    def test_eval_preprocessing_runs_without_errors(self, eval_examples, mlqa_preprocessor):
        eval_examples, eval_features = mlqa_preprocessor.process_eval(eval_examples)
        assert isinstance(eval_examples, Dataset)
        assert isinstance(eval_features, Dataset)
        for example in eval_examples:
            assert min(example['target']['start_positions']) >= -1
            assert min(example['target']['end_positions']) >= -1
            assert min(example['target']['passage_indices']) >= -1
            assert example['answer_language'] == 'zh'
