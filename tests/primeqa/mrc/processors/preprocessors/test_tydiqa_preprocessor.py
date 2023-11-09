import datasets
import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from primeqa.mrc.processors.preprocessors.tydiqa import TyDiQAPreprocessor
from tests.primeqa.mrc.common.base import UnitTest


class TestTyDiQAPreprocessor(UnitTest):

 
    @pytest.fixture(scope='class')
    def tokenizer(self):
        return AutoTokenizer.from_pretrained('xlm-roberta-base')

    @pytest.fixture(scope='class')
    def tydiqa_preprocessor(self, tokenizer):
        return TyDiQAPreprocessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
        )

    def get_example(self, example):
        return example
    
    def test_train_preprocessing_runs_without_errors(self, tydiqa_train_examples, tydiqa_preprocessor):        
        train_examples, train_features = tydiqa_preprocessor.process_train(tydiqa_train_examples)
        assert isinstance(train_examples, Dataset)
        assert isinstance(train_features, Dataset)
        for example in train_examples:
            assert min(example['target']['start_positions']) >= -1
            assert min(example['target']['end_positions']) >= -1
            assert min(example['target']['passage_indices']) >= -1

    def test_eval_preprocessing_runs_without_errors(self, tydiqa_eval_examples, tydiqa_preprocessor):
        eval_examples, eval_features = tydiqa_preprocessor.process_eval(tydiqa_eval_examples)
        assert isinstance(eval_examples, Dataset)
        assert isinstance(eval_features, Dataset)
        for example in eval_examples:
            assert min(example['target']['start_positions']) >= -1
            assert min(example['target']['end_positions']) >= -1
            assert min(example['target']['passage_indices']) >= -1
