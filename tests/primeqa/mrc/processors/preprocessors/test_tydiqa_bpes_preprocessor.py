import datasets
import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from primeqa.mrc.processors.preprocessors.tydiboolqa_bpes import TyDiBoolQAPreprocessor
from tests.primeqa.mrc.common.base import UnitTest
import numpy as np

class TestTyDiBoolQAPreprocessor(UnitTest):

    @pytest.fixture(scope='class')
    def tokenizer(self):
        return AutoTokenizer.from_pretrained('xlm-roberta-base')

    @pytest.fixture(scope='class')
    def tydiqa_preprocessor(self, tokenizer):
        return TyDiBoolQAPreprocessor(
            tokenizer,
            stride=128,
            load_from_cache_file=False,
        )

    def test_train_preprocessing_runs_without_errors(self, tydiqa_train_examples, tydiqa_preprocessor):
        train_examples, train_features = tydiqa_preprocessor.process_train(tydiqa_train_examples)
        assert isinstance(train_examples, Dataset)
        assert isinstance(train_features, Dataset)
        for example in train_examples:
            assert min(example['target']['start_positions']) >= -1
            assert min(example['target']['end_positions']) >= -1
            assert min(example['target']['passage_indices']) >= -1

        # when the target type is yes(3) or no(4) the start positions and end positions should be
        # nonzero (unlike in TydiQAPreprocessor)
        # there is no corresponding block in test_eval... because we only have gold target types 
        # at train time
        tt=np.array(train_features['target_type'])
        sp=np.array(train_features['start_positions'])
        ep=np.array(train_features['end_positions'])
        yn=(tt==3)|(tt==4)
        assert(np.all(ep[yn]>0))
        assert(np.all(sp[yn]>0))            


