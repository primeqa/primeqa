from tests.oneqa.mrc.common.base import UnitTest
import pytest
import os
import json
from typing import Tuple

from oneqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from oneqa.ir.dense.colbert_top.colbert.training.eager_batcher_v2 import EagerBatcher  # support text input

class TestBatchers(UnitTest):

    '''
    @pytest.fixture(scope='session')
    def triples_location(self):
        # The current directory to handle running from IDE or command line
        curdir = os.getcwd()
        if curdir.endswith('tests'):
            triples_path = '../tests/resources/sample_wiki_psgs_train_triples'
        else:
            triples_path = 'tests/resources/sample_wiki_train_triples'
        return triples_path

    @pytest.fixture(scope='session')
    def expected_search_results(self):

        search_results_1 = [
            (0, '20076582', 17.771099090576172, 'Nerine Desmond'),
            (1, '19750546',  2.9017999172210693, 'SOS-Hermann Gmeiner International College')
        ]
        search_results_2 = [
            (0, '15415536', 9.206600189208984, 'Mercury regulation in the United States'),
            (1, '18680280',  2.569499969482422, 'Phonological history of Old English')
        ]
        search_results_3 = [
            (0, '8356488', 5.252799987792969, 'Presanella'),
            (1, '8237529', 2.0999999046325684, 'Idaho State Police')
        ]
        return [search_results_1, search_results_2, search_results_3]
    '''

    #def test_eager_batcher(self, triples_location):
    def test_eager_batcher(self):
        #t_location = '/u/franzm/git8/OneQA/tests/resources'
        triples_location = 'tests/resources/ir_dense'
        rank = 0
        nranks = 1
        config = ColBERTConfig()
        reader = EagerBatcher(config, triples_location + "/ColBERT.C3_3_20_biased200_triples_text_head_100.tsv", rank, nranks)

if __name__ == '__main__':
    test = TestBatchers()
    test.test_eager_batcher()