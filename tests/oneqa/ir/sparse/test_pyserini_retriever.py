
from tests.oneqa.mrc.common.base import UnitTest
import pytest
import os
import json
from typing import Tuple
from oneqa.ir.sparse.retriever import PyseriniRetriever


class TestPyseriniRetriever(UnitTest):


    @pytest.fixture(scope='session')
    def index_location(self):
        # The current directory to handle running from IDE or command line
        curdir = os.getcwd()
        if curdir.endswith('tests'):
            index_path = '../tests/resources/sample_wiki_psgs_w100_index'
        else:
            index_path = 'tests/resources/sample_wiki_psgs_w100_index'
        return index_path

    @pytest.fixture(scope='session')
    def num_docs(self):
        return 100

    @pytest.fixture(scope='session')
    def queries(self):
        return [
            'who designed the South African 1961 one-cent postage stamp',
            'vitamin e deficiency',
            'where is the Presanella located',
            #'1943 Rose Bowl'
        ]

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
    
    def test_instantiate_retriever(self, index_location, num_docs):
        searcher = PyseriniRetriever(index_location)
        assert(searcher.searcher.num_docs == num_docs)

    def test_retrieve(self, index_location, queries, expected_search_results):
        searcher = PyseriniRetriever(index_location)
        for i, query in enumerate(queries):
            hits = searcher.retrieve(query,top_k=2)
            expected_results = expected_search_results[i]
            self._validate_search_results(hits, expected_results)

    def test_batch_retrieve(self, index_location, queries, expected_search_results):
        searcher = PyseriniRetriever(index_location)
        qids = ['0','1','2']
        assert(len(qids) == len(queries))
        qid_to_hits = searcher.batch_retrieve(queries, qids, top_k=2)
        for i, qid in enumerate(qids):
            hits = qid_to_hits[qid]
            expected_results = expected_search_results[i]
            self._validate_search_results(hits, expected_results)


    def _validate_search_results(self, hits, expected_results):
        assert(len(hits) == len(expected_results))
        for h, hit in enumerate(hits):
            rank, docid, score, title = expected_results[h]
            assert(rank == hit['rank'])
            assert(docid == hit['doc_id'])
            assert(score == hit['score'])
            assert(title == hit['title'])
            assert('text' in hit)


    