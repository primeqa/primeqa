from tests.oneqa.mrc.common.base import UnitTest
import pytest
import os
import tempfile
import json
from typing import Tuple

from oneqa.ir.dense.colbert_top.colbert.infra import Run, RunConfig
from oneqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from oneqa.ir.dense.colbert_top.colbert.training.eager_batcher_v2 import EagerBatcher  # support text input
from oneqa.ir.dense.colbert_top.colbert.training.lazy_batcher import LazyBatcher  # support text input
from oneqa.ir.dense.colbert_top.colbert.trainer import Trainer
from oneqa.ir.dense.colbert_top.colbert.utils.parser import Arguments
from oneqa.ir.dense.colbert_top.colbert.training.training import train


class TestTraining(UnitTest):

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
        test_files_location = 'tests/resources/ir_dense'
        if os.getcwd().endswith('pycharm/pycharm-community-2022.1/bin'):
            test_files_location = '/u/franzm/git8/OneQA/tests/resources/ir_dense'

        rank = 0
        nranks = 1
        config = ColBERTConfig()
        text_triples_fn = os.path.join(test_files_location, "ColBERT.C3_3_20_biased200_triples_text_head_100.tsv")
        reader_eager_batcher = EagerBatcher(config, text_triples_fn, rank, nranks)

        numerical_triples_fn = os.path.join(test_files_location, "xorqa.train_ir_negs_5_poss_1_001pct_at_0pct_num.json")
        queries_fn = os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_queries_fornum.tsv")
        collection_fn = os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv")
        reader_lazy_batcher = LazyBatcher(config, numerical_triples_fn, queries_fn, collection_fn, rank, nranks)

    def test_trainer(self):
        test_files_location = 'tests/resources/ir_dense'
        if os.getcwd().endswith('pycharm/pycharm-community-2022.1/bin'):
            test_files_location = '/u/franzm/git8/OneQA/tests/resources/ir_dense'

        '''
        parser = Arguments(description='Training ColBERT with <query, positive passage, negative passage> triples.')

        parser.add_model_parameters()
        parser.add_model_training_parameters()
        parser.add_training_input()
        args = parser.parse()
        # parser.add_argument('--model_type', dest='model_type', default='bert')
        # comment out as we define the argument at model training parameters

        '''
        text_triples_fn = os.path.join(test_files_location, "ColBERT.C3_3_20_biased200_triples_text_head_100.tsv")
        numerical_triples_fn = os.path.join(test_files_location, "xorqa.train_ir_negs_5_poss_1_001pct_at_0pct_num.json")
        queries_fn = os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_queries_fornum.tsv")
        collection_fn = os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv")

        with tempfile.TemporaryDirectory() as working_dir:
            output_dir=os.path.join(working_dir, 'output_dir')

        model_types = ['xlm-roberta-base', 'bert-base-uncased']
        for model_type in model_types:
            args_dict = {'root': output_dir, 'experiment': 'test_training', 'rank': -1, 'similarity': 'l2', 'dim': 128, 'query_maxlen': 32, 'doc_maxlen': 180, 'mask_punctuation': True, 'local_models_repository': None, 'resume': False, 'resume_optimizer': False, 'checkpoint': model_type, 'init_from_lm': None, 'model_type': model_type, 'lr': 1.5e-06, 'maxsteps': 5, 'bsize': 1, 'accumsteps': 1, 'amp': True, 'shuffle_every_epoch': False, 'save_steps': 2000, 'save_epochs': -1, 'epochs': 10, 'teacher_checkpoint': None, 'student_teacher_temperature': 1.0, 'student_teacher_top_loss_weight': 0.5, 'teacher_model_type': None, 'teacher_doc_maxlen': 180, 'distill_query_passage_separately': False, 'query_only': False, 'loss_function': None, 'query_weight': 0.5, 'triples': text_triples_fn, 'queries': None, 'collection': None, 'teacher_triples': None, 'nranks': 1}

            with Run().context(RunConfig(root=args_dict['root'], experiment=args_dict['experiment'], nranks=args_dict['nranks'], amp=args_dict['amp'])):
                colBERTConfig = ColBERTConfig(**args_dict)
                train(colBERTConfig, text_triples_fn, None, None)

                args_dict['triples'] = numerical_triples_fn
                args_dict['queries'] = queries_fn
                args_dict['collection'] = collection_fn
                colBERTConfig = ColBERTConfig(**args_dict)
                train(colBERTConfig, numerical_triples_fn, queries_fn, collection_fn)

        print("ALL DONE")


if __name__ == '__main__':
    test = TestTraining()
    #test.test_eager_batcher()
    test.test_trainer()
