from tests.primeqa.mrc.common.base import UnitTest
import pytest
import os
import tempfile
import json
from typing import Tuple

from primeqa.ir.dense.colbert_top.colbert.utils.utils import create_directory, print_message
from primeqa.ir.dense.colbert_top.colbert.infra import Run, RunConfig
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.training.eager_batcher_v2 import EagerBatcher  # support text input
from primeqa.ir.dense.colbert_top.colbert.training.lazy_batcher import LazyBatcher  # support text input
from primeqa.ir.dense.colbert_top.colbert.trainer import Trainer
from primeqa.ir.dense.colbert_top.colbert.utils.parser import Arguments
from primeqa.ir.dense.colbert_top.colbert.training.training import train
from primeqa.ir.dense.colbert_top.colbert.indexing.collection_indexer import encode
from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher

class TestTraining(UnitTest):
    def test_batchers(self):
        print("entering test_batchers")
        test_files_location = 'tests/resources/ir_dense'
        if os.getcwd().endswith('pycharm/pycharm-community-2022.1/bin'):
            test_files_location = 'PrimeQA/tests/resources/ir_dense'

        rank = 0
        nranks = 1
        config = ColBERTConfig()
        text_triples_fn = os.path.join(test_files_location, "ColBERT.C3_3_20_biased200_triples_text_head_100.tsv")
        reader_eager_batcher = EagerBatcher(config, text_triples_fn, rank, nranks)

        numerical_triples_fn = os.path.join(test_files_location, "xorqa.train_ir_negs_5_poss_1_001pct_at_0pct_num.json")
        queries_fn = os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_queries_fornum.tsv")
        collection_fn = os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv")
        reader_lazy_batcher = LazyBatcher(config, numerical_triples_fn, queries_fn, collection_fn, rank, nranks)
        print("exiting test_batchers")

    def test_trainer(self):
        print("entering test_trainer")
        test_files_location = 'tests/resources/ir_dense'
        if 'DATA_FILES_FOR_DENSE_IR_TESTS_PATH' in os.environ:
            test_files_location = os.environ['DATA_FILES_FOR_DENSE_IR_TESTS_PATH']

        queries_fn = os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_queries_fornum.tsv")
        collection_fn = os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv")
        text_triples_fn = os.path.join(test_files_location, "xorqa.train_ir_negs_5_poss_1_001pct_at_0pct.tsv")
        text_triples_en_fn = os.path.join(test_files_location, "xorqa.train_ir_negs_5_poss_1_001pct_at_0pct_en.tsv")
        numerical_triples_fn = os.path.join(test_files_location, "xorqa.train_ir_negs_5_poss_1_001pct_at_0pct_num.json")
        parallel_non_en_fn = os.path.join(test_files_location, "7lan_notrim_triple_2ep.other.clean.h5")
        parallel_en_fn = os.path.join(test_files_location, "7lan_notrim_triple_2ep.en.clean.h5")

        with tempfile.TemporaryDirectory() as working_dir:
            output_dir=os.path.join(working_dir, 'output_dir')


        model_types = ['bert-base-uncased', 'xlm-roberta-base']
       
        print("test_trainer 1") 
        
        do_training = True
        if do_training:
            for model_type in model_types:
                args_dict = {'root': output_dir, 'experiment': 'test_training', 'rank': 0, 'similarity': 'l2', 'dim': 128, 'query_maxlen': 32, 'doc_maxlen': 180, 'mask_punctuation': True, 'local_models_repository': None, 'resume': False, 'resume_optimizer': False, 'checkpoint': model_type, 'init_from_lm': None, 'model_type': model_type, 'lr': 1.5e-06, 'maxsteps': 5, 'bsize': 1, 'accumsteps': 1, 'amp': True, 'shuffle_every_epoch': False, 'save_steps': 2000, 'save_epochs': -1, 'epochs': 1, 'teacher_checkpoint': None, 'student_teacher_temperature': 1.0, 'student_teacher_top_loss_weight': 0.5, 'teacher_model_type': None, 'teacher_doc_maxlen': 180, 'distill_query_passage_separately': False, 'query_only': False, 'loss_function': None, 'query_weight': 0.5, 'triples': text_triples_fn, 'queries': None, 'collection': None, 'teacher_triples': None, 'nranks': 1}

                with Run().context(RunConfig(root=args_dict['root'], experiment=args_dict['experiment'], nranks=args_dict['nranks'], amp=args_dict['amp'])):
                    # reading text training triples
                    colBERTConfig = ColBERTConfig(**args_dict)
                    print("test_trainer 2") 
                    latest_model_fn = train(colBERTConfig, text_triples_fn, None, None)
                    print("test_trainer 3") 
                    assert False
                    if model_type == 'xlm-roberta-base':
                        # additional modalities done for 'xlm-roberta-base' only

                        # reading numerical training triples
                        print("test_trainer 4") 
                        args_dict['triples'] = numerical_triples_fn
                        args_dict['queries'] = queries_fn
                        args_dict['collection'] = collection_fn
                        colBERTConfig = ColBERTConfig(**args_dict)
                        train(colBERTConfig, numerical_triples_fn, queries_fn, collection_fn)
                        
                        print("test_trainer 5") 

                        # student/teacher training, top level
                        args_dict['teacher_checkpoint'] = model_type
                        args_dict['teacher_model_type'] = model_type
                        args_dict['teacher_triples'] = text_triples_en_fn
                        args_dict['queries'] = None
                        args_dict['collection'] = None
                        colBERTConfig = ColBERTConfig(**args_dict)
                        train(colBERTConfig, text_triples_fn, None, None)
                        
                        print("test_trainer 6") 

                        # student/teacher model, token level
                        args_dict['distill_query_passage_separately'] = True
                        args_dict['teacher_checkpoint'] = model_type
                        args_dict['teacher_model_type'] = model_type
                        args_dict['triples'] = parallel_non_en_fn
                        args_dict['teacher_triples'] = parallel_en_fn
                        args_dict['queries'] = None
                        args_dict['collection'] = None
                        
                        print("test_trainer 7") 

                        colBERTConfig = ColBERTConfig(**args_dict)
                        train(colBERTConfig, parallel_en_fn, None, None)

            print("TRAINING DONE")

        do_indexing = True
        if do_indexing:
            print("test_trainer do_indexing start") 
            args_dict = {'root': os.path.join(output_dir, 'test_indexing'), 'experiment': 'test_indexing', 'rank': 0, 'similarity': 'l2', 'dim': 128, 'query_maxlen': 32, 'doc_maxlen': 180, 'mask_punctuation': True, 'local_models_repository': None, 'checkpoint': latest_model_fn, 'bsize': 256, 'amp': True, 'collection': collection_fn, 'index_root': os.path.join(output_dir, 'test_indexing', 'indexes'), 'index_name': 'index_name', 'num_partitions_max': 2, 'kmeans_niters': 1, 'nway': 1, 'nranks': 1}
            with Run().context(RunConfig(root=args_dict['root'], experiment=args_dict['experiment'], nranks=args_dict['nranks'], amp=args_dict['amp'])):
                    colBERTConfig = ColBERTConfig(**args_dict)
                    create_directory(colBERTConfig.index_path_)
                    encode(colBERTConfig, collection_fn, None, None)

            print("INDEXING DONE")

        do_search = True
        if do_search:
            print("test_trainer do_search start") 
            ranks_fn = os.path.join(output_dir, 'ranking.tsv')
            args_dict = {'root': output_dir, 'experiment': 'test_indexing' , 'rank': -1, 'similarity': 'l2', 'dim': 128, 'query_maxlen': 32, 'doc_maxlen': 180, 'mask_punctuation': True, 'local_models_repository': None, 'checkpoint': latest_model_fn, 'bsize': 1, 'amp': True, 'queries': queries_fn, 'collection': collection_fn, 'ranks_fn': ranks_fn, 'topK': 1, 'index_root': output_dir, 'index_name': 'index_name', 'nprobe': 1, 'nranks': 1, 'model_type': model_type,}

            with Run().context(RunConfig(root=args_dict['root'], experiment=args_dict['experiment'], nranks=args_dict['nranks'], amp=args_dict['amp'])):
                colBERTConfig = ColBERTConfig(**args_dict)
                searcher = Searcher(args_dict['index_name'], checkpoint=args_dict['checkpoint'], collection=args_dict['collection'], config=colBERTConfig)

                rankings = searcher.search_all(args_dict['queries'], args_dict['topK'])
                out_fn = args_dict['ranks_fn']
                rankings.save(out_fn)

            print("SEARCH DONE")

        print("ALL DONE")


if __name__ == '__main__':
    test = TestTraining()
    #test.test_eager_batcher()
    test.test_trainer()
