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
    @classmethod
    def setup_class(cls):
        import torch

        if torch.cuda.is_available():
            import time
            import random

            rank = 0
            nranks = 1

            rng = random.Random(time.time())
            port = str(12355 + rng.randint(0, 1000))

            os.environ["MASTER_PORT"] = port
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["WORLD_SIZE"] = str(nranks)
            os.environ["RANK"] = str(rank)

            torch.cuda.set_device(0)
            torch.distributed.init_process_group(backend='nccl', init_method='env://')

    def test_batchers(self):
        test_files_location = 'tests/resources/ir_dense'
        if 'DATA_FILES_FOR_DENSE_IR_TESTS_PATH' in os.environ:
            test_files_location = os.environ['DATA_FILES_FOR_DENSE_IR_TESTS_PATH']

        rank = 0
        nranks = 1
        config = ColBERTConfig()
        config.checkpoint = 'prajjwal1/bert-tiny'
        text_triples_fn = os.path.join(test_files_location, "ColBERT.C3_3_20_biased200_triples_text_head_100.tsv")
        reader_eager_batcher = EagerBatcher(config, text_triples_fn, rank, nranks)

        numerical_triples_fn = os.path.join(test_files_location, "xorqa.train_ir_negs_5_poss_1_001pct_at_0pct_num.json")
        queries_fn = os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_queries_fornum.tsv")
        collection_fn = os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv")
        reader_lazy_batcher = LazyBatcher(config, numerical_triples_fn, queries_fn, collection_fn, rank, nranks)

    def test_trainer(self):
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

        model_types = {'bert': 'prajjwal1/bert-tiny'}

        do_training = True
        if do_training:
            for model_type, checkpoint in model_types.items():
                args_dict = {'root': output_dir, 'experiment': 'test_training', 'rank': 0, 'similarity': 'cosine', 'dim': 128, 'query_maxlen': 32, 'doc_maxlen': 180, 'mask_punctuation': True, 'local_models_repository': None, 'resume': False, 'resume_optimizer': False, 'checkpoint': checkpoint, 'lr': 1.5e-06, 'maxsteps': 5, 'bsize': 1, 'accumsteps': 1, 'amp': True, 'shuffle_every_epoch': False, 'save_steps': 2000, 'save_epochs': -1, 'epochs': 1, 'teacher_checkpoint': None, 'student_teacher_temperature': 1.0, 'student_teacher_top_loss_weight': 0.5, 'teacher_doc_maxlen': 180, 'distill_query_passage_separately': False, 'query_only': False, 'loss_function': None, 'query_weight': 0.5, 'triples': text_triples_fn, 'queries': None, 'collection': None, 'teacher_triples': None, 'nranks': 1}

                with Run().context(RunConfig(root=args_dict['root'], experiment=args_dict['experiment'], nranks=args_dict['nranks'], amp=args_dict['amp'])):
                    # reading text training triples
                    colBERTConfig = ColBERTConfig(**args_dict)
                    latest_model_fn = train(colBERTConfig, text_triples_fn, None, None)

                    if checkpoint == 'prajjwal1/bert-tiny':

                        # reading numerical training triples
                        args_dict['triples'] = numerical_triples_fn
                        args_dict['queries'] = queries_fn
                        args_dict['collection'] = collection_fn
                        colBERTConfig = ColBERTConfig(**args_dict)
                        train(colBERTConfig, numerical_triples_fn, queries_fn, collection_fn)

                        # student/teacher training, top level
                        args_dict['teacher_checkpoint'] = checkpoint
                        args_dict['teacher_triples'] = text_triples_en_fn
                        args_dict['queries'] = None
                        args_dict['collection'] = None
                        colBERTConfig = ColBERTConfig(**args_dict)
                        train(colBERTConfig, text_triples_fn, None, None)

                        # student/teacher model, token level
                        args_dict['distill_query_passage_separately'] = True
                        args_dict['teacher_checkpoint'] = checkpoint
                        args_dict['triples'] = parallel_non_en_fn
                        args_dict['teacher_triples'] = parallel_en_fn
                        args_dict['queries'] = None
                        args_dict['collection'] = None

                        colBERTConfig = ColBERTConfig(**args_dict)
                        train(colBERTConfig, parallel_en_fn, None, None)

            print("TRAINING DONE")

        do_indexing = True
        if do_indexing:
            args_dict = {'root': os.path.join(output_dir, 'test_indexing'), 'experiment': 'test_indexing', 'rank': 0, 'similarity': 'cosine', 'dim': 128, 'query_maxlen': 32, 'doc_maxlen': 180, 'mask_punctuation': True, 'local_models_repository': None, 'checkpoint': latest_model_fn + '.dnn.epoch_0.model', 'bsize': 256, 'amp': True, 'collection': collection_fn, 'index_root': os.path.join(output_dir, 'test_indexing', 'indexes'), 'index_name': 'index_name', 'num_partitions_max': 2, 'kmeans_niters': 1, 'nway': 1, 'nranks': 1}
            with Run().context(RunConfig(root=args_dict['root'], experiment=args_dict['experiment'], nranks=args_dict['nranks'], amp=args_dict['amp'])):
                    colBERTConfig = ColBERTConfig(**args_dict)
                    create_directory(colBERTConfig.index_path_)
                    encode(colBERTConfig, collection_fn, None, None)

            print("INDEXING DONE")

        do_search = True
        if do_search:
            ranks_fn = os.path.join(output_dir, 'ranking.tsv')
            args_dict = {'root': output_dir, 'experiment': 'test_indexing' , 'rank': -1, 'similarity': 'cosine', 'dim': 128, 'query_maxlen': 32, 'doc_maxlen': 180, 'mask_punctuation': True, 'local_models_repository': None, 'checkpoint': latest_model_fn, 'bsize': 1, 'amp': True, 'queries': queries_fn, 'collection': collection_fn, 'ranks_fn': ranks_fn, 'topK': 1, 'index_root': output_dir, 'index_name': 'index_name', 'nranks': 1, 'index_location' : os.path.join(output_dir, 'test_indexing', 'indexes', 'index_name' ) }
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
