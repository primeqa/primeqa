from tests.primeqa.mrc.common.base import UnitTest
import pytest
import os
import json
import torch
import tempfile
import argparse
from typing import Tuple

from primeqa.ir.dense.xtr_top.xtr import trainer
from primeqa.ir.dense.xtr_top.xtr.indexer import Indexer
from primeqa.ir.dense.xtr_top.xtr.searcher import Searcher


class TestTraining(UnitTest):
    def test_trainer(self):
        test_files_location = 'tests/resources/ir_dense'
        #test_files_location = '../../../../resources/ir_dense'
        if 'DATA_FILES_FOR_DENSE_IR_TESTS_PATH' in os.environ:
            test_files_location = os.environ['DATA_FILES_FOR_DENSE_IR_TESTS_PATH']

        queries_fn = os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_queries_fornum.tsv")
        collection_fn = os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv")
        text_triples_fn = os.path.join(test_files_location, "xorqa.train_ir_negs_5_poss_1_001pct_at_0pct.tsv")

        with tempfile.TemporaryDirectory() as working_dir:
            output_dir=os.path.join(working_dir, 'output_dir')
        experiment_path = os.path.join(output_dir, "experiment")
        index_location = os.path.join(output_dir, "index_name")

        do_indexing = True
        if do_indexing:
            args_dict = {"doc_maxlen": 512, "bsize": 128, "model_name_or_path": "t5-small", 
                        "collection": collection_fn, "index_name": index_location, "nbits": 4, 
                        "kmeans_niters": 2, "num_partitions_max": 10, "nranks": 1, "rank": -1, "dim": 128}
            args = argparse.Namespace(**args_dict)
            indexer = Indexer(args)
            indexer.index(name=args.index_name, collection=args.collection, overwrite=True)
            
            print("INDEXING DONE")

        do_search = True
        if do_search:
            args_dict = {"query_maxlen": 64, "bsize": 128, "model_name_or_path": "t5-small", 
                        "collection": collection_fn, "index_name": index_location, "nbits": 4, 
                        "ndocs": 10, "ncells": 4, "topK": 5, "queries": queries_fn, "dim": 128,
                        "output_dir": output_dir}
            os.makedirs(output_dir, exist_ok=True)
            ranks_fn = os.path.join(output_dir, 'ranking.tsv')
            args = argparse.Namespace(**args_dict)
            searcher = Searcher(args.index_name, checkpoint=args.model_name_or_path, config=args)
            if torch.cuda.is_available():
                rankings = searcher.search_all(args.queries, k=args.topK) #NOTE needs gpu
                with open(ranks_fn, "w") as ofp:
                    json.dump(rankings, ofp)
    
            print("SEARCH DONE")

        do_training = True
        if do_training:
            args_dict = {"doc_maxlen": 512, "bsize": 16, "model_name_or_path": "t5-small", 
                        "triples": text_triples_fn, "nway": 2, "save_steps": 100, "maxsteps": 100,
                        "query_maxlen": 64, "k_train": 32, "experiment_path": experiment_path,
                        "nranks": 1, "rank": -1, "dim": 128, "lr": 1e-3, "max_grad_clip": 2., 
                        "warmup": None, "shuffle_every_epoch": None, "epochs": 1, "rng_seed": 1234,
                        "accumsteps": 1}
            args = argparse.Namespace(**args_dict)
            trainer.train(args)
            print("TRAINING DONE")

        assert True
        print("ALL DONE")

if __name__ == '__main__':
    test = TestTraining()
    test.test_trainer()
