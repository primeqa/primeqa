from tests.primeqa.mrc.common.base import UnitTest
import pytest
import os
import argparse, sys

from unittest.mock import patch
import tempfile

from primeqa.ir.dense.dpr_top.dpr.biencoder_trainer import BiEncoderTrainer
from primeqa.ir.dense.dpr_top.dpr.index_simple_corpus import DPRIndexer
from primeqa.ir.dense.dpr_top.dpr.searcher import DPRSearcher

class TestDprEngine(UnitTest):
    @pytest.fixture(scope='session')
    def test_files_location(self):
        location = 'tests/resources/ir_dense'
        if 'DATA_FILES_FOR_DENSE_IR_TESTS_PATH' in os.environ:
            location = os.environ['DATA_FILES_FOR_DENSE_IR_TESTS_PATH']
        return location

    def test_engine(self, test_files_location):
        #test_files_location = 'tests/resources/ir_dense'
        #if 'DATA_FILES_FOR_DENSE_IR_TESTS_PATH' in os.environ:
        #    test_files_location = os.environ['DATA_FILES_FOR_DENSE_IR_TESTS_PATH']

        with tempfile.TemporaryDirectory() as working_dir:
            output_dir=os.path.join(working_dir, 'output_dir')
        os.makedirs(output_dir, exist_ok=True)

        print("===== DPR TRAINING, -training_data_type kgi_jsonl")

        test_args = [
            "prog",
            "--train_dir", os.path.join(test_files_location, "kgi_jsonl_2_lines.jsonl"),
            "--positive_pids", os.path.join(test_files_location, "kgi_jsonl_2_lines_positive_pids.jsonl"),
            "--output_dir", output_dir,
            "--num_train_epochs", "2",
            "--sample_negative_from_top_k", "5",
            "--encoder_gpu_train_limit", "32",
            "--full_train_batch_size", "1",
            "--max_grad_norm", "1.0",
            "--learning_rate", "5e-5",
            "--training_data_type", "kgi_jsonl",
            "--force_confict_free_batches"
        ]

        with patch.object(sys, 'argv', test_args):
            trainer = BiEncoderTrainer()
            trainer.train()


        print("===== DPR TRAINING, -training_data_type text_triples")

        test_args = [
            "prog",
            "--train_dir", os.path.join(test_files_location, "xorqa.train_ir_negs_5_poss_1_001pct_at_0pct_en.tsv"),
            "--output_dir", output_dir,
            "--num_train_epochs", "2",
            "--sample_negative_from_top_k", "5",
            "--encoder_gpu_train_limit", "32",
            "--full_train_batch_size", "1",
            "--max_grad_norm", "1.0",
            "--learning_rate", "5e-5",
            "--training_data_type", "text_triples"
        ]

        with patch.object(sys, 'argv', test_args):
            trainer = BiEncoderTrainer()
            trainer.train()

        print("===== DPR TRAINING, -training_data_type text_triples_with_title")

        test_args = [
            "prog",
            "--train_dir", os.path.join(test_files_location, "ColBERT.C3_3_20_biased200_triples_text_head_10.tsv"),
            "--output_dir", output_dir,
            "--num_train_epochs", "2",
            "--sample_negative_from_top_k", "5",
            "--encoder_gpu_train_limit", "32",
            "--full_train_batch_size", "1",
            "--max_grad_norm", "1.0",
            "--learning_rate", "5e-5",
            "--training_data_type", "text_triples_with_title"
        ]

        with patch.object(sys, 'argv', test_args):
            trainer = BiEncoderTrainer()
            trainer.train()

        print("===== DPR TRAINING, -training_data_type num_triples")

        test_args = [
            "prog",
            "--train_dir", os.path.join(test_files_location, "xorqa.train_ir_negs_5_poss_1_001pct_at_0pct_num.json"),
            "--collection", os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv"),
            "--queries", os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_queries_fornum_en.tsv"),
            "--output_dir", output_dir,
            "--num_train_epochs", "2",
            "--sample_negative_from_top_k", "5",
            "--encoder_gpu_train_limit", "32",
            "--full_train_batch_size", "1",
            "--max_grad_norm", "1.0",
            "--learning_rate", "5e-5",
            "--training_data_type", "num_triples"
        ]

        with patch.object(sys, 'argv', test_args):
            trainer = BiEncoderTrainer()
            trainer.train()


        print("===== DPR INDEXING")

        test_args = [
            "prog",
            "--dpr_ctx_encoder_model_name", "facebook/dpr-ctx_encoder-multiset-base",
            "--dpr_ctx_encoder_path", os.path.join(output_dir, "ctx_encoder"),
            "--embed", "1of1",
            "--sharded_index",
            "--batch_size", "1",
            "--corpus", os.path.join(test_files_location,"xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv"),
            "--output_dir", output_dir]

        with patch.object(sys, 'argv', test_args):
            indexer = DPRIndexer()
            indexer.index()

        print("===== DPR SEARCH")

        test_args = [
            "prog",
            "--queries", os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_queries_fornum_en.tsv"),
            "--qry_encoder_path", os.path.join(output_dir, "qry_encoder"),
            "--qry_tokenizer_path", "facebook/dpr-question_encoder-multiset-base",
            "--retrieve_batch_size", "1",
            "--include_passages",
            "--corpus_dir", output_dir,
            "--output", os.path.join(output_dir, "search_output"),
            "--n_docs_for_provenance", "1"]

        with patch.object(sys, 'argv', test_args):
            searcher = DPRSearcher()
            searcher.search()


        print("===== DPR ALL DONE")