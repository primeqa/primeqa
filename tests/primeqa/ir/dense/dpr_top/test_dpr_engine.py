from tests.primeqa.mrc.common.base import UnitTest
import pytest
import os
import argparse, sys

from unittest.mock import patch
import tempfile

from transformers import HfArgumentParser

from primeqa.ir.dense.dpr_top.dpr.biencoder_trainer import BiEncoderTrainer
from primeqa.ir.dense.dpr_top.dpr.index_simple_corpus import DPRIndexer
from primeqa.ir.dense.dpr_top.dpr.searcher import DPRSearcher
from primeqa.ir.dense.dpr_top.dpr.config import DPRTrainingArguments, DPRIndexingArguments, DPRSearchArguments


class TestDprEngine(UnitTest):
    @pytest.fixture(scope='session')
    def test_files_location(self):
        location = 'tests/resources/ir_dense'
        if 'DATA_FILES_FOR_DENSE_IR_TESTS_PATH' in os.environ:
            location = os.environ['DATA_FILES_FOR_DENSE_IR_TESTS_PATH']
        return location

    def test_engine(self, test_files_location):
        with tempfile.TemporaryDirectory() as working_dir:
            output_dir=os.path.join(working_dir, 'output_dir')
        os.makedirs(output_dir, exist_ok=True)

        print("===== DPR TRAINING, -training_data_type kgi_jsonl")

        test_args = [
            "prog",
            "--train_dir", os.path.join(test_files_location, "kgi_jsonl_2_lines.jsonl"),
            "--positive_pids", os.path.join(test_files_location, "kgi_jsonl_2_lines_positive_pids.jsonl"),
            "--output_dir", output_dir,
            "--epochs", "2",
            "--sample_negative_from_top_k", "5",
            "--encoder_gpu_train_limit", "32",
            "--bsize", "1",
            "--max_grad_norm", "1.0",
            "--learning_rate", "5e-5",
            "--training_data_type", "kgi_jsonl",
            "--force_confict_free_batches"
        ]

        with patch.object(sys, 'argv', test_args):
            parser = HfArgumentParser([DPRTrainingArguments])
            (dpr_args, remaining_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
            trainer = BiEncoderTrainer(dpr_args)
            trainer.train()

        print("===== DPR TRAINING, -training_data_type text_triples")

        test_args = [
            "prog",
            "--train_dir", os.path.join(test_files_location, "xorqa.train_ir_negs_5_poss_1_001pct_at_0pct_en.tsv"),
            "--output_dir", output_dir,
            "--epochs", "2",
            "--sample_negative_from_top_k", "5",
            "--encoder_gpu_train_limit", "32",
            "--bsize", "1",
            "--max_grad_norm", "1.0",
            "--learning_rate", "5e-5",
            "--training_data_type", "text_triples"
        ]

        with patch.object(sys, 'argv', test_args):
            parser = HfArgumentParser([DPRTrainingArguments])
            (dpr_args, remaining_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
            trainer = BiEncoderTrainer(dpr_args)
            trainer.train()

        print("===== DPR TRAINING, -training_data_type text_triples_with_title")

        test_args = [
            "prog",
            "--train_dir", os.path.join(test_files_location, "ColBERT.C3_3_20_biased200_triples_text_head_10.tsv"),
            "--output_dir", output_dir,
            "--epochs", "2",
            "--sample_negative_from_top_k", "5",
            "--encoder_gpu_train_limit", "32",
            "--bsize", "1",
            "--max_grad_norm", "1.0",
            "--learning_rate", "5e-5",
            "--training_data_type", "text_triples_with_title"
        ]

        with patch.object(sys, 'argv', test_args):
            parser = HfArgumentParser([DPRTrainingArguments])
            (dpr_args, remaining_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
            trainer = BiEncoderTrainer(dpr_args)
            trainer.train()

        print("===== DPR TRAINING, -training_data_type num_triples")

        test_args = [
            "prog",
            "--train_dir", os.path.join(test_files_location, "xorqa.train_ir_negs_5_poss_1_001pct_at_0pct_num.json"),
            "--collection", os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv"),
            "--queries", os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_queries_fornum_en.tsv"),
            "--output_dir", output_dir,
            "--epochs", "2",
            "--sample_negative_from_top_k", "5",
            "--encoder_gpu_train_limit", "32",
            "--bsize", "1",
            "--max_grad_norm", "1.0",
            "--learning_rate", "5e-5",
            "--training_data_type", "num_triples"
        ]

        with patch.object(sys, 'argv', test_args):
            parser = HfArgumentParser([DPRTrainingArguments])
            (dpr_args, remaining_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
            trainer = BiEncoderTrainer(dpr_args)
            trainer.train()

        print("===== DPR INDEXING")

        test_args = [
            "prog",
            "--ctx_encoder_name_or_path", os.path.join(output_dir, "ctx_encoder"),
            "--embed", "1of1",
            "--sharded_index",
            "--bsize", "1",
            "--collection", os.path.join(test_files_location,"xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv"),
            "--output_dir", output_dir]

        with patch.object(sys, 'argv', test_args):
            parser = HfArgumentParser([DPRIndexingArguments])
            (dpr_args, remaining_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
            indexer = DPRIndexer(dpr_args)
            indexer.index()

        print("===== DPR SEARCH")

        test_args = [
            "prog",
            "--queries", os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_queries_fornum_en.tsv"),
            "--model_name_or_path", os.path.join(output_dir, "qry_encoder"),
            "--bsize", "1",
            "--index_location", output_dir,
            "--output_dir", os.path.join(output_dir, "search_output"),
            "--top_k", "1"]

        with patch.object(sys, 'argv', test_args):
            parser = HfArgumentParser([DPRSearchArguments])
            (dpr_args, remaining_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
            searcher = DPRSearcher(dpr_args)
            searcher.search()
            searcher.search(query_batch = ['Who maintained the throne for the longest time in China?'], mode = 'query_list')

        print("===== DPR ALL DONE")