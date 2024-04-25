from tests.primeqa.mrc.common.base import UnitTest
import pytest
import os
import json
import torch
import tempfile
import argparse
from typing import Tuple

from primeqa.ir.dense.xtr_top.xtr import trainer


class TestTraining(UnitTest):

    def test_trainer(self):
        test_files_location = 'tests/resources/ir_dense'
        if 'DATA_FILES_FOR_DENSE_IR_TESTS_PATH' in os.environ:
            test_files_location = os.environ['DATA_FILES_FOR_DENSE_IR_TESTS_PATH']

        text_triples_fn = os.path.join(test_files_location, "xorqa.train_ir_negs_5_poss_1_001pct_at_0pct.tsv")

        with tempfile.TemporaryDirectory() as working_dir:
            output_dir=os.path.join(working_dir, 'output_dir')

        experiment_path = os.path.join(output_dir, "experiment")
        args_dict = {"doc_maxlen": 512, "bsize": 16, "model_name_or_path": "t5-small", 
                    "triples": text_triples_fn, "nway": 2, "save_steps": 50, "maxsteps": 100,
                    "query_maxlen": 64, "k_train": 32, "experiment_path": experiment_path,
                    "nranks": 1, "rank": -1, "dim": 128, "lr": 1e-3, "max_grad_clip": 2., 
                    "warmup": None, "shuffle_every_epoch": None, "epochs": 10, "rng_seed": 1234,
                    "accumsteps": 1}
        args = argparse.Namespace(**args_dict)
        trainer.train(args)
        print("TRAINING DONE")

if __name__ == '__main__':
    test = TestTraining()
    test.test_trainer()
