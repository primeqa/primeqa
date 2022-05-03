from tests.oneqa.mrc.common.base import UnitTest
import pytest
import os
import tempfile

from argparse import ArgumentParser
from oneqa.ir.dense.colbert_top.colbert.utils.parser import Arguments
from oneqa.ir.dense.colbert_top.utility.preprocess.docs2passages import main as docs2passages_main

class TestOther(UnitTest):

    def test_utility(self):

        parser = Arguments(description='')
        parser.add_model_parameters()
        parser = Arguments(description='')
        parser.add_model_training_parameters()
        parser = Arguments(description='')
        parser.add_model_inference_parameters()
        parser = Arguments(description='')
        parser.add_training_input()
        parser = Arguments(description='')
        parser.add_ranking_input()
        parser = Arguments(description='')
        parser.add_reranking_input()
        parser = Arguments(description='')
        parser.add_indexing_input()
        parser = Arguments(description='')
        parser.add_compressed_index_input()
        parser = Arguments(description='')
        parser.add_index_use_input()
        parser = Arguments(description='')
        parser.add_retrieval_input()
        parser = Arguments(description='')
        #args = parser.parse()

        # utility.preprocess.docs2passages

        test_files_location = 'tests/resources/ir_dense'
        if os.getcwd().endswith('pycharm/pycharm-community-2022.1/bin'):
            test_files_location = '/u/franzm/git8/OneQA/tests/resources/ir_dense'
        collection_fn = os.path.join(test_files_location, "xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv")

        with tempfile.TemporaryDirectory() as working_dir:
            output_dir=os.path.join(working_dir, 'output_dir')

        Format1 = 'docid,text'  # MS MARCO Passages
        Format2 = 'docid,text,title'   # DPR Wikipedia
        Format3 = 'docid,url,title,text'  # MS MARCO Documents

        parser = ArgumentParser(description="docs2passages.")

        parser.add_argument('--input', dest='input', default=collection_fn)
        parser.add_argument('--format', dest='format', default=Format2, choices=[Format1, Format2, Format3])

        # Output Arguments.
        parser.add_argument('--use-wordpiece', dest='use_wordpiece', default=True, action='store_true')
        parser.add_argument('--nwords', dest='nwords', default=10, type=int)
        parser.add_argument('--overlap', dest='overlap', default=0, type=int)

        # Other Arguments.
        parser.add_argument('--nthreads', dest='nthreads', default=1, type=int)

        parser.add_argument('--output_path', dest='output_path', default=output_dir)

        args = parser.parse_args()

        docs2passages_main(args)

if __name__ == '__main__':
    test = TestOther()
    test.test_utility()
