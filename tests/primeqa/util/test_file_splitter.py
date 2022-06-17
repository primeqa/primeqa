from primeqa.util.dataloader.file_splitter import main, Options
import os
import pytest


class Tester:
    @pytest.fixture(scope='session')
    def test_resources_dir(self):
        curdir = os.getcwd()
        if curdir.endswith('tests'):
            resources_path = '../tests/resources'
        else:
            resources_path = 'tests/resources'
        return resources_path

    def test_hypers(self, tmpdir, test_resources_dir):
        opts = Options()
        opts.input = os.path.join(test_resources_dir,
                                  'ir_dense', 'xorqa.train_ir_001pct_at_0_pct_collection_fornum.tsv')
        opts.outdirs = ','.join([os.path.join(tmpdir, split) for split in ['train', 'test']])
        opts.split_fractions = '0.8,0.2'
        main(opts)
