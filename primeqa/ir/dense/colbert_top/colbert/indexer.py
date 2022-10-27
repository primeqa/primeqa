import os
import time

import torch
import random
import numpy as np

import torch.multiprocessing as mp

from primeqa.ir.dense.colbert_top.colbert.infra.run import Run
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig, RunConfig
from primeqa.ir.dense.colbert_top.colbert.infra.launcher import Launcher

from primeqa.ir.dense.colbert_top.colbert.utils.utils import create_directory, print_message

from primeqa.ir.dense.colbert_top.colbert.indexing.collection_indexer import encode


class Indexer:
    def __init__(self, checkpoint, config=None):
        """
           Use Run().context() to choose the run's configuration. They are NOT extracted from `config`.
        """
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)
        torch.cuda.manual_seed(12345)

        self.index_path = None
        self.checkpoint = checkpoint
        self.checkpoint_config = ColBERTConfig.load_from_checkpoint(checkpoint)

        # set model_type from checkpoint's config
        # config.model_type = self.checkpoint_config.model_type

        self.config = ColBERTConfig.from_existing(self.checkpoint_config, config, Run().config)

        # set model_type from checkpoint's config
        # self.config.model_type = self.checkpoint_config.model_type

        self.configure(checkpoint=checkpoint)



    def configure(self, **kw_args):
        self.config.configure(**kw_args)

    def get_index(self):
        return self.index_path

    def erase(self):
        assert self.index_path is not None
        directory = self.index_path
        deleted = []

        for filename in sorted(os.listdir(directory)):
            filename = os.path.join(directory, filename)

            delete = filename.endswith(".json")
            delete = delete and ('metadata' in filename or 'doclen' in filename or 'plan' in filename)
            delete = delete or filename.endswith(".pt")
            
            if delete:
                deleted.append(filename)
        
        if len(deleted):
            print_message(f"#> Will delete {len(deleted)} files already at {directory} in 20 seconds...")
            time.sleep(20)

            for filename in deleted:
                os.remove(filename)

        return deleted

    def index(self, name, collection, overwrite=False):
        assert overwrite in [True, False, 'reuse']

        self.configure(collection=collection, index_name=name)
        self.configure(bsize=64, partitions=None)
        # self.configure(bsize=1, partitions=None)

        self.index_path = self.config.index_path_
        index_does_not_exist = (not os.path.exists(self.config.index_path_))

        assert (overwrite in [True, 'reuse']) or index_does_not_exist, self.config.index_path_
        create_directory(self.config.index_path_)

        if overwrite is True:
            self.erase()

        if index_does_not_exist or overwrite != 'reuse':
            self.__launch(collection)

        return self.index_path

    def __launch(self, collection):
        manager = mp.Manager()
        shared_lists = [manager.list() for _ in range(self.config.nranks)]
        shared_queues = [manager.Queue(maxsize=1) for _ in range(self.config.nranks)]

        launcher = Launcher(encode)
        launcher.launch(self.config, collection, shared_lists, shared_queues)
