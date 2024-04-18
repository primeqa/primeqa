import os
import time

import torch
import random

import numpy as np

import torch.multiprocessing as mp

from primeqa.ir.dense.colbert_top.colbert.infra.launcher import Launcher

from primeqa.ir.dense.xtr_top.xtr.indexing.collection_indexer import encode


class Indexer:
    def __init__(self, config):
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)
        torch.cuda.manual_seed(12345)

        self.config = config
        self.index_name = None
        self.config.gpus_ = []
        if torch.cuda.is_available():
            self.config.gpus_ = [1]

        self.config.rng_seed = 12345

    def configure(self, **kw_args):
        for key, value in kw_args.items():
            setattr(self.config, key, value)

    def get_index(self):
        return self.index_name

    def erase(self):
        assert self.index_name is not None
        directory = self.index_name
        deleted = []

        for filename in sorted(os.listdir(directory)):
            filename = os.path.join(directory, filename)

            delete = filename.endswith(".json")
            delete = delete and ('metadata' in filename or 'doclen' in filename or 'plan' in filename)
            delete = delete or filename.endswith(".pt")
            
            if delete:
                deleted.append(filename)
        
        if len(deleted):
            print(f"#> Will delete {len(deleted)} files already at {directory} in 20 seconds...")
            time.sleep(20)

            for filename in deleted:
                os.remove(filename)

        return deleted

    def index(self, name=None, collection=None, overwrite=False):
        assert overwrite in [True, False, 'reuse']

        self.configure(collection=collection, index_name=name)
        self.configure(bsize=64, partitions=None)

        self.index_name = self.config.index_name
        index_does_not_exist = (not os.path.exists(self.config.index_name))

        assert (overwrite in [True, 'reuse']) or index_does_not_exist, self.config.index_name
        os.makedirs(self.config.index_name, exist_ok=True)

        if overwrite is True:
            self.erase()

        if index_does_not_exist or overwrite != 'reuse':
            self.__launch(collection)

        return self.index_name

    def __launch(self, collection):
        manager = mp.Manager()
        shared_lists = [manager.list() for _ in range(self.config.nranks)]
        shared_queues = [manager.Queue(maxsize=1) for _ in range(self.config.nranks)]

        launcher = Launcher(encode)
        launcher.launch(self.config, collection, shared_lists, shared_queues)
