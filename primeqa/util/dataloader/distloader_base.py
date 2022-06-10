import logging
import random
import os
import glob
import torch
from primeqa.util.transformers_utils.hypers_base import HypersBase
from primeqa.util.file_utils import jsonl_lines, read_open
from typing import Iterable
import ujson as json
from typing import List
import itertools
from transformers import PreTrainedTokenizer
import numpy as np

logger = logging.getLogger(__name__)


def sentence_to_inputs(sentence: str, tokenizer: PreTrainedTokenizer, max_seq_length: int, start_end_tok_ids=None):
    """
    :param sentence:
    :param tokenizer:
    :param max_seq_length: we don't pad to this length, only truncate
    :param start_end_tok_ids: if not None, then start_id, end_id = start_end_tok_ids
    :return:
    """
    sentence = sentence.strip()
    if not sentence:
        return None
    sent_tokens = tokenizer.tokenize(sentence)

    if len(sent_tokens) > max_seq_length - 2:
        sent_tokens = sent_tokens[0:max_seq_length - 2]

    input_ids = tokenizer.convert_tokens_to_ids(sent_tokens)
    if start_end_tok_ids is not None:
        input_ids = [start_end_tok_ids[0]] + input_ids + [start_end_tok_ids[1]]

    return np.array(input_ids, dtype=np.int32)


class DistBatchesBase:
    def __init__(self, insts, hypers: HypersBase):
        self.insts = insts
        self.hypers = hypers
        self.batch_size = None
        self.num_batches = None
        self.displayer = None
        self.uneven_batches = False

    def post_init(self, *, batch_size, displayer=None, uneven_batches=False, random=None):
        # CONSIDER: put batch_size in post_init, since we always pass per_gpu_batch_size to the MultiFileLoader
        self.batch_size = batch_size
        self.num_batches = len(self.insts) // self.batch_size
        self.displayer = displayer
        self.uneven_batches = uneven_batches
        if random is not None:
            random.shuffle(self.insts)
        if self.uneven_batches or self.hypers.world_size == 1:
            if len(self.insts) % self.batch_size != 0:
                self.num_batches += 1
        else:
            self._distributed_min()

    def _distributed_min(self):
        if self.hypers.world_size == 1:
            return
        # we need to take the minimum to allow for cases where the dataloaders may have a bit different counts
        num_batches = torch.tensor(self.num_batches, dtype=torch.long).to(self.hypers.device)
        torch.distributed.all_reduce(num_batches, torch.distributed.ReduceOp.MIN)
        batch_limit = num_batches.item()
        if self.num_batches > batch_limit:
            logger.warning(f'truncating from {self.num_batches} to {batch_limit} batches on {self.hypers.global_rank}, '
                           f'lost {(1 - batch_limit/self.num_batches)*100} percent')
            self.num_batches = batch_limit
        else:
            logger.warning(f'world rank {self.hypers.global_rank}: all workers doing '
                           f'{self.num_batches} batches of size {self.batch_size}')

    def __len__(self):
        return self.num_batches

    def make_batch(self, index, insts):
        # NOTE: subclasses will override this
        raise NotImplementedError

    def __getitem__(self, index):
        if index >= self.num_batches:
            raise IndexError
        if self.hypers.world_size == 1:
            batch_insts = self.insts[index::self.num_batches]
        else:
            batch_insts = self.insts[index * self.batch_size: (index + 1) * self.batch_size]
        batch = self.make_batch(index, batch_insts)
        if index == 0 and self.displayer is not None:
            self.displayer(batch)
        return batch


class MultiFileLoader:
    """
    handles the multi-file splitting across processes and the checkpointing
    """
    def __init__(self, hypers: HypersBase, per_gpu_batch_size: int, train_dir: str, *,
                 checkpoint_info=None, files_per_dataloader=1, uneven_batches=False):
        self.hypers = hypers
        self.train_dir = train_dir
        self.per_gpu_batch_size = per_gpu_batch_size
        if hypers.resume_from and os.path.isfile(os.path.join(hypers.resume_from, "loader_checkpoint.json")):
            resume_from = hypers.resume_from
        elif os.path.isfile(os.path.join(hypers.model_name_or_path, "loader_checkpoint.json")):
            resume_from = hypers.model_name_or_path
        else:
            resume_from = None
        if checkpoint_info is None and resume_from is not None:
            with read_open(os.path.join(resume_from, "loader_checkpoint.json")) as f:
                checkpoint_info = json.load(f)
            logger.info(f'loaded distloader checkpoint from {resume_from}')
        # CONSIDER: get checkpoint as a json.load from hypers.output_dir/loader_checkpoint.json
        if checkpoint_info and 'completed_files' in checkpoint_info:
            self.completed_files = checkpoint_info['completed_files']
        else:
            self.completed_files = []
        if checkpoint_info and 'on_epoch' in checkpoint_info:
            self.on_epoch = checkpoint_info['on_epoch']
        else:
            self.on_epoch = 1
        self.num_epochs = hypers.num_train_epochs
        self.files_per_dataloader = files_per_dataloader
        self.uneven_batches = uneven_batches
        self.first_batches_loaded = False
        self.train_files = None

    def get_checkpoint_info(self):
        # completed_files, on_epoch
        return {'completed_files': self.completed_files, 'on_epoch': self.on_epoch}

    def _get_files(self, leftover_files=None):
        logger.info('completed files = %s, count = %i',
                    str(self.completed_files[:5]), len(self.completed_files))

        if os.path.isfile(self.train_dir):
            self.train_files = [self.train_dir]
        else:
            if not self.train_dir.endswith('/'):
                self.train_dir = self.train_dir + '/'
            self.train_files = glob.glob(self.train_dir + '**', recursive=True)
            self.train_files = [f for f in self.train_files if not os.path.isdir(f)]

        # exclude completed files
        self.train_files = [f for f in self.train_files if f not in self.completed_files]

        self.train_files.sort()
        random.Random(123 * self.on_epoch).shuffle(self.train_files)
        if leftover_files is not None:
            self.train_files = leftover_files + self.train_files
        if self.files_per_dataloader == -1:
            self.files_per_dataloader = max(1, len(self.train_files) // self.hypers.world_size)
        logger.info('epoch %i, pending files = %s, count = %i',
                    self.on_epoch, str(self.train_files[:5]), len(self.train_files))

    def reset(self, *, files_per_dataloader=None, uneven_batches=None, num_epochs=None):
        if files_per_dataloader is not None:
            self.files_per_dataloader = files_per_dataloader
        if uneven_batches is not None:
            self.uneven_batches = uneven_batches
        if num_epochs is not None:
            self.num_epochs = num_epochs
        self.on_epoch = 1
        self.completed_files = []
        self.train_files = None

    def _get_input_files(self):
        if self.train_files is None:
            self._get_files()
        num_files_to_use = self.files_per_dataloader * self.hypers.world_size
        files_are_shared = False
        # start new epoch if we have fewer train_files than world_size
        if len(self.train_files) == 0:
            self.completed_files = []
            self.on_epoch += 1
            if self.on_epoch > self.num_epochs:
                return None, None
            self._get_files(leftover_files=self.train_files)
        if len(self.train_files) < self.hypers.world_size * self.files_per_dataloader:
            num_files_to_use = len(self.train_files)
            files_are_shared = True

        assert num_files_to_use > 0
        used_files = []
        while len(used_files) < num_files_to_use:
            used_files.append(self.train_files.pop())
        self.completed_files.extend(used_files)

        if files_are_shared:
            return used_files, files_are_shared
        else:
            return used_files[self.hypers.global_rank::self.hypers.world_size], files_are_shared

    def quick_test(self, lines: List[str]):
        batches = self._one_load(lines)
        batches.post_init(batch_size=self.per_gpu_batch_size * self.hypers.n_gpu, displayer=self.display_batch,
                          uneven_batches=True, random=random.Random(123))
        logger.info(f'batch size = {batches.batch_size}, batch count = {batches.num_batches}')
        for b in batches:
            logger.info(f'after first batch')
            return b

    def get_dataloader(self):
        input_files, files_are_shared = self._get_input_files()
        if input_files is None:
            return None
        lines = jsonl_lines(input_files)
        # if input_files are supposed to be shared then get only the lines for our global_rank
        if files_are_shared:
            lines = itertools.islice(lines, self.hypers.global_rank, None, self.hypers.world_size)
        logger.warning(f'on {self.hypers.global_rank} rank, using files: {input_files}, shared: {files_are_shared}')
        batches = self._one_load(lines)
        displayer = None
        if not self.first_batches_loaded:
            self.first_batches_loaded = True
            displayer = self.display_batch
        batches.post_init(batch_size=self.per_gpu_batch_size * self.hypers.n_gpu, displayer=displayer,
                          uneven_batches=self.uneven_batches, random=random.Random(123 * self.on_epoch))
        return batches

    def all_batches(self):
        while True:
            loader = self.get_dataloader()
            if loader is None:
                break
            for batch in loader:
                yield batch

    def display_batch(self, batch):
        pass

    def batch_dict(self, batch):
        raise NotImplementedError

    def _one_load(self, lines: Iterable[str]) -> DistBatchesBase:
        raise NotImplementedError
