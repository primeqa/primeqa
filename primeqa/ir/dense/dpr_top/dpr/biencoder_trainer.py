import os
import logging
import time
import torch
import numpy as np
import random

from transformers import (DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizerFast)

from primeqa.ir.dense.dpr_top.torch_util.transformer_optimize import TransformerOptimize
from primeqa.ir.dense.dpr_top.dpr.biencoder_hypers import BiEncoderHypers
from primeqa.ir.dense.dpr_top.dpr.biencoder_gcp import BiEncoder
from primeqa.ir.dense.dpr_top.dpr.dataloader_biencoder import BiEncoderLoader
from primeqa.ir.dense.dpr_top.util.line_corpus import jsonl_lines, jsonl_records
from primeqa.ir.dense.dpr_top.dpr.config import DPRTrainingArguments

logger = logging.getLogger(__name__)


class BiEncoderTrainArgs(BiEncoderHypers):
    def __init__(self):
        super().__init__()
        self.train_dir = ''
        self.positive_pids = ''
        self.num_instances = -1
        self.resume_from_checkpoint = ''
        self.save_every_num_batches = 0
        self.log_every_num_batches = 0
        self.log_all_losses = False
        self.max_negatives = 0
        self.max_hard_negatives = 1
        self.__required_args__ = ['train_dir', 'output_dir']

    def _post_init(self):
        super()._post_init()
        if self.num_instances <= 0:
            if self.training_data_type == 'dpr':      # .json, as in https://dl.fbaipublicfiles.com/dpr/data/retriever/biencoder-nq-dev.json.gz
                for rec in jsonl_records(self.train_dir):
                    self.num_instances += len(rec['positive_ctxs']) * (min(self.max_negatives, len(rec['negative_ctxs'])) + min(self.max_hard_negatives, len(rec['hard_negative_ctxs']))) # as expanded in _one_load(self, lines)
            elif self.training_data_type == 'kgi_jsonl':  # ,jsonl, as in the original code in https://github.com/IBM/kgi-slot-filling/tree/re2g
                self.num_instances = sum(1 for _ in jsonl_lines(self.train_dir, file_suffix='*.jsonl*'))
            elif self.training_data_type == 'text_triples':    # .tsv, containing [query, positive, negative] triples
                self.num_instances = sum(1 for _ in jsonl_lines(self.train_dir, file_suffix='*.tsv'))
            elif self.training_data_type == 'text_triples_with_title':    # .tsv, containing [query, positive, negative] triples, with "title | text" for passages
                self.num_instances = sum(1 for _ in jsonl_lines(self.train_dir, file_suffix='*.tsv'))
            elif self.training_data_type == 'num_triples':    # text file, containing [query_id, positive_id, negative_id] triples
                self.num_instances = sum(1 for _ in jsonl_lines(self.train_dir, file_suffix='*'))
            else:
                raise NotImplementedError(f"Input data type {self.training_data_type} is not implemented (yet).")
            logger.info(f'Counted num_instances = {self.num_instances}')

class BiEncoderTrainer():
    def __init__(self, config: DPRTrainingArguments):
        self.args = BiEncoderTrainArgs().fill_from_config(config)

        if self.args.n_gpu > 1:
            logger.error('Multi-GPU training must be through torch.distributed')
            exit(1)
        if self.args.world_size > 1 and 0 < self.args.encoder_gpu_train_limit:
            logger.error('Cannot support both distributed training and gradient checkpointing.  '
                         'Train with a single GPU or with --encoder_gpu_train_limit 0')
            exit(1)

        self.qry_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(self.args.qry_encoder_name_or_path)
        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(self.args.ctx_encoder_name_or_path)
        self.model = BiEncoder(self.args)
        self.model.to(self.args.device)
        self.model.train()
        self.optimizer = TransformerOptimize(self.args, self.args.epochs * self.args.num_instances, self.model)
        self.loader = BiEncoderLoader(self.args, self.args.per_gpu_train_batch_size, self.qry_tokenizer, self.ctx_tokenizer,
                                 self.args.train_dir, self.args.positive_pids, files_per_dataloader=-1)

        self.last_save_time = time.time()

        self.first_batch_num = 0
        self.args.set_seed()


    def save_checkpoint(self, save_to_path, batch_num):
        logger.info(f'saving checkpoint to {save_to_path}')

        checkpoint = {}

        checkpoint['epoch'] = self.loader.on_epoch
        checkpoint['batch'] = batch_num

        checkpoint['qry_model_state_dict'] = self.model.qry_model.state_dict()
        checkpoint['ctx_model_state_dict'] = self.model.ctx_model.state_dict()

        checkpoint['optimizer_state_dict'] = self.optimizer.optimizer.state_dict()
        checkpoint['scheduler_state_dict'] = self.optimizer.scheduler.state_dict()

        checkpoint['torch_rng_state'] = torch.get_rng_state()
        checkpoint['torch_cuda_rng_states'] = torch.cuda.get_rng_state_all()
        checkpoint['np_rng_state'] = np.random.get_state()
        checkpoint['python_rng_state'] = random.getstate()

        torch.save(checkpoint, save_to_path)
        logger.info(f'saved checkpoint to {save_to_path}')


    def load_checkpoint(self, load_from_path):
        logger.info(f'loading checkpoint from {load_from_path}')

        checkpoint = torch.load(load_from_path, map_location='cpu')

        if checkpoint['batch'] + 1 >= self.batches.num_batches:  # checkpoint based on the last batch in the epoch
            checkpoint['batch'] = -1
            checkpoint['epoch'] += 1

        while self.loader.on_epoch < checkpoint['epoch']:
            self.batches = self.loader.get_dataloader()

        self.loader.on_epoch = checkpoint['epoch']
        self.first_batch_num = checkpoint['batch'] + 1

        self.model.qry_model.load_state_dict(checkpoint['qry_model_state_dict'])  # , strict=False)
        self.model.ctx_model.load_state_dict(checkpoint['ctx_model_state_dict'])  # , strict=False)

        self.optimizer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.optimizer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        torch.set_rng_state(checkpoint['torch_rng_state'].to(torch.get_rng_state().device))
        if torch.cuda.is_available():
            torch.cuda.set_rng_state_all([ state.to(torch.cuda.get_rng_state_all()[pos].device) for pos, state in enumerate(checkpoint['torch_cuda_rng_states']) ] )
        np.random.set_state(checkpoint['np_rng_state'])
        random.setstate(checkpoint['python_rng_state'])

        logger.info(f'loaded checkpoint from {load_from_path}')

    def save_tokenizers(self):
        self.ctx_tokenizer.save_pretrained(os.path.join(self.args.output_dir, 'ctx_encoder'))
        self.qry_tokenizer.save_pretrained(os.path.join(self.args.output_dir, 'qry_encoder'))

    def train(self):
        self.save_tokenizers()
        while True:
            self.batches = self.loader.get_dataloader()
            if not self.optimizer.should_continue() or self.batches is None:
                if not self.optimizer.should_continue():
                    logger.info(f'Breaking, self.optimizer.should_continue() is False')
                if self.batches is None:
                    logger.info(f'Breaking, self.batches is None')
                break
            logger.info(f'len(self.batches) {len(self.batches)}')

            if self.args.resume_from_checkpoint != '':
                if self.args.world_size != 1:
                    raise NotImplementedError(f'Resuming training from a checkpoint is not supported (yet) for world_size != 1.')

                self.load_checkpoint(self.args.resume_from_checkpoint)
                self.args.resume_from_checkpoint = ''

            for batch_num in list(range(self.first_batch_num, self.batches.num_batches)):
                batch = self.batches[batch_num]
                loss, accuracy = self.optimizer.model(**self.loader.batch_dict(batch))
                if self.args.log_all_losses:
                    logger.info(f'batch_num: {batch_num}, {loss}, {accuracy}')
                self.optimizer.step_loss(loss, accuracy=accuracy)
                if not self.optimizer.should_continue():
                    break

                if self.args.log_every_num_batches > 0 and (batch_num % self.args.log_every_num_batches) == 0 :
                    logger.info(f'batch_num: {batch_num}')
                    self.optimizer.optimizer_report()

                if time.time() - self.last_save_time > 60 * 60 or \
                        (self.args.save_every_num_batches > 0 and batch_num > 0 and (batch_num % self.args.save_every_num_batches) == 0):
                    # save once an hour or after each "save_every_num_batches" (whichever is more frequent)
                    self.save_checkpoint(os.path.join(self.args.output_dir, "latest_checkpoint"), batch_num)
                    model_to_save = (self.optimizer.model.module if hasattr(self.optimizer.model, "module") else self.optimizer.model)
                    logger.info(f'saving to {self.args.output_dir}')
                    model_to_save.save(self.args.output_dir)
                    self.last_save_time = time.time()

            # save after each epoch
            self.save_checkpoint(os.path.join(self.args.output_dir, "latest_checkpoint"), batch_num)
            model_to_save = (self.optimizer.model.module if hasattr(self.optimizer.model, "module") else self.optimizer.model)
            logger.info(f'saving to {self.args.output_dir}')
            model_to_save.save(self.args.output_dir)
            self.last_save_time = time.time()
            self.first_batch_num = 0

        # save after running out of files or target num_instances
        logger.info(f'All done')
        self.optimizer.reporting.display()
        model_to_save = (self.optimizer.model.module if hasattr(self.optimizer.model, "module") else self.optimizer.model)
        logger.info(f'saving to {self.args.output_dir}')
        model_to_save.save(self.args.output_dir)
        logger.info(f'Took {self.optimizer.reporting.elapsed_time_str()}')

def main():
    trainer = BiEncoderTrainer()
    trainer.train()

if __name__ == "__main__":
    main()

