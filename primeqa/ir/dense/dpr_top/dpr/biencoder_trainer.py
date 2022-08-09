from primeqa.ir.dense.dpr_top.torch_util.transformer_optimize import TransformerOptimize
from primeqa.ir.dense.dpr_top.dpr.biencoder_hypers import BiEncoderHypers
from primeqa.ir.dense.dpr_top.dpr.biencoder_gcp import BiEncoder
from primeqa.ir.dense.dpr_top.dpr.dataloader_biencoder import BiEncoderLoader
from transformers import (DPRContextEncoderTokenizerFast, DPRQuestionEncoderTokenizerFast)
import logging
import time
from primeqa.ir.dense.dpr_top.util.line_corpus import jsonl_lines, jsonl_records

logger = logging.getLogger(__name__)


class BiEncoderTrainArgs(BiEncoderHypers):
    def __init__(self):
        super().__init__()
        self.train_dir = ''
        self.positive_pids = ''
        self.num_instances = -1
        self.__required_args__ = ['train_dir', 'output_dir', 'positive_pids']

    def _post_init(self):
        super()._post_init()
        if self.num_instances <= 0:
            if self.training_data_type == 'dpr':
                self.num_instances = sum(1 for _ in jsonl_records(self.train_dir))
            else:
                self.num_instances = sum(1 for _ in jsonl_lines(self.train_dir))
            logger.info(f'Counted num_instances = {self.num_instances}')

class BiEncoderTrainer():
    def __init__(self):
        self.args = BiEncoderTrainArgs().fill_from_args()
        if self.args.n_gpu > 1:
            logger.error('Multi-GPU training must be through torch.distributed')
            exit(1)
        if self.args.world_size > 1 and 0 < self.args.encoder_gpu_train_limit:
            logger.error('Cannot support both distributed training and gradient checkpointing.  '
                         'Train with a single GPU or with --encoder_gpu_train_limit 0')
            exit(1)
        self.qry_tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained('facebook/dpr-question_encoder-multiset-base')
        self.ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained('facebook/dpr-ctx_encoder-multiset-base')
        self.model = BiEncoder(self.args)
        self.model.to(self.args.device)
        self.model.train()
        self.optimizer = TransformerOptimize(self.args, self.args.num_train_epochs * self.args.num_instances, self.model)
        self.loader = BiEncoderLoader(self.args, self.args.per_gpu_train_batch_size, self.qry_tokenizer, self.ctx_tokenizer,
                                 self.args.train_dir, self.args.positive_pids, files_per_dataloader=-1)
        self.last_save_time = time.time()
        self.args.set_seed()

    def train(self):
        while True:
            batches = self.loader.get_dataloader()
            if not self.optimizer.should_continue() or batches is None:
                break
            for batch in batches:
                loss, accuracy = self.optimizer.model(**self.loader.batch_dict(batch))
                self.optimizer.step_loss(loss, accuracy=accuracy)
                if not self.optimizer.should_continue():
                    break
            if time.time()-self.last_save_time > 60*60:
                # save once an hour or after each file (whichever is less frequent)
                model_to_save = (self.optimizer.model.module if hasattr(self.optimizer.model, "module") else self.optimizer.model)
                logger.info(f'saving to {self.args.output_dir}')
                model_to_save.save(self.args.output_dir)
                self.last_save_time = time.time()

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

