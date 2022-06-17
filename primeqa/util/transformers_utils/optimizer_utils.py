import torch
import os
import logging
try:
    from apex import amp
except ModuleNotFoundError:
    pass
from primeqa.util.transformers_utils.hypers_base import HypersBase
from primeqa.util.reporting import Reporting
from transformers import (
    AdamW,
    get_linear_schedule_with_warmup,
)
from primeqa.util.transformers_utils.torch_utils import reduce

logger = logging.getLogger(__name__)


class LossHistory:
    def __init__(self, one_epoch_batch_count, *, loss_points_per_epoch=10, recency_weight=0.001):
        self.avg_loss = 0
        self.batch_count = 0
        self.recency_weight = recency_weight
        self.loss_history = []
        self.record_loss_every = max(1, one_epoch_batch_count // loss_points_per_epoch)

    def note_loss(self, loss_val, *, hypers: HypersBase = None):
        self.batch_count += 1
        rweight = max(self.recency_weight, 1.0 / self.batch_count)
        self.avg_loss = (1.0 - rweight) * self.avg_loss + rweight * loss_val
        if self.batch_count % self.record_loss_every == 0:
            if hypers is not None and hypers.world_size > 1:
                self.avg_loss = reduce(hypers, self.avg_loss).item() / hypers.world_size
            self.loss_history.append(self.avg_loss)
            logger.info(f'loss point {self.batch_count//self.record_loss_every} = {self.avg_loss}')
            return True
        return False


class TransformerOptimize:
    """
    Collects standard steps to train transformer
    call step_loss after computing each loss
    """
    def __init__(self, hypers: HypersBase, num_instances_to_train_over: int, model):
        self.step = 0
        self.global_step = 0
        self.hypers = hypers
        self.model = model
        instances_per_step = hypers.full_train_batch_size // hypers.gradient_accumulation_steps
        self.reporting = Reporting(recency_weight=0.0001 * instances_per_step)
        args = self.hypers

        self.t_total = num_instances_to_train_over // args.full_train_batch_size

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]

        warmup_instances = args.warmup_instances
        if hasattr(args, 'warmup_fraction') and args.warmup_fraction > 0 and args.warmup_instances <= 0:
            warmup_instances = args.warmup_fraction * num_instances_to_train_over
            logger.info(f'From {args.warmup_fraction} warm up fraction, computed warm up instances = {warmup_instances}')
        if warmup_instances < 0:
            warmup_instances = 0

        self.optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_instances // args.full_train_batch_size,
            num_training_steps=self.t_total
        )

        # Check if saved optimizer or scheduler states exist
        if args.resume_from and os.path.isfile(os.path.join(args.resume_from, "optimizer.pt")) and \
                os.path.isfile(os.path.join(args.resume_from, "scheduler.pt")):
            resume_from = args.resume_from
        elif os.path.isfile(os.path.join(args.model_name_or_path, "optimizer.pt")) and \
                os.path.isfile(os.path.join(args.model_name_or_path, "scheduler.pt")):
            resume_from = args.model_name_or_path
        else:
            resume_from = None
        if resume_from is not None:
            # Load in optimizer and scheduler states
            self.optimizer.load_state_dict(torch.load(os.path.join(resume_from, "optimizer.pt"), map_location='cpu'))
            self.scheduler.load_state_dict(torch.load(os.path.join(resume_from, "scheduler.pt"), map_location='cpu'))
            logger.info(f'loaded optimizer and scheduler from {resume_from}')

        if args.fp16:
            self.model, optimizer = amp.initialize(self.model, self.optimizer, opt_level=args.fp16_opt_level)

        # multi-gpu training (should be after apex fp16 initialization)
        if args.n_gpu > 1:
            # NOTE: won't work at O2, only O1
            self.model = torch.nn.DataParallel(self.model, device_ids=list(range(args.n_gpu)))

        # Distributed training (should be after apex fp16 initialization)
        if args.local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True,
            )
        # set_seed(args)
        assert args.per_gpu_train_batch_size * (args.n_gpu if args.n_gpu > 0 else 1) * \
               args.world_size * args.gradient_accumulation_steps == args.full_train_batch_size
        logger.info("***** Running training *****")
        logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
        logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d", args.full_train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", self.t_total)

    def should_continue(self):
        return self.global_step < self.t_total

    def backward_on_loss(self, loss, **moving_averages):
        if self.hypers.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training
        loss_val = loss.item()
        if self.hypers.gradient_accumulation_steps > 1:
            loss = loss / self.hypers.gradient_accumulation_steps
        if self.hypers.fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        self.reporting.moving_averages(loss=loss_val, **moving_averages)
        return loss_val

    def optimizer_step(self):
        if self.global_step >= self.t_total:
            logger.warning(f'Warning, exceeded total steps! {self.global_step} step of {self.t_total}')
            return False
        if (self.step + 1) % self.hypers.gradient_accumulation_steps == 0:
            if self.hypers.max_grad_norm > 0:
                if self.hypers.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optimizer), self.hypers.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.hypers.max_grad_norm)

            self.optimizer.step()
            self.scheduler.step()  # Update learning rate schedule
            self.model.zero_grad()
            self.global_step += 1
        self.step += 1

        if self.reporting.is_time():
            self.reporting.display()
            inst_count = self.hypers.world_size * self.hypers.n_gpu * self.hypers.per_gpu_train_batch_size * self.reporting.check_count
            learning_rate_scalar = self.scheduler.get_lr()[0]
            logger.info(f'{inst_count/self.reporting.elapsed_seconds()} instances per second; {inst_count} total ({learning_rate_scalar} learn rate)')
        return True

    def step_loss(self, loss, **moving_averages):
        loss_val = self.backward_on_loss(loss, **moving_averages)
        if self.optimizer_step():
            return loss_val
        else:
            return None
