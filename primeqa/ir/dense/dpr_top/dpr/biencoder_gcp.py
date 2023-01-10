from primeqa.ir.dense.dpr_top.dpr.biencoder_hypers import BiEncoderHypers
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from transformers import (DPRQuestionEncoder, DPRContextEncoder)
import os
from typing import Union
import logging

logger = logging.getLogger(__name__)


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask, dummy):
        pooler_output = self.encoder(input_ids, attention_mask)[0]
        return pooler_output


class BiEncoder(torch.nn.Module):
    """
    This trains the DPR encoders to maximize dot product between queries and positive contexts.
    We only use this model during training.
    """
    def __init__(self, hypers: BiEncoderHypers):
        super().__init__()
        self.hypers = hypers
        logger.info(f'BiEncoder: initializing from {hypers.qry_encoder_name_or_path} and {hypers.ctx_encoder_name_or_path}')
        self.qry_model = EncoderWrapper(DPRQuestionEncoder.from_pretrained(hypers.qry_encoder_name_or_path))
        self.ctx_model = EncoderWrapper(DPRContextEncoder.from_pretrained(hypers.ctx_encoder_name_or_path))
        self.saved_debug = False

    def encode(self, model, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if 0 < self.hypers.encoder_gpu_train_limit:
            # checkpointing
            # dummy requries_grad to deal with checkpointing issue:
            #   https://discuss.pytorch.org/t/checkpoint-with-no-grad-requiring-inputs-problem/19117/13
            dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)
            all_pooled_output = []
            for sub_bndx in range(0, input_ids.shape[0], self.hypers.encoder_gpu_train_limit):
                sub_input_ids = input_ids[sub_bndx:sub_bndx+self.hypers.encoder_gpu_train_limit]
                sub_attention_mask = attention_mask[sub_bndx:sub_bndx + self.hypers.encoder_gpu_train_limit]
                pooler_output = checkpoint(model, sub_input_ids, sub_attention_mask, dummy_tensor)
                all_pooled_output.append(pooler_output)
            return torch.cat(all_pooled_output, dim=0)
        else:
            return model(input_ids, attention_mask, None)

    def gather(self, tensor):
        dtensor = tensor.detach()
        gather_list = [torch.zeros_like(dtensor) for _ in range(self.hypers.world_size)]
        torch.distributed.all_gather(gather_list, dtensor)
        gather_list[self.hypers.global_rank] = tensor
        return torch.cat(gather_list, 0)

    def save_for_debug(self, qry_reps, ctx_reps, positive_indices):
        if self.hypers.global_rank == 0 and not self.saved_debug and \
                self.hypers.debug_location and not os.path.exists(self.hypers.debug_location):
            os.makedirs(self.hypers.debug_location)
            torch.save(qry_reps, os.path.join(self.hypers.debug_location, 'qry_reps.bin'))
            torch.save(ctx_reps, os.path.join(self.hypers.debug_location, 'ctx_reps.bin'))
            torch.save(positive_indices, os.path.join(self.hypers.debug_location, 'positive_indices.bin'))
            self.saved_debug = True
            logger.warning(f'saved debug info at {self.hypers.debug_location}')

    def forward(
        self,
        input_ids_q: torch.Tensor,
        attention_mask_q: torch.Tensor,
        input_ids_c: torch.Tensor,
        attention_mask_c: torch.Tensor,
        positive_indices: torch.Tensor
    ):
        """
        All batches must be the same size (q and c are fixed during training)
        :param input_ids_q: q x seq_len_q [0, vocab_size)
        :param attention_mask_q: q x seq_len_q [0, 1]
        :param input_ids_c: c x seq_len_c
        :param attention_mask_c: c x seq_len_c
        :param positive_indices: q [0, c)
        :return:
        """
        qry_reps = self.encode(self.qry_model, input_ids_q, attention_mask_q)
        ctx_reps = self.encode(self.ctx_model, input_ids_c, attention_mask_c)
        if self.hypers.world_size > 1:
            # also gather across processes
            positive_indices = self.gather(positive_indices + (self.hypers.global_rank * ctx_reps.shape[0]))
            qry_reps = self.gather(qry_reps)
            ctx_reps = self.gather(ctx_reps)
        # for debugging, lets save qry_reps, ctx_reps and positive_indices on the first pass
        self.save_for_debug(qry_reps, ctx_reps, positive_indices)
        dot_products = torch.matmul(qry_reps, ctx_reps.transpose(0, 1))  # (q * world_size) x (c * world_size)
        probs = F.log_softmax(dot_products, dim=1)
        loss = F.nll_loss(probs, positive_indices)
        predictions = torch.max(probs, 1)[1]
        accuracy = (predictions == positive_indices).sum() / positive_indices.shape[0]
        return loss, accuracy

    def save(self, save_dir: Union[str, os.PathLike]):
        self.qry_model.encoder.save_pretrained(os.path.join(save_dir, 'qry_encoder'))
        self.ctx_model.encoder.save_pretrained(os.path.join(save_dir, 'ctx_encoder'))
