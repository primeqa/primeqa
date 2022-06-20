# from data collator (index, target_ids, target_mask, passage_ids, passage_masks)

import torch
import torch.nn.functional as F
from torch import nn
import random
import json
import numpy as np
import transformers
from transformers.modeling_outputs import BaseModelOutput


class FiDBART(transformers.BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.wrap_encoder()
    def forward_(self, **kwargs):
        if "input_ids" in kwargs:
            kwargs["input_ids"] = kwargs["input_ids"].view(kwargs["input_ids"].size(0), -1)
        if "attention_mask" in kwargs:
            kwargs["attention_mask"] = kwargs["attention_mask"].view(kwargs["attention_mask"].size(0), -1)

        return super(FiDBART, self).forward(
            **kwargs
        )    
    # We need to resize as B x (N * L) instead of (B * N) x L here
    # because the T5 forward method uses the input tensors to infer
    # dimensions used in the decoder.
    # EncoderWrapper resizes the inputs as (B * N) x L.
    def forward(self, input_ids=None, attention_mask=None, labels=None, return_dict=False, **kwargs): # set return_dict=False, as we need encoder outputs as tuple. TODO verify this
        if input_ids != None:
            if input_ids.dim() == 3:
                self.model.encoder.n_passages = input_ids.size(1)
            input_ids = input_ids.view(input_ids.size(0), -1)
        if attention_mask != None:
            attention_mask = attention_mask.view(attention_mask.size(0), -1)
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            **kwargs
        ) # tuple: (loss, lm_logits,) + BartModel outputs
        return outputs
    
    def generate(self, input_ids, **gen_kwargs):
        self.model.encoder.n_passages = input_ids.size(1)
        return super().generate(
            input_ids,
            **gen_kwargs
        )
    
    def wrap_encoder(self, use_checkpoint=False):
        """
        Wrap encoder to obtain an FiD model
        """
        self.model.encoder = EncoderWrapper(self.model.encoder, use_checkpoint=use_checkpoint)

    def unwrap_encoder(self):
        """
        Unwrap FiD, useful to load weights
        """
        self.model.encoder = self.model.encoder.encoder
        # TODO the original code assign layers, do we need to do this?
        # layers = []
        # for mod in self.model.encoder.layers:
        #     layers.append(mod.modules())
        # layers = nn.ModuleList(layers)
        # self.model.encoder.layers = layers

    def load_pretrained(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()
    
    def set_checkpoint(self, use_checkpoint):
        """
        Enable or disable checkpointing in the encoder.
        See https://pytorch.org/docs/stable/checkpoint.html
        """
        for mod in self.model.encoder.encoder.layers:
            mod.use_checkpoint = use_checkpoint
    
    # it was load_t5 in the original repo, we modify it to BART
    def load_pretrained(self, state_dict):
        self.unwrap_encoder()
        self.load_state_dict(state_dict)
        self.wrap_encoder()


class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()
        self.encoder = encoder
        # TODO checkpointing
    
    def forward(self, input_ids=None, attention_mask=None, return_dict=False, **kwargs):
        # total_length = n_passages * passage_length
        if input_ids.dim() == 3: # the generate() function directly call the encoder, so we don't have chance to resize before encoder TODO
            input_ids = input_ids.view(input_ids.size(0), -1)
        bsz, total_length = input_ids.shape # B * (N * L)
        passage_length = total_length // self.n_passages
        input_ids = input_ids.view(bsz*self.n_passages, passage_length) # resize to (B * N) * L
        attention_mask = attention_mask.view(bsz*self.n_passages, passage_length)
        outputs = self.encoder(input_ids, attention_mask, **kwargs)

        if not return_dict:
            return (outputs[0].view(bsz, self.n_passages*passage_length, -1), ) + outputs[1:] # concatenate encoder outputs #TODO support when return_dict=True

        return BaseModelOutput( # TODO pass hidden_states and attentions
            last_hidden_state=outputs[0].view(bsz, self.n_passages*passage_length, -1),
        )



        
    
