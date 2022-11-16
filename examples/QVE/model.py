# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model. """

import math
import os
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
from torch.distributions.binomial import Binomial

from transformers.activations import ACT2FN
from transformers import BertConfig
from transformers.file_utils import (
    add_code_sample_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
)

from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BertPooler,
    BertModel,
)

from copy import deepcopy
from typing import Union, Optional

import torch
from torch.nn.functional import normalize
from transformers import PretrainedConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from primeqa.mrc.models.heads.abstract import AbstractTaskHead
from primeqa.mrc.data_models.model_outputs.extractive import ExtractiveQAModelOutput
from primeqa.mrc.data_models.target_type import TargetType


class BertPoolerMarginal(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sequential1 = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.Tanh(),
            nn.Linear(config.hidden_size, 64),
            nn.Tanh()
        )
        self.sequential2 = nn.Sequential(
            nn.Linear(66, config.hidden_size),
            nn.Tanh()
        )

    def forward(self, hidden_states, marginal_info):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = self.sequential1(hidden_states)
        pooled_output = torch.cat((pooled_output, marginal_info), dim=1)
        pooled_output = self.sequential2(pooled_output)
        return pooled_output


class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.pooler = BertPoolerMarginal(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            marginal_info=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        if self.config.marginal:
            pooled_output = self.pooler(pooled_output, marginal_info)

        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class ExtractiveQAHeadForQVE(AbstractTaskHead):
    """
    Task head for extractive Question Answering.
    """
    def __init__(self, config: PretrainedConfig, num_labels_override: Optional[int] = None):
        """
        Args:
            config: Language model config.
            num_labels_override: Set this to override number of answer types from default `len(TargetType)`.
        """
        super().__init__(config)
        self.num_labels = config.num_labels
        self.qa_outputs = torch.nn.Linear(config.hidden_size, self.num_labels)

        config_for_classification_head = deepcopy(config)
        if num_labels_override is None:
            config_for_classification_head.num_labels = len(TargetType)
        else:
            config_for_classification_head.num_labels = num_labels_override
        self.num_classification_head_labels = config_for_classification_head.num_labels

        dropout_names = ["classifier_dropout", "hidden_dropout_prob", "classifier_dropout_prob"]
        for name in dropout_names:
            dropout_value = getattr(config_for_classification_head, name, None)
            if dropout_value is not None:
                self._logger.info(f"Loading dropout value {dropout_value} from config attribute '{name}'")
                config_for_classification_head.classifier_dropout = dropout_value
                break
        else:
            self._logger.warning("No dropout value found -- setting to 0")
            config_for_classification_head.classifier_dropout = 0.

        self.classifier = RobertaClassificationHead(config_for_classification_head)

    def forward(self,
                model_outputs: Union[tuple, BaseModelOutputWithPoolingAndCrossAttentions],
                start_positions=None,
                end_positions=None,
                input_values=None,
                **kwargs) -> Union[tuple, ExtractiveQAModelOutput]:
        """
        Compute the task head's forward pass.

        Args:
            model_outputs: Language model outputs.
            start_positions: (training only) Ground-truth start positions.
            end_positions: (training only) Ground-truth end positions.
            target_type: (training only) Ground-truth target type.

        Returns:
            Extractive QA task head result in data structure corresponding to type of `model_outputs`.
        """
        sequence_output = model_outputs[0]

        # Predict start and end logits by token
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)


            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction='none')
            # weighted training
            # Exception (When selection probability is 0)
            if input_values is None:
                input_values = torch.ones_like(start_positions)

            while input_values.sum() == 0:
                input_values = 0.5 * torch.ones_like(input_values)
                input_values = Binomial(1, input_values).sample()

            start_loss = loss_fct(start_logits, start_positions) * input_values
            start_loss = start_loss.sum() / input_values.sum()
            end_loss = loss_fct(end_logits, end_positions) * input_values
            end_loss = end_loss.sum() / input_values.sum()
            total_loss = (start_loss + end_loss) / 2


        return_dict = isinstance(model_outputs, ModelOutput)
        if not return_dict:
            output = (start_logits, end_logits) + model_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return ExtractiveQAModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )

EXTRACTIVE_HEAD = dict(qa_head=ExtractiveQAHeadForQVE)
