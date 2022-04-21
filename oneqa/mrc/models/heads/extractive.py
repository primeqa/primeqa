from copy import deepcopy
from typing import Union, Optional

import torch
from transformers import PretrainedConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from oneqa.mrc.models.heads.abstract import AbstractTaskHead
from oneqa.mrc.data_models.model_outputs.extractive import ExtractiveQAModelOutput
from oneqa.mrc.data_models.target_type import TargetType


class ExtractiveQAHead(AbstractTaskHead):
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
                target_type=None) -> Union[tuple, ExtractiveQAModelOutput]:
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

        # Predict target answer type for the whole question answer pair
        answer_type_logits = self.classifier(sequence_output)

        # Predict start and end logits by token
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None and target_type is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            if len(target_type.size()) > 1:
                target_type = target_type.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            answer_type_loss = loss_fct(answer_type_logits, target_type)
            total_loss = (start_loss + end_loss + answer_type_loss) / 3

        # (loss), start_logits, end_logits, target_type_logits, (hidden_states), (attentions)
        return_dict = isinstance(model_outputs, ModelOutput)
        if not return_dict:
            output = (start_logits, end_logits, answer_type_logits) + model_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return ExtractiveQAModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            target_type_logits=answer_type_logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
        )


EXTRACTIVE_HEAD = dict(qa_head=ExtractiveQAHead)
