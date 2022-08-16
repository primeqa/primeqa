from copy import deepcopy
from typing import Union, Optional

import torch
from torch.nn.functional import normalize
from transformers import PretrainedConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from primeqa.mrc.models.heads.abstract import AbstractTaskHead
from primeqa.mrc.data_models.model_outputs.extractive import ExtractiveQAModelOutput, ExtractiveQAWithConfidenceModelOutput
from primeqa.mrc.data_models.target_type import TargetType


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
                target_type=None,
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


class ExtractiveQAWithConfidenceHead(AbstractTaskHead):
    """
    Task head for extractive Question Answering supporting confidence calibration.
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

        self.output_dropout_rate = getattr(config, "output_dropout_rate", 0.25)
        self.decoding_times_with_dropout = getattr(config, "decoding_times_with_dropout", 5)
        self.sep_token_id = getattr(config, "sep_token_id", None)

    def forward(self,
                model_outputs: Union[tuple, BaseModelOutputWithPoolingAndCrossAttentions],
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                start_positions=None,
                end_positions=None,
                target_type=None,
                **kwargs)-> Union[tuple, ExtractiveQAWithConfidenceModelOutput]:
        """
        Compute the task head's forward pass.

        Args:
            model_outputs: Language model outputs.
            input_ids: Indices of input sequence tokens in the vocabulary.
            attention_mask: Mask to avoid performing attention on padding token indices.
            token_type_ids: Segment token indices to indicate question and context portions of the inputs.
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
        start_stdev = None
        end_stdev = None
        query_passage_similarity = None
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

        else:
            # confidence features only generated for test mode
            # dropout feature
            lm_output_dropout_fct = torch.nn.Dropout(self.output_dropout_rate)
            start_logits_with_dropout = []
            end_logits_with_dropout = []
            for k in range(self.decoding_times_with_dropout):
                logits = self.qa_outputs(lm_output_dropout_fct(sequence_output))
                s_logits, e_logits = logits.split(1, dim=-1)
                start_logits_with_dropout.append(s_logits.squeeze(-1))
                end_logits_with_dropout.append(e_logits.squeeze(-1))
            start_stdev = torch.std(torch.stack(start_logits_with_dropout), axis=0)
            end_stdev = torch.std(torch.stack(end_logits_with_dropout), axis=0)

            # colbert style feature to measure the similarity between query and passage
            if attention_mask is not None and token_type_ids is not None:
                passage_mark = torch.mul(token_type_ids, attention_mask)
                query_mark = 1 - token_type_ids
                # the separator (first 0) between query and passage need be masked
                first_zero = query_mark.sum(dim=1)
                query_mark[torch.arange(first_zero.size()[0]), (first_zero - 1)] = 0
#                query_mask[torch.arange(first_zero.size()[0]), (first_zero - 2)] = 0
                # <cls> need be masked
                query_mark[:, 0] = 0
                normalized_output = normalize(sequence_output, p=2.0, dim = 2)
                # add 1 to product to make the value positive
                query_passage_product = normalized_output @ normalized_output.permute(0, 2, 1) + 1.0
                query_passage_similarity = ((query_passage_product * query_mark.unsqueeze(1)).max(2).values
                                            * passage_mark).sum(-1) / passage_mark.sum(-1)
            elif input_ids is not None and self.sep_token_id: # Roberta and XLM-R don't use token_type_ids
                position_ids = torch.arange(input_ids.size()[1]).repeat(input_ids.size()[0], 1).to(input_ids.device)
                # Find the position of the first spe_id in input_ids which is the end of query
                first_sep = (input_ids == self.sep_token_id).long().argmax(-1)
                query_mark = position_ids < first_sep[..., None]
                # <cls> need be masked
                query_mark[:, 0] = 0
                # Find the position of the last sep_id in input_ids which is the end of passage
                last_sep = (input_ids == self.sep_token_id).long().cumsum(-1).argmax(-1)
                attention_mark = position_ids < last_sep[..., None]
                passage_mark_from_left = position_ids > (first_sep + 1)[..., None]
                passage_mark = torch.mul(passage_mark_from_left, attention_mark)
                normalized_output = normalize(sequence_output, p=2.0, dim = 2)
                # add 1 to product to make the value positive
                query_passage_product = normalized_output @ normalized_output.permute(0, 2, 1) + 1.0
                query_passage_similarity = ((query_passage_product * query_mark.unsqueeze(1)).max(2).values
                                            * passage_mark).sum(-1) / passage_mark.sum(-1)
            else:
                query_passage_similarity = torch.zeros(sequence_output.size()[0], dtype=start_logits.dtype)

        # (loss), start_logits, end_logits, target_type_logits,
        # start_stdev, end_stdev, query_passage_similarity,
        # (hidden_states), (attentions)
        return_dict = isinstance(model_outputs, ModelOutput)
        if not return_dict:
            if start_stdev is not None and end_stdev is not None and query_passage_similarity is not None:
                output = (start_logits, end_logits, answer_type_logits, start_stdev, end_stdev,
                          query_passage_similarity) + model_outputs[2:]
            else:
                output = (start_logits, end_logits, answer_type_logits) + model_outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return ExtractiveQAWithConfidenceModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            target_type_logits=answer_type_logits,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            start_stdev=start_stdev,
            end_stdev=end_stdev,
            query_passage_similarity=query_passage_similarity
        )


EXTRACTIVE_WITH_CONFIDENCE_HEAD = dict(qa_head=ExtractiveQAWithConfidenceHead)



