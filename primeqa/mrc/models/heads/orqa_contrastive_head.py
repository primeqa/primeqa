from copy import deepcopy
from typing import Union, Optional

import torch
from torch.nn.functional import normalize
from torch.nn.functional import cosine_similarity
from transformers import PretrainedConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from primeqa.mrc.models.heads.abstract import AbstractTaskHead
from primeqa.mrc.data_models.model_outputs.extractive import ExtractiveQAModelOutput, ExtractiveQAWithConfidenceModelOutput
from primeqa.mrc.data_models.target_type import TargetType


class ExtractiveOpenNQContrastiveHead(AbstractTaskHead):
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

#        config_for_classification_head_2 = deepcopy(config)
#        config_for_classification_head_2.num_labels = 2
#        self.num_classification_head_labels_2 = config_for_classification_head_2.num_labels
#        self.classifier_2 = RobertaClassificationHead(config_for_classification_head_2)


    def forward(self,
                model_outputs: Union[tuple, BaseModelOutputWithPoolingAndCrossAttentions],
                pos_model_outputs: Union[tuple, BaseModelOutputWithPoolingAndCrossAttentions],
                neg_model_outputs: Union[tuple, BaseModelOutputWithPoolingAndCrossAttentions],
                num_ctrv_examples,
                start_positions=None,
                end_positions=None,
                target_type=None,
                pos_start_positions=None,
                pos_end_positions=None,
                neg_start_positions=None,
                neg_end_positions=None,
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

#        combined_cls = sequence_output[:, 0, :].sum(dim=0, keepdim=True).unsqueeze(0) / sequence_output.size()[0]
#        has_answer_logits = self.classifier_2(combined_cls)
        if start_positions is not None and end_positions is not None and target_type is not None:
#            if (start_positions.count_nonzero() > 0):
#                has_answer_label = torch.tensor([1], device=start_positions.device)
#            else:
#                has_answer_label = torch.tensor([0], device=start_positions.device)

            batch_size, num_passage = start_positions.size()
            start_positions = start_positions.view(batch_size * num_passage)
            end_positions = end_positions.view(batch_size * num_passage)
            target_type = target_type.view(batch_size * num_passage)

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

#            has_answer_loss = loss_fct(has_answer_logits, has_answer_label)
#            total_loss = (start_loss + end_loss + answer_type_loss + has_answer_loss) / 4
            total_loss = (start_loss + end_loss + answer_type_loss) / 3 # + has_answer_loss) / 3
            
            # This part only applies to batches that have contrastive examples
            if pos_model_outputs:
                pos_start_positions = pos_start_positions[:,0:num_ctrv_examples[0]]
                pos_end_positions = pos_end_positions[:,0:num_ctrv_examples[0]]
                phi_labels = torch.zeros([pos_start_positions.size()[1]], dtype=torch.int64, device=pos_start_positions.device) 
                pos_phi = self._compute_ctrv_phi(model_outputs, start_positions, end_positions,
                                                 pos_model_outputs, pos_start_positions, pos_end_positions)
                neg_start_positions = neg_start_positions[:,0:num_ctrv_examples[0]]
                neg_end_positions = neg_end_positions[:,0:num_ctrv_examples[0]]
                neg_phi = self._compute_ctrv_phi(model_outputs, start_positions, end_positions,
                                                 neg_model_outputs, neg_start_positions, neg_end_positions)

                phi = torch.cat((pos_phi/0.05, neg_phi/0.05), dim=1)
                
                ctrv_loss = loss_fct(phi,phi_labels)
                
                total_loss = 0.5*total_loss+0.5*ctrv_loss
        else:
            start_logits = start_logits.unsqueeze(0)
            end_logits = end_logits.unsqueeze(0)
            answer_type_logits = answer_type_logits.unsqueeze(0)

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

    def _compute_ctrv_phi(self, model_output, starts, ends, ctrv_model_output, ctrv_starts, ctrv_ends):
        batch_size, num_passage = ctrv_starts.size()
        ctrv_starts = ctrv_starts.view(batch_size * num_passage)
        ctrv_ends = ctrv_ends.view(batch_size * num_passage)
        ctrv_embeddings = ctrv_model_output['last_hidden_state']
        ctrv_start_embeddings = ctrv_embeddings[range(num_passage),ctrv_starts]
        ctrv_end_embeddings = ctrv_embeddings[range(num_passage),ctrv_ends]
        ctrv_z = torch.cat((ctrv_start_embeddings,ctrv_end_embeddings),1)
       
        orig_embeddings = model_output['last_hidden_state'][0:num_passage, :, :]
        orig_starts = starts[0:num_passage]
        orig_ends = ends[0:num_passage]
        orig_start_embeddings = orig_embeddings[range(num_passage),orig_starts]
        orig_end_embeddings = orig_embeddings[range(num_passage),orig_ends]
        orig_z = torch.cat((orig_start_embeddings,orig_end_embeddings),1)
        
        phi = cosine_similarity(orig_z,ctrv_z)
        phi = torch.unsqueeze(phi, dim=1)
        return phi
        


EXTRACTIVE_OPENNQ_CONTRASTIVE_HEAD = dict(qa_head=ExtractiveOpenNQContrastiveHead)

