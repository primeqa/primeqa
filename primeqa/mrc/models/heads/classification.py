from copy import deepcopy
from typing import Union, Optional

import torch
from torch.nn.functional import normalize
from transformers import PretrainedConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.roberta.modeling_roberta import RobertaClassificationHead

from primeqa.mrc.models.heads.abstract import AbstractTaskHead
from primeqa.mrc.data_models.model_outputs.classification import ClassificationModelOutput
from primeqa.mrc.data_models.target_type import TargetType


class ClassificationHead(AbstractTaskHead):
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

        config_for_classification_head = deepcopy(config)
        if num_labels_override is not None:
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
                labels=None,
                **kwargs) -> Union[tuple, ClassificationModelOutput]:
        """
        Compute the task head's forward pass.

        Args:
            model_outputs: Language model outputs.
            target_type: (training only) Ground-truth target type.

        Returns:
            Extractive QA task head result in data structure corresponding to type of `model_outputs`.
        """
        sequence_output = model_outputs[0]
        answer_type_logits = self.classifier(sequence_output)

        total_loss = None
        if labels is not None:
            # Predict target answer type for the whole question answer pair
            loss_fct = torch.nn.CrossEntropyLoss()
            total_loss = loss_fct(answer_type_logits, labels)


        return ClassificationModelOutput(
            loss=total_loss,
            target_type_logits=answer_type_logits
        )



CLASSIFICATION_HEAD = dict(qa_head=ClassificationHead)




