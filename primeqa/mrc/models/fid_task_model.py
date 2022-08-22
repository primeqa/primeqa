import logging
from typing import Dict, Type

import torch
from transformers import PretrainedConfig, PreTrainedModel, MODEL_FOR_PRETRAINING_MAPPING, MODEL_MAPPING
from transformers.modeling_outputs import BaseModelOutput

from primeqa.mrc.models.heads.abstract import AbstractTaskHead

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

class FiDModelForDownstreamTasks(PreTrainedModel):
    """
    Language model for downstream tasks.  Tasks are implemented via task heads which subclass `AbstractTaskHead`.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 task_heads: Dict[str, Type[AbstractTaskHead]]):
        """
        Args:
            config: Model config
            task_heads: dict mapping task head name to constructor
        """
        super().__init__(config)
        self._logger = logging.getLogger(self.__class__.__name__)

        if type(self) is FiDModelForDownstreamTasks:
            raise TypeError(f"{FiDModelForDownstreamTasks.__class__.__name__} is not intended to be directly "
                            f"instantiated and should be subclassed together with a XPreTrainedModel type. "
                            f"See {self.model_class_from_config.__name__} or {self.from_config.__name__} "
                            f"for creating and instantiating these subclasses.")

        # if not task_heads:
        #     raise ValueError("No task heads provided")

        self._task_head = None

        # Set the model to match the pre-trained name (e.g. self.roberta) so it can be loaded from pretrained
        setattr(self, self.base_model_prefix, MODEL_MAPPING[config.__class__](config))

        # self.task_heads = torch.nn.ModuleDict({
        #     name: model(config) for name, model in task_heads.items()
        # })
        self.init_weights()

    @property
    def model_(self) -> PreTrainedModel:  # using 'model' instead of 'model_' causes conflicts with some LMs (e.g. BART)
        """
        Returns the underlying language model. This is an alias to simplify access.
        """
        return getattr(self, self.base_model_prefix)

    @property
    def task_head(self) -> AbstractTaskHead:
        """
        Return the current task head or raises a `ValueError` if it has not yet been set.
        """
        if self._task_head is not None:
            # noinspection PyTypeChecker
            return self.task_heads[self._task_head]
        else:
            raise ValueError(f"Task head is not set.  Call {FiDModelForDownstreamTasks.set_task_head.__name__} to set it")

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                **kwargs):
    
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
        
    @classmethod
    def model_class_from_config(cls, config: PretrainedConfig) -> Type['FiDModelForDownstreamTasks']:
        """
        Dynamically creates and returns a model class from a PreTrainedConfig.
        """
        ptm_base_class = MODEL_FOR_PRETRAINING_MAPPING[config.__class__]
        model_name = config.__class__.__name__.rstrip('Config')
        class_name = f'{model_name}{cls.__name__}'
        model_class = type(class_name, (cls, ptm_base_class), {})

        # noinspection PyTypeChecker
        return model_class

    @classmethod
    def from_config(cls, config: PretrainedConfig, *args, **kwargs) -> 'FiDModelForDownstreamTasks':
        """
        Dynamically creates a model class from a PreTrainedConfig and then uses the config
        with `args` or `kwargs` to instantiate and return a model.
        """
        model_class = cls.model_class_from_config(config)
        model = model_class.from_pretrained(*args, config=config, **kwargs)
        model.model.encoder = EncoderWrapper(model.model.encoder)
        return model

    def set_task_head(self, task_head: str) -> None:
        """
        Args:
            task_head: name of the task head to activate.

        Raises:
            KeyError: model does not have task head with name `task_head`.
        """
        if task_head not in self.task_heads:
            raise KeyError(f"Task head '{task_head}' not in task_heads: {list(self.task_heads)}")
        elif self._task_head is not None:
            self._logger.info(f"Changing default task head from '{self._task_head}' to '{task_head}'")
        else:
            self._logger.info(f"Setting task head for first time to '{self._task_head}'")
        self._task_head = task_head
