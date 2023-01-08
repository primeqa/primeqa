import logging
from typing import Dict, Type

import torch
from transformers import PretrainedConfig, PreTrainedModel, MODEL_FOR_PRETRAINING_MAPPING, MODEL_MAPPING
from transformers.modeling_outputs import BaseModelOutput

from primeqa.mrc.models.heads.abstract import AbstractTaskHead

class EncoderWrapper(torch.nn.Module):
    """
    EncoderWrapper for the FiD model
    
    B - Batch size
    N the number of passages per example
    L the max seq length
    
    The EncoderWrapper transforms the input from  B * (N * L) to (B * N) * L
    Every passage of size L is encoded separatelly 
    After the encoder, concatenate encoder output for all N passages
    """
    def __init__(self, encoder, use_checkpoint=False):
        super().__init__()
        self.encoder = encoder
        self.main_input_name = encoder.main_input_name
    
    def forward(self, input_ids=None, attention_mask=None, return_dict=False, **kwargs):
        # total_length = n_passages * passage_length
        if input_ids.dim() == 3: # the generate() function directly calls the encoder, so we don't have chance to resize before encoder
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
    Generative language model for downstream tasks.  
    For the generative model the task head is not used
    """

    def __init__(self,
                 config: PretrainedConfig,
                 task_heads: Dict[str, Type[AbstractTaskHead]]):
        """
        Args:
            config: Model config
            task_heads: dict mapping task head name to constructor
                        Task heads are given in the input for easier integration with run_mrc.py 
                        The task head is expected to be None and not used in the code
        """
        super().__init__(config)
        self._logger = logging.getLogger(self.__class__.__name__)

        if type(self) is FiDModelForDownstreamTasks:
            raise TypeError(f"{FiDModelForDownstreamTasks.__class__.__name__} is not intended to be directly "
                            f"instantiated and should be subclassed together with a XPreTrainedModel type. "
                            f"See {self.model_class_from_config.__name__} or {self.from_config.__name__} "
                            f"for creating and instantiating these subclasses.")

        self._task_head = task_heads


    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=False,
                **kwargs):
    
        if input_ids != None:
            if input_ids.dim() == 3:
                encoder = self.get_encoder()
                encoder.n_passages = input_ids.size(1)
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
        encoder = self.get_encoder()
        encoder.n_passages = input_ids.size(1)
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
        encoder_model = model.get_encoder()
        model.set_encoder(EncoderWrapper(encoder_model))
        return model

    def set_task_head(self, task_head: str) -> None:
        pass
    
    def save_pretrained(self, *args, **kwargs,):
        encoder_wrapper = self.get_encoder()
        self.set_encoder(encoder_wrapper.encoder)
        super().save_pretrained(*args,**kwargs) 
        self.set_encoder(encoder_wrapper)

    def get_encoder(self):
        if hasattr(self, "model"):
            return self.model.encoder
        elif hasattr(self, "encoder"):
            return self.encoder
        else:
            raise NotImplementedError
        
    def set_encoder(self, encoder_model):
        if hasattr(self, "model"):
            self.model.encoder = encoder_model
        elif hasattr(self, "encoder"):
            self.encoder = encoder_model
        else:
            raise NotImplementedError            
