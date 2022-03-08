import logging
from typing import Dict, Type

import torch
from transformers import PretrainedConfig, PreTrainedModel, MODEL_FOR_PRETRAINING_MAPPING, MODEL_MAPPING

from oneqa.mrc.models.heads.abstract import AbstractTaskHead


# def create_task_model_class_from_config(config: PretrainedConfig) -> Type['ModelForDownstreamTasks']:
#     ptm_base_class = MODEL_FOR_PRETRAINING_MAPPING[config.__class__]
#     # inner_model_class = MODEL_MAPPING[config]
#     # base_model_prefix = getattr(ptm_base_class, 'base_model_prefix', config.model_type)
#     # model_type = config.model_type
#     # model_name = ''.join(map(str.title, re.split(r'[^\w\d]', model_type)))
#     model_name = config.__class__.__name__.rstrip('Config')
#     class_name = f'{model_name}{ModelForDownstreamTasks.__name__}'
#     model_class = type(class_name, (ModelForDownstreamTasks, ptm_base_class), {})
#     return model_class


class ModelForDownstreamTasks(PreTrainedModel):
    # model_class: Type[PreTrainedModel] = None

    def __init__(self,
                 config: PretrainedConfig,
                 task_heads: Dict[str, Type[AbstractTaskHead]]):
        super().__init__(config)
        self._logger = logging.getLogger(self.__class__.__name__)

        if type(self) is ModelForDownstreamTasks:
            raise TypeError(f"{ModelForDownstreamTasks.__class__.__name__} is not intended to be directly "
                            f"instantiated and should be subclassed together with a XPreTrainedModel type. "
                            f"See {self.model_class_from_config.__name__} or {self.from_config.__name__} "
                            f"for creating and instantiating these subclasses.")

        if not task_heads:
            raise ValueError("No task heads provided")

        self._task_head = None

        # Set the model to match the pre-trained name (e.g. self.roberta) so it can be loaded from pretrained
        setattr(self, self.base_model_prefix, MODEL_MAPPING[config.__class__](config))

        self.task_heads = torch.nn.ModuleDict(
            {name: model(config) for name, model in task_heads.items()})
        self.init_weights()

    @property
    def model_(self):  # using 'model' instead of 'model_' causes conflicts with some LMs (e.g. BART)
        """
        Returns the underlying language model. This is an alias to simplify access.
        """
        return getattr(self, self.base_model_prefix)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                **kwargs):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if not self._task_head:
            raise ValueError(f"Must provide task head by calling {ModelForDownstreamTasks.set_task_head.__name__}.")

        outputs = self.model_(
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
        task_head = self.task_heads[self._task_head]
        return task_head(outputs, **kwargs)

    @classmethod
    def model_class_from_config(cls, config: PretrainedConfig) -> Type['ModelForDownstreamTasks']:
        """
        Dynamically creates a model class from a PreTrainedConfig.
        """
        ptm_base_class = MODEL_FOR_PRETRAINING_MAPPING[config.__class__]
        model_name = config.__class__.__name__.rstrip('Config')
        class_name = f'{model_name}{cls.__name__}'
        model_class = type(class_name, (cls, ptm_base_class), {})

        # noinspection PyTypeChecker
        return model_class

    @classmethod
    def from_config(cls, config: PretrainedConfig, *args, **kwargs) -> 'ModelForDownstreamTasks':
        """
        Dynamically creates a model class from a PreTrainedConfig and then uses the config
        with args/kwargs to instantiate.
        """
        model_class = cls.model_class_from_config(config)
        model = model_class.from_pretrained(*args, config=config, **kwargs)
        return model

    def set_task_head(self, task_head: str):
        if self._task_head is not None:
            self._logger.info(f"Changing default task head from '{self._task_head}' to '{task_head}'")
        else:
            self._logger.info(f"Setting task head for first time to '{self._task_head}'")
        self._task_head = task_head

    def clear_task_head(self):
        if self._task_head is not None:
            self._logger.info(f"Clearing default task head '{self._task_head}'")
            self._task_head = None
        else:
            self._logger.info("Requested to clear task head but is already cleared")
