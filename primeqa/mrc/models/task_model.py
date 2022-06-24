import logging
from typing import Dict, Type

import torch
from transformers import PretrainedConfig, PreTrainedModel, MODEL_FOR_PRETRAINING_MAPPING, MODEL_MAPPING

from primeqa.mrc.models.heads.abstract import AbstractTaskHead


class ModelForDownstreamTasks(PreTrainedModel):
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

        self.task_heads = torch.nn.ModuleDict({
            name: model(config) for name, model in task_heads.items()
        })
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
            raise ValueError(f"Task head is not set.  Call {ModelForDownstreamTasks.set_task_head.__name__} to set it")

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
        """
        Returns task head applied to language model outputs and any additional arguments supplied via `kwargs`.
        See HF transformers documentation for more details on other parameters.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        return self.task_head(outputs, input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)

    @classmethod
    def model_class_from_config(cls, config: PretrainedConfig) -> Type['ModelForDownstreamTasks']:
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
    def from_config(cls, config: PretrainedConfig, *args, **kwargs) -> 'ModelForDownstreamTasks':
        """
        Dynamically creates a model class from a PreTrainedConfig and then uses the config
        with `args` or `kwargs` to instantiate and return a model.
        """
        model_class = cls.model_class_from_config(config)
        model = model_class.from_pretrained(*args, config=config, **kwargs)
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
