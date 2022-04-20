import logging
from abc import ABCMeta, abstractmethod
from typing import Union

import torch
from transformers import PretrainedConfig
from transformers.file_utils import ModelOutput


class AbstractTaskHead(torch.nn.Module, metaclass=ABCMeta):
    """
    Base class for task heads.
    """
    def __init__(self, config: PretrainedConfig):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)
        self.config = config

    @abstractmethod
    def forward(self, model_outputs: Union[tuple, ModelOutput], *args, **kwargs) -> Union[tuple, ModelOutput]:
        """
        Args:
            model_outputs: Language model outputs.
            *args: Additional args for task head.
            **kwargs: Additional keyword args for task head.

        Returns:
            Task head result data structure corresponding to type of `model_outputs`.
        """
        raise NotImplementedError
