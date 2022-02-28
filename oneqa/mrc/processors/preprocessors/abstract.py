import logging
from abc import ABCMeta, abstractmethod
from typing import Optional

from transformers import PreTrainedTokenizerFast
from datasets import Dataset


class AbstractPreProcessor(metaclass=ABCMeta):
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerFast,
                 stride: int,
                 max_seq_len: Optional[int] = None):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._tokenizer = tokenizer
        self._stride = stride
        self._max_seq_len = max_seq_len

    @abstractmethod
    def process_train(self, examples):  # TODO return type?
        pass

    @abstractmethod
    def process_eval(self, examples):  # TODO return type? one method with is_train param?
        pass

    @abstractmethod
    def adapt_dataset(self, dataset: Dataset) -> Dataset:
        pass