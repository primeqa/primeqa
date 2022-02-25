import logging
from abc import ABCMeta, abstractmethod
from typing import Optional

from transformers import PreTrainedTokenizerFast
from datasets import Dataset


class AbstractPreProcessor(metaclass=ABCMeta):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, stride: int, max_q_len: int,
                 max_seq_len: Optional[int] = None):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._tokenizer = tokenizer  # TODO: fast toks only??
        self._stride = stride
        self._max_q_len = max_q_len
        self._max_seq_len = max_seq_len or self._tokenizer.max_len_sentences_pair
        self._max_c_len = self._max_seq_len - self._max_q_len

    @abstractmethod
    def process_train(self, examples):  # TODO return type?
        pass

    @abstractmethod
    def process_eval(self, examples):  # TODO return type? one method with is_train param?
        pass

    @abstractmethod
    def adapt_dataset(self, dataset: Dataset) -> Dataset:  # TODO return type
        pass