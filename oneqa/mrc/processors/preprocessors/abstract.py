import logging
from abc import ABCMeta, abstractmethod
from typing import Optional

from transformers import PreTrainedTokenizerFast, BatchEncoding
from datasets import Dataset


class AbstractPreProcessor(metaclass=ABCMeta):
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerFast,
                 stride: int,
                 max_seq_len: Optional[int] = None,
                 negative_sampling_prob_when_has_answer: float = 0.01,
                 negative_sampling_prob_when_no_answer: float = 0.04):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._tokenizer = tokenizer
        self._stride = stride
        self._max_seq_len = max_seq_len
        self._negative_sampling_prob_when_has_answer = negative_sampling_prob_when_has_answer
        self._negative_sampling_prob_when_no_answer = negative_sampling_prob_when_no_answer

    @abstractmethod
    def process_train(self, examples: Dataset) -> BatchEncoding:
        pass

    @abstractmethod
    def process_eval(self, examples: Dataset) -> BatchEncoding:
        pass

    @abstractmethod
    def adapt_dataset(self, dataset: Dataset) -> Dataset:
        pass

    @abstractmethod
    def label_features_for_subsampling(self, tokenized_examples: BatchEncoding) -> BatchEncoding:
        pass

    @abstractmethod
    def subsample_features(self, dataset: Dataset) -> Dataset:
        pass