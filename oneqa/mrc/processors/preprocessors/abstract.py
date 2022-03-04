import logging
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Type

from datasets.arrow_dataset import Batch
from transformers import PreTrainedTokenizerFast, BatchEncoding
from datasets import Dataset


class AbstractPreProcessor(metaclass=ABCMeta):  # TODO type signatures and docstrings for all methods
    def __init__(self, 
                 tokenizer: PreTrainedTokenizerFast,
                 stride: int,
                 max_seq_len: Optional[int] = None,
                 negative_sampling_prob_when_has_answer: float = 0.01,
                 negative_sampling_prob_when_no_answer: float = 0.04,
                 num_workers: Optional[int] = None,
                 load_from_cache_file: bool = False):
        self._logger = logging.getLogger(self.__class__.__name__)
        self._tokenizer = tokenizer
        self._stride = stride
        self._max_seq_len = max_seq_len
        self._negative_sampling_prob_when_has_answer = negative_sampling_prob_when_has_answer
        self._negative_sampling_prob_when_no_answer = negative_sampling_prob_when_no_answer
        self._num_workers = num_workers
        self._load_from_cache_file = load_from_cache_file

        if not (0. <= self._negative_sampling_prob_when_has_answer <= 1.):
            raise ValueError(f"Expected 0 <= negative_sampling_prob_when_has_answer <= 1 but got: "
                             f"{self._negative_sampling_prob_when_has_answer:.02f}")

        if not (0. <= self._negative_sampling_prob_when_has_answer <= 1.):
            raise ValueError(f"Expected 0 <= negative_sampling_prob_when_no_answer <= 1 but got: "
                             f"{self._negative_sampling_prob_when_no_answer:.02f}")

    @abstractmethod
    def process_train(self, examples: Dataset) -> Tuple[Dataset, Dataset]:
        pass

    @abstractmethod
    def process_eval(self, examples: Dataset) -> Tuple[Dataset, Dataset]:
        pass

    @abstractmethod
    def adapt_dataset(self, dataset: Dataset, is_train: bool) -> Dataset:
        pass

    @abstractmethod
    def label_features_for_subsampling(self, tokenized_examples: BatchEncoding, examples: Batch) -> BatchEncoding:
        pass

    @abstractmethod
    def subsample_features(self, dataset: Dataset) -> Dataset:
        pass

    @abstractmethod
    def validate_schema(self, dataset: Dataset, is_train: bool, pre_adaptation: bool = True) -> None:
        pass
