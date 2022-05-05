import logging
from abc import ABCMeta, abstractmethod
from typing import List, Dict, Any

from datasets import Dataset


class AbstractPostProcessor(metaclass=ABCMeta):
    """
    Base class for post processors.
    """
    def __init__(self,
                 k: int,

                 max_answer_length: int, single_context_multiple_passages: bool = False):
        """
        Args:
            k: Max number of answers to return.
            max_answer_length: Maximum Answer Length.
            single_context_multiple_passages: See `AbstractPreProcessor` for more details.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._k = k

        self._max_answer_length = max_answer_length
        self._single_context_multiple_passages = single_context_multiple_passages

    @abstractmethod
    def process(self, examples: Dataset, features: Dataset, predictions: tuple):
        """
        Convert data and model predictions into MRC answers.
        """
        pass

    @abstractmethod
    def prepare_examples_as_references(self, examples: Dataset) -> List[Dict[str, Any]]:
        """
        Convert examples into references for use with metrics.
        """
        pass

    @abstractmethod
    def process_references_and_predictions(self, examples: Dataset, features: Dataset, predictions):
        """
        Convert data and model predictions into MRC answers and references for use in metrics.
        """
        pass
