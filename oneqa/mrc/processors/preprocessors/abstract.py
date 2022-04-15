import logging
from abc import ABCMeta, abstractmethod
from typing import Optional, Tuple, Type

from datasets.arrow_dataset import Batch
from transformers import PreTrainedTokenizerFast, BatchEncoding
from datasets import Dataset


class AbstractPreProcessor(metaclass=ABCMeta):
    """
    Abstract preprocessor which provides interface for all preprocessors.
    """

    def __init__(self,
                 tokenizer: PreTrainedTokenizerFast,
                 stride: int,
                 max_seq_len: Optional[int] = None,
                 negative_sampling_prob_when_has_answer: float = 0.01,
                 negative_sampling_prob_when_no_answer: float = 0.04,
                 num_workers: Optional[int] = None,
                 load_from_cache_file: bool = True,
                 max_q_char_len: int = 128,
                 single_context_multiple_passages: bool = False,
                 max_contexts: Optional[int] = None):
        """
        Args:
            tokenizer:
                Tokenizer used to prepare model inputs.
            stride:
                Step size to move sliding window across context.
            max_seq_len:
                Maximum length of question and context inputs to the model (in word pieces/bpes).
                Uses tokenizer default if not given.
            negative_sampling_prob_when_has_answer:
                Probability to select a negative feature from an example which has an answer.
            negative_sampling_prob_when_no_answer:
                Probability to select a negative feature from an example which does not have an answer.
            num_workers:
                Number of workers to use for preprocessing.
                Uses all available logical cores by default.
            load_from_cache_file:
                Whether to attempt loading features from cache file.
            max_q_char_len :
                Max length allowed per question (in characters).
                Remainder will be trimmed.
            single_context_multiple_passages:
                Iff true allow multiple context passages from the same example in the same feature span.
                Note some preprocessors may override this parameter.
            max_contexts:
                Maximum number of contexts to search per example.
                Remainder will be trimmed.
                Defaults to searching all contexts.
        """
        self._logger = logging.getLogger(self.__class__.__name__)
        self._tokenizer = tokenizer
        self._stride = stride
        self._max_seq_len = max_seq_len
        self._negative_sampling_prob_when_has_answer = negative_sampling_prob_when_has_answer
        self._negative_sampling_prob_when_no_answer = negative_sampling_prob_when_no_answer
        self._num_workers = num_workers
        self._load_from_cache_file = load_from_cache_file
        self._max_q_char_len = max_q_char_len
        self._single_context_multiple_passages = single_context_multiple_passages
        self._max_contexts = max_contexts

        if not (0. <= self._negative_sampling_prob_when_has_answer <= 1.):
            raise ValueError(f"Expected 0 <= negative_sampling_prob_when_has_answer <= 1 but got: "
                             f"{self._negative_sampling_prob_when_has_answer:.02f}")

        if not (0. <= self._negative_sampling_prob_when_has_answer <= 1.):
            raise ValueError(f"Expected 0 <= negative_sampling_prob_when_no_answer <= 1 but got: "
                             f"{self._negative_sampling_prob_when_no_answer:.02f}")

    @abstractmethod
    def process_train(self, examples: Dataset) -> Tuple[Dataset, Dataset]:
        """
        Process training examples into features.

        Args:
            examples: examples to process into features.

        Returns:
            tuple (examples, features) comprising examples adapted into standardized format and processed input features for model.
        """
        pass

    @abstractmethod
    def process_eval(self, examples: Dataset) -> Tuple[Dataset, Dataset]:
        """
        Process eval examples into features.

        Args:
            examples: examples to process into features.

        Returns:
            tuple (examples, features) comprising examples adapted into standardized format and processed input features for model.
        """
        pass

    @abstractmethod
    def adapt_dataset(self, dataset: Dataset, is_train: bool) -> Dataset:
        """
        Convert dataset into standardized format accepted by the preprocessor.
        This method will likely need to be overridden when subclassing.

        Args:
            dataset: data to adapt.
            is_train: whether the dataset is for training.

        Returns:
            Adapted dataset.
        """
        pass

    @abstractmethod
    def label_features_for_subsampling(self, tokenized_examples: BatchEncoding, examples: Batch) -> BatchEncoding:
        """
        Annotate each training feature with a 'subsample_type' of type `SubsampleType` for subsampling.

        Args:
            tokenized_examples: featurized examples to annotate.
            examples: original examples corresponding to the `tokenized_examples` features.

        Returns: `tokenized_examples` annotated with 'subsample_type' for subsampling.

        """
        pass

    @abstractmethod
    def subsample_features(self, dataset: Dataset) -> Dataset:
        """
        Subsample training features according to 'subsample_type':

        * All positive features are kept.
        * All negative features from an example that has an answer are kept with probability `self._negative_sampling_prob_when_has_answer`.
        * All negative features from an example that has no answer are kept with probability `self._negative_sampling_prob_when_no_answer`.

        Args:
            dataset: features to subsample.

        Returns:
            subsampled features.
        """
        pass

    @abstractmethod
    def validate_schema(self, dataset: Dataset, is_train: bool, pre_adaptation: bool = True) -> None:
        """
        Validate the data schema is correct for this preprocessor.

        Args:
            dataset: data to validate schema of
            is_train: whether the data is for training
            pre_adaptation: whether adapt_dataset has been called. This allows for optional fields
                            (e.g. example_id) to be imputed during adaptation.

        Returns:
            None

        Raises:
            ValueError: The data is not in the correct schema.
        """
        pass
