import itertools
from lib2to3.pgen2.tokenize import tokenize
import random
import uuid
from operator import sub
from typing import List, Iterable, Tuple, Any, Dict, Union
import logging

from datasets.arrow_dataset import Batch
from transformers import BatchEncoding
from datasets import Dataset
from datasets.features.features import Sequence, Value
from transformers import PreTrainedTokenizerFast, BatchEncoding


from primeqa.mrc.processors.preprocessors.abstract import AbstractPreProcessor
from primeqa.mrc.data_models.subsample_type import SubsampleType
from primeqa.mrc.data_models.target_type import TargetType

logger = logging.getLogger(__name__)


# AbstractPreProcessor is too specific to extractive, we can't inherit from it here
class TextClassifierPreProcessor:
    """
    Preprocessor for text classifier
    """    
    def __init__(self,
                example_id_key: str,
                sentence1_key: str,
                sentence2_key: str,
                language_key: str,
                tokenizer: PreTrainedTokenizerFast,
                max_seq_len: int,
                padding: bool,
                label_list: List,
                load_from_cache_file: bool = True):
        """
        Args:
            sentence1_key:
                the key for the first input field, typically "question"
            sentence2_key:
                the key for the second input field which is used as a passage
            tokenizer:
                Tokenizer used to prepare model inputs.             
                Step size to move sliding window across context.
            max_seq_len:
                Maximum length of question and context inputs to the model (in word pieces/bpes).
                Uses tokenizer default if not given.
            padding:
                padding argument for tokenizer
            load_from_cache_file:
                load_from_cache argument of dataset mapper
        """                
        self._example_id_key=example_id_key
        self._sentence1_key=sentence1_key
        self._sentence2_key=sentence2_key
        self._language_key=language_key
        self._tokenizer=tokenizer
        self._max_seq_len=max_seq_len
        self._padding=padding
        self._load_from_cache_file=load_from_cache_file
        self._label_list = label_list





    def process_train(self, examples: Dataset) -> Tuple[Dataset, Dataset]:
        return self._process(examples, is_train=True)

    def process_eval(self, examples: Dataset) -> Tuple[Dataset, Dataset]:
        return self._process(examples, is_train=False)

    def _get_label(self, label):
        
        for l in range(len(self._label_list)):
            if self._label_list[l] == label:
                return l
        return 0

    def _preprocess_function(self, examples):
        # Tokenize the texts
        args = (
            (examples[self._sentence1_key],) if self._sentence2_key not in examples
             else (examples[self._sentence1_key], examples[self._sentence2_key])
        )
        result = self._tokenizer(*args, padding=self._padding, max_length=self._max_seq_len, truncation=True)

        return result




    def _process(self, examples: Dataset, is_train: bool) -> Tuple[Dataset, Dataset]:
        if examples.num_rows == 0:
            raise ValueError("No examples to process")
        if not 'question' in examples.column_names:
            msg="""The file must contain a field 'question'."""
            logger.error(msg)
            raise ValueError("incorrectly formatted eval_predictions.json file")

        if is_train:
            labels = examples['label']
             # Map labels to IDs
            if self._label_list is not None:
                labels = [(self._get_label(l) if l != -1 else -1) for l in labels]
        else:
            labels = [0] * examples.num_rows

        input_features = Dataset.from_dict(
            {'example_id': examples[self._example_id_key],
            #  'language': examples[self._language_key] * examples.num_rows,  # TODO english->none?
             self._sentence1_key: examples[self._sentence1_key],
             'label': labels
            })
        if self._sentence2_key is not None:
            input_features = input_features.add_column(self._sentence2_key, examples[self._sentence2_key])

        features = input_features.map(self._preprocess_function, batched=True, load_from_cache_file=self._load_from_cache_file)
        return examples, features

