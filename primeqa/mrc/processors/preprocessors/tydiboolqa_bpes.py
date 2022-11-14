from curses.ascii import SP
import itertools
from operator import itemgetter
from operator import sub
from typing import Optional
from typing import List, Iterable, Tuple, Any, Dict, Union

from datasets.arrow_dataset import Batch
from transformers import BatchEncoding

from datasets import Dataset
from datasets.features.features import Sequence, Value

from primeqa.mrc.processors.preprocessors.tydiqa import TyDiQAPreprocessor
from primeqa.mrc.data_models.target_type import TargetType


# BPES preprocessor

class TyDiBoolQAPreprocessor(TyDiQAPreprocessor):  # TODO type signatures for all methods
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)




    def _make_passage_spans(self, examples, features_dict):
        target_type=features_dict['target_type']
        example_idx=features_dict['example_idx']
        offsets=features_dict['offset_mapping']
        if target_type == TargetType.NO or target_type == TargetType.YES:
            input_ids = features_dict['input_ids']
            example = examples[example_idx]
            target=example['target']
            passage_index=target['passage_indices'][0]
            passage_start_offset=example['passage_candidates']['start_positions'][passage_index]
            passage_end_offset=example['passage_candidates']['end_positions'][passage_index]

            token_is_inside_passage = [passage_start_offset <= tok_start and
                                        tok_end < passage_end_offset
                                        for (tok_start, tok_end) in offsets]

            # start/end in tokens is given by the first True and last True in after the query in token_is_inside_passage
            # this step converts (character-based) offsets to (token-based) token positions,
            # as well as ensuring that the token is not split across a passage boundary
            query_end_position=input_ids.index(self._tokenizer.sep_token_id)+2
            try:
                passage_start_position = token_is_inside_passage.index(True,query_end_position)
            except ValueError as x:
                passage_start_position = query_end_position
            try:
                passage_end_position = token_is_inside_passage.index(False, passage_start_position)
            except ValueError as x:
                passage_end_position = len(token_is_inside_passage)-1

#            print(f'{target_type} {passage_start_position} {start_position} {end_position} {passage_end_position}')
#            if target_type == TargetType.SPAN_ANSWER:
#                assert( passage_start_position <= start_position <= end_position <= passage_end_position)
            features_dict['start_positions'] = passage_start_position
            features_dict['end_positions'] = passage_end_position
        # always return features_dict
        return features_dict


    def process_train(self, examples: Dataset) -> Tuple[Dataset, Dataset]:
        """ 
sample comparison of features and updated_features on Tydi train
start and end positions are introduced on the two items with target_type==3 (YES)
and not on other items

In [28]: features[0:20]['start_positions']
Out[28]: [0, 0, 0, 0, 0, 0, 14, 0, 0, 291, 0, 246, 0, 189, 0, 101, 0, 0, 0, 0]
In [29]: updated_features[0:20]['start_positions']
Out[29]: [0, 0, 0, 0, 0, 0, 14, 0, 0, 291, 0, 246, 0, 189, 0, 101, 0, 191, 17, 0]
In [30]: features[0:20]['end_positions']
Out[30]: [0, 0, 0, 0, 0, 0, 26, 0, 0, 302, 0, 249, 0, 193, 0, 106, 0, 0, 0, 0]
In [31]: updated_features[0:20]['end_positions']
Out[31]: [0, 0, 0, 0, 0, 0, 26, 0, 0, 302, 0, 249, 0, 193, 0, 106, 0, 511, 278, 0]
In [32]: features[0:20]['target_type']
Out[32]: [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 2, 1, 2, 1, 0, 1, 0, 3, 3, 0]
In [33]: updated_features[0:20]['target_type']
Out[33]: [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 2, 1, 2, 1, 0, 1, 0, 3, 3, 0]
        """
        examples, features = self._process(examples, is_train=True)
        updated_features = features.map(lambda x:self._make_passage_spans(examples,x))
        return examples, updated_features
