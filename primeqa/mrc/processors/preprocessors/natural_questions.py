import functools
from operator import itemgetter
from typing import Optional, List
from transformers import BatchEncoding
from datasets import Dataset
from datasets.arrow_dataset import Example, Batch
from datasets.features.features import Sequence, Value, ClassLabel
from primeqa.mrc.processors.preprocessors.base import BasePreProcessor



class NaturalQuestionsPreProcessor(BasePreProcessor):
    """
    Preprocessor for NQ data.
    Note this preprocessor only supports `single_context_multiple_passages=True` and will
    override the value accordingly.
    """

    _feature_types = {'question': {'text': Value(dtype='string', id=None),
                                   'tokens': Sequence(feature=Value(dtype='string', id=None),
                                                       length=-1, id=None)},
                      'document': {'html': Value(dtype='string', id=None),
                                   'title': Value(dtype='string', id=None),
                                   'tokens': Sequence(feature={'is_html': Value(dtype='bool', id=None),
                                                               'token': Value(dtype='string', id=None),
                                                               "start_byte": Value("int64"),
                                                               "end_byte": Value("int64")},
                                                    length=-1, id=None),
                                   'url': Value(dtype='string', id=None)}}
    _train_feature_types = {
        'annotations': Sequence(feature={
            'id': Value(dtype='string', id=None),
             'long_answer': {'end_byte': Value(dtype='int64', id=None),
                             'end_token': Value(dtype='int64', id=None),
                             'start_byte': Value(dtype='int64', id=None),
                             'start_token': Value(dtype='int64', id=None),
                             "candidate_index": Value("int64")},
            'short_answers': Sequence(feature={'end_byte': Value(dtype='int64', id=None),
                                               'end_token': Value(dtype='int64', id=None),
                                               'start_byte': Value(dtype='int64', id=None),
                                               'start_token': Value(dtype='int64', id=None),
                                               'text': Value(dtype='string', id=None)},
                                      length=-1, id=None),
            'yes_no_answer': ClassLabel(num_classes=2, names=['NO', 'YES'], id=None)},
            length=-1, id=None),
    }
    _single_context_type = {
        'long_answer_candidates': Sequence(feature={'end_byte': Value(dtype='int64', id=None),
                                                    'end_token': Value(dtype='int64', id=None),
                                                    'start_byte': Value(dtype='int64', id=None),
                                                    'start_token': Value(dtype='int64', id=None),
                                                    'top_level': Value(dtype='bool', id=None)},
                                  length=-1, id=None),
    }
    _yes_no_answer_to_str = {-1: 'NONE', 0: 'NO', 1: 'YES'}
    _LIST_TAGS = {'<ol', '<ul', '<dl', '<li', '<dd', '<dt'}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._single_context_multiple_passages:
            self._logger.info(f"{self.__class__.__name__} only supports single context multiple passages -- enabling")
            self._single_context_multiple_passages = True


    def adapt_dataset(self, dataset: Dataset, is_train: bool, keep_html: bool=True) -> Dataset:
        """
        Process dataset examples to rename fields, create context and set answer offset.
        Args:
             dataset: Dataset to be processed.
             is_train: True for training otherwise False.
             keep_html: True if keep html token in context otherwise false.
        Returns:
             Precossed dataset.
        """
        
        self.validate_schema(dataset, is_train)
        dataset = dataset.map(
            functools.partial(self._rename_examples_create_context_and_adjust_offset, is_train=is_train, keep_html=keep_html),
            load_from_cache_file=self._load_from_cache_file,
            num_proc=self._num_workers
        )
        dataset = super().adapt_dataset(dataset, is_train)
        return dataset


    def _rename_examples_create_context_and_adjust_offset(self, example: Example, is_train: bool, keep_html: bool=True):
        """
        Rename examples to BasePreProcessor schema,
        create context from document token,
        and set the start/end positions of target and passage candidates to the new context.
        
        Args:
             example: Dataset example.
             is_train: True for training otherwise False.
             keep_html: True if keep html token in context otherwise false.
        Returns:
             Precossed example.
        """

        # rename example
        example['example_id'] = example['id']
        example['question'] = example['question']['text']
        example['language'] = 'english'
        example['document_html'] = example['document']['html']
        example['document_tokens'] = example['document']['tokens']

        passage_candidates = {}
        passage_candidates['start_positions'] = example['long_answer_candidates']['start_byte']
        passage_candidates['end_positions'] = example['long_answer_candidates']['end_byte']
        example['passage_candidates'] = passage_candidates
        del example['long_answer_candidates']

        example['target'] = self.get_annotations(example['annotations'], example['passage_candidates'])

        # create context from document tokens, and build alignment between char and token.
        context = ""
        char_to_token = []
        token_to_char = []
        num_tokens = len(example['document_tokens']['token'])

        for i in range(num_tokens):
            if not keep_html and example['document_tokens']['is_html'][i]:
                token_to_char.append(-1)
                continue
            if context:
                char_to_token.append(-1)
                context += ' '
            for j in range(len(example['document_tokens']['token'][i])):
                char_to_token.append(i)
            token_to_char.append(len(context))
            context += example['document_tokens']['token'][i]

        example['context'] = [context]
        example['context_char_to_token'] = char_to_token
        example['context_token_to_char'] = token_to_char

        # change target offsets to document token based context (only needed by training)
        if not is_train:
           return example

        for i in range(len(example['target']['passage_indices'])):
            pidx = example['target']['passage_indices'][i]
            if pidx == -1 or example['target']['start_positions'][i] == -1:
                continue

            for j in range(num_tokens):
                if example['context_token_to_char'][j] == -1:
                    continue
                if example['document_tokens']['start_byte'][j] >= example['target']['start_positions'][i]:
                    break
            if j < num_tokens:
                example['target']['start_positions'][i] = example['context_token_to_char'][j]
            else:
                raise ValueError('Start position of short answer can not be set to the token based context.')

            for j in range(num_tokens - 1, -1, -1):
                if example['context_token_to_char'][j] == -1:
                    continue
                if example['document_tokens']['end_byte'][j] <= example['target']['end_positions'][i]:
                    break
            if j >= 0:
                example['target']['end_positions'][i] = example['context_token_to_char'][j] + \
                                                        len(example['document_tokens']['token'][j])
            else:
                raise ValueError('End position of short answer can not be set to the token based context.')

        num_passages = len(example['passage_candidates']['start_positions'])
        for i in range(num_passages):
            passage_start_position = example['passage_candidates']['start_positions'][i]
            passage_end_position = example['passage_candidates']['end_positions'][i]

            for j in range(num_tokens):
                if example['context_token_to_char'][j] == -1:
                    continue
                if example['document_tokens']['start_byte'][j] >= passage_start_position:
                    break
            if j < num_tokens:
                example['passage_candidates']['start_positions'][i] = example['context_token_to_char'][j]
            else:
                raise ValueError('Start position of passage candidate can not be set to the token based context.')

            for j in range(num_tokens - 1, -1, -1):
                if example['context_token_to_char'][j] == -1:
                    continue
                if example['document_tokens']['end_byte'][j] <= passage_end_position:
                    break
            if j >= 0:
                example['passage_candidates']['end_positions'][i] = example['context_token_to_char'][j] + \
                                                                    len(example['document_tokens']['token'][j])
            else:
                raise ValueError('End position of passage candidate can not be set to the token based context.')

        return example


    def get_annotations(self, annotations, paragraphs):
        """
        Process NQ annotations into preprocessor format.
        Args:
             annotations: Annotations of NQ example.
             paragraphs: Passage_candidates of NQ example.
        Returns:
             Annotations in preprocessor format.
        """

        nq_annotations = {}
        nq_annotations['end_positions'] = []
        nq_annotations['start_positions'] = []
        nq_annotations['passage_indices'] = []
        nq_annotations['yes_no_answer'] = []
        for i in range(len(annotations['id'])):
            start_byte, end_byte = annotations['short_answers'][i]['start_byte'],annotations['short_answers'][i]['end_byte']
            
            if len(start_byte) == 0:
                start_byte = -1
                end_byte = -1
                candidate_index = -1
            else:
                start_byte = start_byte[0]
                end_byte = end_byte[0]
                candidate_index = annotations['long_answer'][i]['candidate_index']
            if end_byte < start_byte:
                self._logger.error("end_byte < start_byte")

            yes_no_answer = annotations['yes_no_answer'][i]
            yes_no_answer = self._yes_no_answer_to_str[yes_no_answer]
                
            nq_annotations['end_positions'].append(end_byte)
            nq_annotations['start_positions'].append(start_byte)
            nq_annotations['passage_indices'].append(candidate_index)
            nq_annotations['yes_no_answer'].append(yes_no_answer)
        return nq_annotations

