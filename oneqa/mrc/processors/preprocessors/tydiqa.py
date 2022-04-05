from operator import itemgetter
from typing import Optional

from datasets import Dataset
from datasets.features.features import Sequence, Value

from oneqa.mrc.processors.preprocessors.default import DefaultPreProcessor


class TyDiQAPreprocessor(DefaultPreProcessor):  # TODO type signatures for all methods
    _feature_types = {'question_text': Value(dtype='string', id=None),
                      'document_plaintext': Value(dtype='string', id=None)}
    _train_feature_types = {
        'annotations': Sequence(feature={
                   'minimal_answers_end_byte': Value(dtype='int32', id=None),
                   'minimal_answers_start_byte': Value(dtype='int32', id=None),
                   'passage_answer_candidate_index': Value(dtype='int32', id=None),
                   'yes_no_answer': Value(dtype='string', id=None)})
    }
    _byte_itemgetter = itemgetter('plaintext_start_byte', 'plaintext_end_byte')
    _rename_fields = {'question_text': 'question', 'annotations': 'target'}
    _rename_target = {'passage_answer_candidate_index': 'passage_indices',
                      'minimal_answers_start_byte': 'start_positions',
                      'minimal_answers_end_byte': 'end_positions'}

    def __init__(self, *args, max_contexts: Optional[int] = 48, **kwargs):
        super().__init__(*args, **kwargs)
        self._max_contexts = max_contexts
        self._single_context_multiple_passages = True

    def adapt_dataset(self, dataset: Dataset, is_train: bool) -> Dataset:
        self.validate_schema(dataset, is_train)
        dataset = dataset.rename_columns(self._rename_fields)
        dataset = dataset.map(self._create_target,
                              load_from_cache_file=self._load_from_cache_file,
                              num_proc=self._num_workers
                              )
        dataset = dataset.map(self._split_context,
                              load_from_cache_file=self._load_from_cache_file,
                              num_proc=self._num_workers
        )
        dataset = super().adapt_dataset(dataset, is_train)
        return dataset
    
    def _split_context(self, example):
        context = example['document_plaintext']
        context_bytes = context.encode('utf-8')

        for i in range(len(example['target']['passage_indices'])):
            pidx = example['target']['passage_indices'][i]
            if pidx == -1 or example['target']['start_positions'][i] == -1:
                continue

            # offset = example['passage_answer_candidates']['plaintext_start_byte'][pidx]
            # example['target']['start_positions'][i] -= offset
            # example['target']['end_positions'][i] -= offset

            example['target']['start_positions'][i] = len(context_bytes[:example['target']['start_positions'][i]].decode('utf-8', errors='replace'))
            example['target']['end_positions'][i] = len(context_bytes[:example['target']['end_positions'][i]].decode('utf-8', errors='replace'))

        for i in range(len(example['passage_answer_candidates']['plaintext_start_byte'])):
            example['passage_answer_candidates']['plaintext_start_byte'][i] = len(context_bytes[:example['passage_answer_candidates']['plaintext_start_byte'][i]].decode('utf-8', errors='replace'))
            example['passage_answer_candidates']['plaintext_end_byte'][i] = len(context_bytes[:example['passage_answer_candidates']['plaintext_end_byte'][i]].decode('utf-8', errors='replace'))

        # if any(x < -1 for x in example['target']['start_positions']) or any(x < -1 for x in example['target']['end_positions']) or any(x < -1 for x in example['target']['passage_indices']):
        #     raise ValueError(f"Error processing example: {example}")

        if self._max_contexts and len(example['passage_answer_candidates']) > self._max_contexts:
            context_bytes = context_bytes[:example['passage_answer_candidates'][self._max_contexts - 1]['plaintext_end_byte']]
            context = context_bytes.decode('utf-8')
        example['context'] = [context]
        return example

    def _create_target(self, example):
        target = example['target']
        for old_key, new_key in self._rename_target.items():
            target[new_key] = target.pop(old_key)
        # TODO text extraction by byte from document_plaintext (generative support)
        example['target'] = target
        return example
