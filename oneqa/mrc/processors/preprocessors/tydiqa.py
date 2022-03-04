from operator import itemgetter

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

    def adapt_dataset(self, dataset: Dataset, is_train: bool) -> Dataset:
        self.validate_schema(dataset, is_train)
        dataset = dataset.rename_columns(self._rename_fields)
        dataset = dataset.map(self._split_context, load_from_cache_file=self._load_from_cache_file)
        dataset = dataset.map(self._create_target, load_from_cache_file=self._load_from_cache_file)
        dataset = super().adapt_dataset(dataset, is_train)
        return dataset
    
    def _split_context(self, example):
        context_bytes = example['document_plaintext'].encode('utf-8')
        context = [
            context_bytes[start_byte: end_byte].decode('utf-8')
            for start_byte, end_byte in zip(*self._byte_itemgetter(example['passage_answer_candidates']))
        ]
        example['context'] = context
        return example

    def _create_target(self, example):
        target = example['target']
        for old_key, new_key in self._rename_target.items():
            target[new_key] = target.pop(old_key)
        # TODO text extraction by byte from document_plaintext (generative support)
        example['target'] = target
        return example
