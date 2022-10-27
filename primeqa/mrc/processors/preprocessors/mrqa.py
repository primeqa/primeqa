from datasets import Dataset
from datasets.arrow_dataset import Example
from datasets.features.features import Sequence, Value

from primeqa.mrc.processors.preprocessors.base import BasePreProcessor


class MRQAPreprocessor(BasePreProcessor):
    """
    Preprocessor for the MRQA.
    Note this preprocessor only supports `single_context_multiple_passages=True` and will
    override the value accordingly.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._single_context_multiple_passages:
            self._logger.info(f"{self.__class__.__name__} only supports single context multiple passages -- enabling")
            self._single_context_multiple_passages = True

    def _rename_examples(self, example: Example):
        """Rename examples from MRQA schema to `BasePreProcessor` schema."""
        
        example["example_id"] = example['id'] if 'id' in example else example['qid']

        target = example.pop('detected_answers')
        char_spans = target.pop('char_spans')
        target.pop('token_spans')

        example.pop('qid')
        example.pop('subset')
        example.pop('context_tokens')
        example.pop('question_tokens')
        example.pop('answers')
        
        target["start_positions"] = [ l["start"][0] for l in char_spans]
        target["end_positions"] = [ l["end"][0] for l in char_spans]
        target["passage_indices"] = [0 for _ in target["start_positions"]]
        target["yes_no_answer"] = ['NONE' for _ in target["start_positions"]]
        example['target'] = target
        
        # may not be needed
        example["answer_text"] = target.pop("text")
        
        passage_candidates = {"start_positions": [0],
                               "end_positions" : [len(example["context"])]}
        example['passage_candidates'] = passage_candidates

        # context is a list 
        context = [ example['context'] ]
        example['context'] = context        
        return example

    def adapt_dataset(self, dataset: Dataset, is_train: bool) -> Dataset:
        dataset = dataset.map(self._rename_examples,
                              load_from_cache_file=self._load_from_cache_file,
                              num_proc=self._num_workers
                              )
        dataset = super().adapt_dataset(dataset, is_train)
        return dataset
        
