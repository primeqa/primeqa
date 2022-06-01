from datasets import Dataset
from datasets.arrow_dataset import Example
from datasets.features.features import Sequence, Value

from oneqa.mrc.processors.preprocessors.base import BasePreProcessor


class SQUADPreprocessor(BasePreProcessor):
    """
    Preprocessor for the SQuAD 1.1 data.
    Note this preprocessor only supports `single_context_multiple_passages=True` and will
    override the value accordingly.
    """
    _feature_types = {'question': Value(dtype='string', id=None),
                      'context': Value(dtype='string', id=None)}
    _train_feature_types = {
        'answers': Sequence(feature={
                   'text': Value(dtype='string', id=None),
                   'answer_start': Value(dtype='int32', id=None)})
    }
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not self._single_context_multiple_passages:
            self._logger.info(f"{self.__class__.__name__} only supports single context multiple passages -- enabling")
            self._single_context_multiple_passages = True

    def adapt_dataset(self, dataset: Dataset, is_train: bool) -> Dataset:
        dataset = dataset.map(self._augment_examples,
                              load_from_cache_file=self._load_from_cache_file,
                              num_proc=self._num_workers
                              )
        dataset = super().adapt_dataset(dataset, is_train)
        return dataset
    

    def _augment_examples(self, example: Example):
        """Rename examples from SQUAD schema to `BasePreProcessor` schema."""
        
        example["example_id"] = example.pop("id")
        
        target = example.pop('answers')
        target["start_positions"] = target.pop("answer_start")
        target["end_positions"] = [s + len(t) for (s,t) in zip(target["start_positions"],target["text"])]
        target.pop("text")
        target["passage_indices"] = [0 for _ in target["start_positions"]]
        target["yes_no_answer"] = ['NONE' for _ in target["start_positions"]]
        example['target'] = target
        
        passage_candidates = {"start_positions": [0],
                               "end_positions" : [len(example["context"])]}
        example['passage_candidates'] = passage_candidates
        
        context = [ example['context'] ]
        example['context'] = context  
        
        return example