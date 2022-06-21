from datasets import Dataset

from primeqa.mrc.processors.preprocessors.squad import SQUADPreprocessor


class MLQAPreprocessor(SQUADPreprocessor):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def adapt_dataset(self, dataset: Dataset, is_train: bool) -> Dataset:
        dataset = super().adapt_dataset(dataset, is_train)
        # Example of the dataset config name: mlqa.zh.en
        # zh: context language; en: question language
        # the answer language is the context language
        answer_language = dataset.config_name[5:7]
        dataset = dataset.map(lambda example: {'answer_language': answer_language},
                              load_from_cache_file=self._load_from_cache_file,
                              num_proc=self._num_workers)
        return dataset
    

   
