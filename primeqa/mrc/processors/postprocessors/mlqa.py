from primeqa.mrc.processors.postprocessors.squad import SQUADPostProcessor

from datasets import Dataset

from typing import List, Dict, Any


class MLQAPostProcessor(SQUADPostProcessor):
    """
    Post processor for extractive QA (use with `ExtractiveQAHead`).
    """
    
    def prepare_examples_as_references(self, examples: Dataset) -> List[Dict[str, Any]]:
        references = super().prepare_examples_as_references(examples)
        for example_idx in range(examples.num_rows):
            example = examples[example_idx]
            reference = references[example_idx]
            reference['answer_language'] = example['answer_language']
        return references
    
   