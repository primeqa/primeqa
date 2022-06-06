from oneqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from datasets import Dataset
import numpy as np
from typing import Tuple

class ExtractivePipelinePostProcessor(ExtractivePostProcessor):
    
    def process(self, examples: Dataset, features: Dataset, predictions: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        predictions=super().process(examples, features, predictions)
        for example_idx in range(examples.num_rows):     
            example = examples[example_idx]
            preds = predictions[example['example_id']]
            for pred in preds:
                pred["question"] = example["question"]
                pred["language"] = example["language"]    
        return predictions
