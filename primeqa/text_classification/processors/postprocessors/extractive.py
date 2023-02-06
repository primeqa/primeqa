from primeqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from datasets import Dataset
import numpy as np
from typing import Tuple
import logging

logger = logging.getLogger(__name__)

logger.setLevel(logging.DEBUG)


    
def _get_passage_answer_span( example: dict, pred: dict ) -> str:
    """ safely extract the text of the predicted passage answer
    falls back to short_answer text if the extraction fails

    Args:
        example (dict): a row of the examples dataset (from process)
        pred (dict): the predictions for output

    Returns:
        str: text of the passage answer
    """ 
    try:
        passage_index=pred["passage_index"]
        # these are chars not bytes
        # see TyDiQAPreprocessor._convert_start_and_end_positions_from_bytes_to_chars
        # where we also see the example['context'] is list of length 1
        passage_start=example['passage_candidates']['start_positions'][passage_index]
        passage_end=example['passage_candidates']['end_positions'][passage_index]
        span=example['context'][0][passage_start:passage_end]
    except Exception as x:
        logger.info('unable to extract passage text - default to span_answer_text')
        logger.info(str(pred))
        logger.info(str(x))
        span=pred["span_answer_text"]
    return span


class ExtractivePipelinePostProcessor(ExtractivePostProcessor):
    """
        PostProcessor that is just like ExtractivePostProcssor, but provides additional fields
        needed for downstream boolean pipeline classifiers:
        new fields:   language, question, passage_answer_text
    """ 
    def process(self, examples: Dataset, features: Dataset, predictions: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        predictions=super().process(examples, features, predictions)
        for example_idx in range(examples.num_rows):     
            example = examples[example_idx]
            preds = predictions[example['example_id']]
            for pred in preds:
                pred["question"] = example["question"]
                pred["language"] = example["language"]  
                pred["passage_answer_text"] = _get_passage_answer_span(example, pred)

        return predictions
