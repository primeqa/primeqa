from primeqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from primeqa.mrc.data_models.eval_prediction_with_processing import EvalPredictionWithProcessing

from datasets import Dataset
from tqdm import tqdm

from typing import List, Dict, Any, Tuple


class SQUADPostProcessor(ExtractivePostProcessor):
    """
    Post processor for extractive QA (use with `ExtractiveQAHead`).
    """
    
    def prepare_examples_as_references(self, examples: Dataset) -> List[Dict[str, Any]]:
        references = []
        for example_idx in range(examples.num_rows):
            example = examples[example_idx]
            answers = {"text": example["answer_text"], 
                       "answer_start": example['target']['start_positions'] }
            label = {
                'id': example['example_id'],
                'answers': answers
            }
            references.append(label)
        return references
    
    def prepare_predictions_for_squad(self, examples, predictions):
        contexts = {}
        for _, example in enumerate(tqdm(examples)):
            contexts[example["example_id"]] = example["context"]
        predictions_for_metric = []
        for example_id, preds in predictions.items():
            top_pred = preds[0]
            context = contexts[example_id][0]
            prediction_text = context[top_pred['span_answer']['start_position'] : top_pred['span_answer']['end_position']]
            prediction_for_metric = {
                'id': example_id,
                'prediction_text': prediction_text
            }
            predictions_for_metric.append(prediction_for_metric)
        return predictions_for_metric
    
    def process_references_and_predictions(self, examples, features, predictions) -> EvalPredictionWithProcessing:
        references = self.prepare_examples_as_references(examples)        
        predictions = self.process(examples, features, predictions)
        predictions_for_metric = self.prepare_predictions_for_squad(examples, predictions)

        # noinspection PyTypeChecker
        return EvalPredictionWithProcessing(
            label_ids=references,
            predictions=predictions,
            processed_predictions=predictions_for_metric
        )

    