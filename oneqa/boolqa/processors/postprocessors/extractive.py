from oneqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from oneqa.mrc.data_models.eval_prediction_with_processing import EvalPredictionWithProcessing

from datasets import Dataset
from tqdm import tqdm

from typing import List, Dict, Any, Tuple

class ExtractivePipelinePostProcessor(ExtractivePostProcessor):
    
    def process_references_and_predictions(self, examples, features, predictions) -> EvalPredictionWithProcessing:
        references = self.prepare_examples_as_references(examples)
        predictions = self.process(examples, features, predictions)
        predictions_for_metric = []

        for example_idx in range(examples.num_rows):
            example = examples[example_idx]
            preds = predictions[example['example_id']]
            for pred in preds:
                pred["question"] = example["question"]
                pred["language"] = example["language"]

        for example_id, preds in predictions.items():
            top_pred = preds[0]
            prediction_for_metric = {
                'example_id': example_id,
                'start_position': top_pred['span_answer']['start_position'],
                'end_position': top_pred['span_answer']['end_position'],
                'passage_index': top_pred['passage_index'],
                'yes_no_answer': top_pred['yes_no_answer'],
                'confidence_score': top_pred['span_answer_score']
            }
            predictions_for_metric.append(prediction_for_metric)

        # noinspection PyTypeChecker
        return EvalPredictionWithProcessing(
            label_ids=references,
            predictions=predictions,
            processed_predictions=predictions_for_metric
        )