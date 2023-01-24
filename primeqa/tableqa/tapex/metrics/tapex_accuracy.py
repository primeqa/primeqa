import ast
import logging
import numpy as np
from collections import defaultdict
from typing import List
logger = logging.getLogger(__name__)


class TapexAccuracy:
    def __init__(self,tokenizer,data_args):
        self.tokenizer = tokenizer
        self.data_args=data_args

    def postprocess_text(self,preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        return preds, labels

    def compute_metrics(self,eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        if self.data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        delimiter = ", "

        # define example evaluation
        def evaluate_example(predict_str: str, ground_str: str):
            predict_spans = predict_str.split(delimiter)
            ground_spans = ground_str.split(delimiter)
            predict_values = defaultdict(lambda: 0)
            ground_values = defaultdict(lambda: 0)
            for span in predict_spans:
                try:
                    predict_values[float(span)] += 1
                except ValueError:
                    predict_values[span.strip()] += 1
            for span in ground_spans:
                try:
                    ground_values[float(span)] += 1
                except ValueError:
                    ground_values[span.strip()] += 1
            _is_correct = predict_values == ground_values
            return _is_correct

        def get_denotation_accuracy(predictions: List[str], references: List[str]):
            assert len(predictions) == len(references)
            correct_num = 0
            for predict_str, ground_str in zip(predictions, references):
                is_correct = evaluate_example(predict_str.lower(), ground_str.lower())
                if is_correct:
                    correct_num += 1
            return correct_num / len(predictions)

        accuracy = get_denotation_accuracy(decoded_preds, decoded_labels)
        result = {"denotation_accuracy": accuracy}

        return result