import inspect
import json
import logging
import os
from typing import Optional

import datasets
import torch
from datasets import Dataset
from packaging import version
from torch.utils.data import DataLoader
from transformers import Trainer, is_datasets_available
from transformers.trainer_pt_utils import IterableDatasetShard

logger = logging.getLogger(__name__)


class NWayTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, eval_dataset=None, post_process_function=None, **kwargs):
        """
        NWay training and evaluation.

        Args:
            *args: Arguments for super-class constructor.
            eval_examples: Eval examples `Dataset` from `BasePreprocessor.process_eval`.
            eval_dataset: Eval features `Dataset` from `BasePreprocessor.process_eval`.
            post_process_function:  Function to create predictions from model outputs.
            **kwargs: Keyword arguments for super-class constructor.
        """
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.eval_dataset = eval_dataset
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        """
        Evaluate model using either eval data passed to method (if given).
        Otherwise use data given to constructor at instantiation.

        Args:
            eval_examples: Eval examples `Dataset` from `BasePreprocessor.process_eval`.
            eval_dataset: Eval features `Dataset` from `BasePreprocessor.process_eval`.
            ignore_keys: Keys to ignore in evaluation loop.
            metric_key_prefix: Append this prefix to metric names.

        Returns:
            Evaluation metrics if post-processing and metric computation functions
            were provided to constructor at instantiation, otherwise an empty dict.
        """
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                # gather predictions if running in eval mode
                prediction_loss_only=self.args.prediction_loss_only, #True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)

            # TODO: return eval_preds and metrics, write save function for preds
            with open(os.path.join(self.args.output_dir, 'eval_predictions.json'), 'w') as f:
                json.dump(eval_preds.predictions, f, indent=4)
            with open(os.path.join(self.args.output_dir, 'eval_predictions_processed.json'), 'w') as f:
                json.dump(eval_preds.processed_predictions, f, indent=4)
            with open(os.path.join(self.args.output_dir, 'eval_references.json'), 'w') as f:
                json.dump(eval_preds.label_ids, f, indent=4)

        if self.post_process_function is not None and self.compute_metrics is not None:
            metrics = self.compute_metrics(eval_preds)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics
    
    def predict(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        """
        Obtain the predictions using either eval data passed to method (if given).
        Otherwise use data given to constructor at instantiation.

        Args:
            eval_examples: Eval examples `Dataset` from `BasePreprocessor.process_eval`.
            eval_dataset: Eval features `Dataset` from `BasePreprocessor.process_eval`.
            ignore_keys: Keys to ignore in evaluation loop.

        Returns:
            Answer predictions if post-processing function was provided to constructor 
            at instantiation, otherwise an empty dict.
        """
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                # gather predictions if running in eval mode
                prediction_loss_only=self.args.prediction_loss_only, #True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            pass

        if self.post_process_function is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output.predictions)
        else:
            eval_preds = {}

        return eval_preds
