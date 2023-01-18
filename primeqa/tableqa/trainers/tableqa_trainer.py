from transformers import Trainer
import os
import json
class TableQATrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        """Basic tableQA trainer which extends huggingface's transformers Trainer class

        Args:
            eval_examples (_type_, optional): _description_. Defaults to None.
            post_process_function (_type_, optional): _description_. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function
        


    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                #prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics     
        if self.post_process_function is not None:
            eval_preds,gold_answers = self.post_process_function(eval_examples, eval_dataset, output.predictions[1])
            with open(os.path.join(self.args.output_dir, 'eval_predictions.json'), 'w') as f:
                json.dump(eval_preds, f, indent=4)
        if self.compute_metrics is not None:
            metrics = self.compute_metrics(eval_preds,gold_answers)

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

      

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        return metrics