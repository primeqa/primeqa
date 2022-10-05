import logging
from typing import Union, List
import time
from operator import itemgetter

from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset

from primeqa.pipelines.base import ReaderPipeline

from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from primeqa.mrc.models.task_model import ModelForDownstreamTasks
from primeqa.mrc.processors.preprocessors.base import BasePreProcessor
from primeqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from primeqa.mrc.trainers.mrc import MRCTrainer


class ExtractiveReader(ReaderPipeline):
    """
    Reader: Extractive
    """

    def __init__(
        self,
        logger: Union[logging.Logger, None] = None,
    ) -> None:
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger

        # Default class variables
        self.pipeline_id = self.__class__.__name__
        self.pipeline_name = "Extractive Reader"
        self.pipeline_description = ""
        self.pipeline_type = ReaderPipeline.__name__
        self.parameters = {
            "model": {
                "parameter_id": "model",
                "name": "Model",
                "type": "String",
                "value": "PrimeQA/tydiqa-primary-task-xlm-roberta-large",
                "options": ["PrimeQA/tydiqa-primary-task-xlm-roberta-large"],
            },
            "use_fast": {
                "parameter_id": "use_fast",
                "name": "Use the fast version of the tokenizer",
                "type": "Boolean",
                "value": True,
                "options": [True, False],
            },
            "stride": {
                "parameter_id": "stride",
                "name": "Step size to move sliding window across context",
                "type": "Numeric",
                "value": 128,
                "range": [8, 256, 8],
            },
            "max_seq_len": {
                "parameter_id": "max_seq_len",
                "name": "Maximum length of question and context inputs to the model (in word pieces/bpes)",
                "type": "Numeric",
                "value": 512,
                "range": [32, 512, 8],
            },
            "max_num_answers": {
                "parameter_id": "max_num_answers",
                "name": "Max number of answers",
                "type": "Numeric",
                "value": 3,
                "range": [1, 5, 1],
            },
            "n_best_size": {
                "parameter_id": "n_best_size",
                "name": "Max number of start/end logits to consider (max values)",
                "type": "Numeric",
                "value": 20,
                "range": [1, 50, 1],
            },
            "max_answer_length": {
                "parameter_id": "max_answer_length",
                "name": "Maximum answer length",
                "type": "Numeric",
                "value": 32,
                "range": [2, 128, 2],
            },
            "scorer_type": {
                "parameter_id": "scorer_type",
                "name": "Scoring algorithm",
                "type": "String",
                "value": "weighted_sum_target_type_and_score_diff",
                "options": ["weighted_sum_target_type_and_score_diff"],
            },
            "min_score_threshold": {
                "parameter_id": "min_score_threshold",
                "name": "Minimum score threshold",
                "type": "Numeric",
                "value": 0.0,
                "range": [-10.0, 10.0, 0.01],
            },
        }

        # Placeholder class variables
        self.preprocessor = None
        self.trainer = None

    def load(self, *args, **kwargs):
        start_t = time.time()
        task_heads = EXTRACTIVE_HEAD
        # Load configuration for model
        config = AutoConfig.from_pretrained(self.parameters["model"]["value"])

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.parameters["model"]["value"],
            use_fast=True,
            config=config,
        )

        config.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        model = ModelForDownstreamTasks.from_config(
            config,
            self.parameters["model"]["value"],
            task_heads=task_heads,
        )
        model.set_task_head(next(iter(task_heads)))

        # Initialize preprocessor
        self.preprocessor = BasePreProcessor(
            stride=self.parameters["stride"]["value"],
            max_seq_len=self.parameters["max_seq_len"]["value"],
            tokenizer=tokenizer,
        )

        data_collator = DataCollatorWithPadding(tokenizer)
        postprocessor = ExtractivePostProcessor(
            k=self.parameters["max_num_answers"]["value"],
            n_best_size=self.parameters["n_best_size"]["value"],
            max_answer_length=self.parameters["max_answer_length"]["value"],
            scorer_type=SupportedSpanScorers.WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF,
            single_context_multiple_passages=self.preprocessor._single_context_multiple_passages,
        )

        self.trainer = MRCTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=postprocessor.process,
        )

        self._logger.info(
            "%s pipeline - loading took %s seconds",
            self.pipeline_name,
            time.time() - start_t,
        )

    def get_parameters(self):
        return [self.parameters.values()]

    def set_parameter(self, parameter):
        self.parameters[parameter["parameter_id"]] = parameter

    def get_parameter(self, parameter_id: str):
        return self.parameters[parameter_id]

    def get_parameter_type(self, parameter_id: str):
        return self.parameters[parameter_id]["type"]

    def set_parameter_value(self, parameter_id: str, parameter_value: int):
        self.parameters[parameter_id]["value"] = parameter_value

    def get_parameter_value(self, parameter_id: str):
        return self.parameters[parameter_id]["value"]

    def serialize(self):
        return {
            "pipeline_id": self.pipeline_id,
            "parameters": {
                parameter["parameter_id"]: parameter["value"]
                for parameter in self.parameters.values()
            },
        }

    def apply(self, input_texts: List[str], context: List[List[str]], *args, **kwargs):
        eval_examples = Dataset.from_dict(
            dict(
                question=input_texts,
                context=context,
                example_id=[str(idx) for idx in range(len(input_texts))],
            )
        )

        eval_examples, eval_dataset = self.preprocessor.process_eval(eval_examples)

        # Run predict
        predictions = [[] for _ in range(len(input_texts))]
        for passage_idx, raw_predictions in self.trainer.predict(
            eval_dataset=eval_dataset, eval_examples=eval_examples
        ).items():
            for raw_prediction in raw_predictions:
                predictions[int(passage_idx)].append(raw_prediction)

        sorted_predictions = [
            sorted(entry, key=itemgetter("confidence_score"), reverse=True)
            for entry in predictions
        ]

        filtered_predictions = []
        for sorted_predictions_for_passage in sorted_predictions:
            filtered_predictions_for_passage = []
            for sorted_prediction in sorted_predictions_for_passage:
                if sorted_prediction["confidence_score"] >= self.get_parameter_value(
                    "min_score_threshold"
                ):
                    filtered_predictions_for_passage.append(sorted_prediction)

            filtered_predictions.append(filtered_predictions_for_passage)
        return filtered_predictions
