from typing import List, Dict
from dataclasses import dataclass, field
import json

from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset

from primeqa.components.base import Reader as BaseReader
from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from primeqa.mrc.models.task_model import ModelForDownstreamTasks
from primeqa.mrc.processors.preprocessors.base import BasePreProcessor
from primeqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from primeqa.mrc.trainers.mrc import MRCTrainer


@dataclass
class ExtractiveReader(BaseReader):
    """_summary_

    Args:
        model (str, optional): Model. Defaults to "PrimeQA/nq_tydi_sq1-reader-xlmr_large-20221110".
        use_fast (bool, optional): If set to "True", uses the fast version of the tokenizer. Defaults to True.
        stride (int, optional): Step size to move sliding window across context. Defaults to 128.
        max_seq_len (int, optional): Maximum length of question and context inputs to the model (in word pieces/bpes). Defaults to 512.
        n_best_size (int, optional): Maximum number of start/end logits to consider (max values). Defaults to 20.
        max_num_answers (int, optional): Maximum number of answers. Defaults to 5.
        max_answer_length (int, optional): Maximum answer length. Defaults to 32.
        scorer_type (str, optional): Scoring algorithm. Defaults to "weighted_sum_target_type_and_score_diff".
        min_score_threshold: (float, optional): Minimum score threshold. Defaults to None.

    Important:
        1. Each field has metadata property which can carry additional information for other downstream usages.
        2. Two special keys (api_support and exclude_from_hash) are defined in "metadata" property.
            a. api_support (bool, optional): If set to True, that parameter is exposed via service layer. Defaults to False.
            b. exclude_from_hash (bool,optional): If set to True, that parameter is not considered while building the hash representation for the object. Defaults to False.


    Returns:
        _type_: _description_
    """

    model: str = field(
        default="PrimeQA/nq_tydi_sq1-reader-xlmr_large-20221110",
        metadata={"name": "Model", "api_support": True},
    )
    use_fast: bool = field(
        default=True,
        metadata={
            "name": "Use the fast version of the tokenizer",
            "options": [True, False],
        },
    )
    stride: int = field(
        default=128,
        metadata={
            "name": "Stride",
            "description": "Step size to move sliding window across context",
            "range": [8, 256, 8],
        },
    )
    max_seq_len: int = field(
        default=512,
        metadata={
            "name": "Maximum sequence length",
            "description": "Maximum length of question and context inputs to the model (in word pieces/bpes)",
            "range": [32, 512, 8],
        },
    )
    n_best_size: int = field(
        default=20,
        metadata={
            "name": "N",
            "description": "Maximum number of start/end logits to consider (max values)",
            "range": [1, 50, 1],
        },
    )
    max_num_answers: int = field(
        default=3,
        metadata={
            "name": "Maximum number of answers",
            "range": [1, 5, 1],
            "api_support": True,
            "exclude_from_hash": True,
        },
    )
    max_answer_length: int = field(
        default=1000,
        metadata={
            "name": "Maximum answer length",
            "range": [2, 2000, 2],
            "api_support": True,
            "exclude_from_hash": True,
        },
    )
    scorer_type: str = field(
        default=SupportedSpanScorers.WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF.value,
        metadata={
            "name": "Scoring algorithm",
            "options": [
                SupportedSpanScorers.SCORE_DIFF_BASED.value,
                SupportedSpanScorers.TARGET_TYPE_WEIGHTED_SCORE_DIFF.value,
                SupportedSpanScorers.WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF.value,
            ],
        },
    )
    min_score_threshold: float = field(
        default=None,
        metadata={
            "name": "Minimum score threshold",
            "api_support": False,
            "exclude_from_hash": True,
        },
    )

    def __post_init__(self):
        # Placeholder variables
        self._loaded_model = None
        self._tokenizer = None
        self._preprocessor = None
        self._scorer_type_as_enum = None
        self._data_collector = None

    def __hash__(self) -> int:
        # Step 1: Identify all fields to be included in the hash
        hashable_fields = [
            k
            for k, v in self.__class__.__dataclass_fields__.items()
            if not "exclude_from_hash" in v.metadata
            or not v.metadata["exclude_from_hash"]
        ]

        # Step 2: Run
        return hash(
            f"{self.__class__.__name__}::{json.dumps({k: v for k, v in vars(self).items() if k in hashable_fields }, sort_keys=True)}"
        )

    def load(self, *args, **kwargs):
        task_heads = EXTRACTIVE_HEAD
        # Load configuration for model
        config = AutoConfig.from_pretrained(self.model)

        # Initialize tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            use_fast=self.use_fast,
            config=config,
        )

        config.sep_token_id = self._tokenizer.convert_tokens_to_ids(
            self._tokenizer.sep_token
        )
        self._loaded_model = ModelForDownstreamTasks.from_config(
            config,
            self.model,
            task_heads=task_heads,
        )
        self._loaded_model.set_task_head(next(iter(task_heads)))

        # Initialize preprocessor
        self._preprocessor = BasePreProcessor(
            stride=self.stride,
            max_seq_len=self.max_seq_len,
            tokenizer=self._tokenizer,
            single_context_multiple_passages=False,
        )

        # Configure scorer
        if self.scorer_type == SupportedSpanScorers.SCORE_DIFF_BASED.value:
            self._scorer_type_as_enum = SupportedSpanScorers.SCORE_DIFF_BASED
        elif (
            self.scorer_type
            == SupportedSpanScorers.TARGET_TYPE_WEIGHTED_SCORE_DIFF.value
        ):
            self._scorer_type_as_enum = (
                SupportedSpanScorers.TARGET_TYPE_WEIGHTED_SCORE_DIFF
            )
        elif (
            self.scorer_type
            == SupportedSpanScorers.WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF.value
        ):
            self._scorer_type_as_enum = (
                SupportedSpanScorers.WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF
            )
        else:
            raise ValueError(f"Unsupported scorer type: {self.scorer_type}")

        # Configure data collector
        self._data_collector = DataCollatorWithPadding(self._tokenizer)

    def predict(
        self,
        questions: List[str],
        contexts: List[List[str]],
        *args,
        example_ids: List[str] = None,
        **kwargs,
    ) -> Dict[str, List[Dict]]:
        # Step 1: Locally update object variable values, if provided
        max_num_answers = (
            kwargs["max_num_answers"]
            if "max_num_answers" in kwargs
            else self.max_num_answers
        )

        max_answer_length = (
            kwargs["max_answer_length"]
            if "max_answer_length" in kwargs
            else self.max_answer_length
        )

        min_score_threshold = (
            kwargs["min_score_threshold"]
            if "min_score_threshold" in kwargs
            else self.min_score_threshold
        )

        # Step 2: Initialize post processor
        postprocessor = ExtractivePostProcessor(
            k=max_num_answers,
            n_best_size=self.n_best_size,
            max_answer_length=max_answer_length,
            scorer_type=self._scorer_type_as_enum,
        )

        # Step 3: Load trainer
        trainer = MRCTrainer(
            model=self._loaded_model,
            tokenizer=self._tokenizer,
            data_collator=self._data_collector,
            post_process_function=postprocessor.process,
        )

        # Step 4: Prepare dataset from input texts and contexts
        assert len(questions) == len(contexts)

        if example_ids is None:
            example_ids = [str(idx) for idx in range(len(questions))]

        assert len(example_ids) == len(questions)

        examples_dict = dict(
            question=questions, context=contexts, example_id=example_ids
        )

        eval_examples, eval_dataset = self._preprocessor.process_eval(
            Dataset.from_dict(examples_dict)
        )

        # Step 5: Run predict
        predictions = {}
        for example_id, raw_predictions in trainer.predict(
            eval_dataset=eval_dataset, eval_examples=eval_examples
        ).items():
            predictions[example_id] = []
            for raw_prediction in raw_predictions:
                if (
                    min_score_threshold
                    and raw_prediction["confidence_score"] < min_score_threshold
                ):
                    continue
                processed_prediction = {}
                processed_prediction["example_id"] = raw_prediction["example_id"]
                processed_prediction["passage_index"] = raw_prediction["passage_index"]
                processed_prediction["span_answer_text"] = raw_prediction[
                    "span_answer_text"
                ]
                processed_prediction["span_answer"] = raw_prediction["span_answer"]
                processed_prediction["span_answer_score"] = raw_prediction[
                    "span_answer_score"
                ]
                processed_prediction["confidence_score"] = raw_prediction[
                    "confidence_score"
                ]

                predictions[example_id].append(processed_prediction)

        return predictions

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass
