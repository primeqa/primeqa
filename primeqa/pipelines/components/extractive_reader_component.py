import logging
from typing import List
import time
from dataclasses import dataclass, field

from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding
from datasets import Dataset

from primeqa.pipelines.components.base import ReaderComponent
from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from primeqa.mrc.models.task_model import ModelForDownstreamTasks
from primeqa.mrc.processors.preprocessors.base import BasePreProcessor
from primeqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from primeqa.mrc.trainers.mrc import MRCTrainer


@dataclass
class ExtractiveReaderComponent(ReaderComponent):
    """_summary_

    Args:
        model (str, optional): Model. Defaults to "PrimeQA/tydiqa-primary-task-xlm-roberta-large".
        use_fast (bool, optional): If set to "True", uses the fast version of the tokenizer. Defaults to True.
        stride (int, optional): Step size to move sliding window across context. Defaults to 128.
        max_seq_len (int, optional): Maximum length of question and context inputs to the model (in word pieces/bpes). Defaults to 512.
        n_best_size (int, optional): Maximum number of start/end logits to consider (max values). Defaults to 20.
        max_num_answers (int, optional): Maximum number of answers. Defaults to 5.
        max_answer_length (int, optional): Maximum answer length. Defaults to 32.
        scorer_type (str, optional): Scoring algorithm. Defaults to "weighted_sum_target_type_and_score_diff".
        min_score_threshold: (float, optional): Minimum score threshold. Defaults to None.
        logger (logging.Logger, optional): logger object. Defaults to logging.getLogger(ExtractiveReaderComponent).


    Returns:
        _type_: _description_
    """

    model: str = "PrimeQA/tydiqa-primary-task-xlm-roberta-large"
    use_fast: bool = True
    stride: int = 128
    max_seq_len: int = 512
    n_best_size: int = 20
    max_num_answers: int = 5
    max_answer_length: int = 32
    scorer_type: str = SupportedSpanScorers.WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF
    min_score_threshold: float = None
    logger: logging.Logger = logging.getLogger("ExtractiveReaderComponent")

    def __post_init__(self):
        self.name = "Extractive Reader"
        self.type = ReaderComponent.__name__

        # Placeholder variables
        self._preprocessor = None
        self._trainer = None

    def load(self, *args, **kwargs):
        start_t = time.time()
        task_heads = EXTRACTIVE_HEAD
        # Load configuration for model
        config = AutoConfig.from_pretrained(self.model)

        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model,
            use_fast=self.use_fast,
            config=config,
        )

        config.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        model = ModelForDownstreamTasks.from_config(
            config,
            self.model,
            task_heads=task_heads,
        )
        model.set_task_head(next(iter(task_heads)))

        # Initialize preprocessor
        self._preprocessor = BasePreProcessor(
            stride=self.stride,
            max_seq_len=self.max_seq_len,
            tokenizer=tokenizer,
        )

        data_collator = DataCollatorWithPadding(tokenizer)
        postprocessor = ExtractivePostProcessor(
            k=self.max_num_answers,
            n_best_size=self.n_best_size,
            max_answer_length=self.max_answer_length,
            scorer_type=self.scorer_type,
            single_context_multiple_passages=self._preprocessor._single_context_multiple_passages,
        )

        self._trainer = MRCTrainer(
            model=model,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=postprocessor.process,
        )

        self.logger.info(
            "%s pipeline - loading took %s seconds",
            self.name,
            time.time() - start_t,
        )

    def apply(self, input_texts: List[str], context: List[List[str]], *args, **kwargs):
        eval_examples = Dataset.from_dict(
            dict(
                question=input_texts,
                context=context,
                example_id=[str(idx) for idx in range(len(input_texts))],
            )
        )

        eval_examples, eval_dataset = self._preprocessor.process_eval(eval_examples)

        # Run predict
        predictions = [[] for _ in range(len(input_texts))]
        for passage_idx, raw_predictions in self._trainer.predict(
            eval_dataset=eval_dataset, eval_examples=eval_examples
        ).items():
            for raw_prediction in raw_predictions:
                predictions[int(passage_idx)].append(raw_prediction)

        # If min_score_threshold is provide, use it to filter out predictions
        if "min_score_threshold" in kwargs:
            filtered_predictions = []
            for sorted_predictions_for_passage in predictions:
                filtered_predictions_for_passage = []
                for sorted_prediction in sorted_predictions_for_passage:
                    if (
                        sorted_prediction["confidence_score"]
                        >= kwargs["min_score_threshold"]
                    ):
                        filtered_predictions_for_passage.append(sorted_prediction)

                filtered_predictions.append(filtered_predictions_for_passage)
            return filtered_predictions
        else:
            return predictions
