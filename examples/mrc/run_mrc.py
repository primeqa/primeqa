import logging
import os
import sys
import traceback
import glob
from dataclasses import dataclass, field
from importlib import import_module
from operator import attrgetter
from typing import Optional, Type

import datasets
from transformers import HfArgumentParser, TrainingArguments, DataCollatorWithPadding, AutoConfig, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint, set_seed

from oneqa.mrc.data_models.eval_prediction_with_processing import EvalPredictionWithProcessing
from oneqa.mrc.metrics.tydi_f1.tydi_f1 import TyDiF1
from oneqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from oneqa.mrc.models.task_model import ModelForDownstreamTasks
from oneqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from oneqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from oneqa.mrc.processors.preprocessors.tydiqa import TyDiQAPreprocessor
from oneqa.mrc.processors.preprocessors.squad import SQUADPreprocessor
from oneqa.mrc.processors.postprocessors.squad import SQUADPostProcessor
from oneqa.mrc.trainers.mrc import MRCTrainer


def object_reference(reference_as_str: str) -> object:
    """
    Given a fully qualified path to a class reference, return a pointer to the reference.
    This will work with types, functions, methods, and other objects (e.g. dict).

    Args:
        reference_as_str: the fully qualified path (expects the fully qualified path in dot notation,
                          e.g. oneqa.mrc.processors.postprocessors.extractive.ExtractivePostProcessor).

    Returns:
        reference to path given by input

    Raises:
        TypeError: Unable to resolve input path
    """
    def _split_into_class_and_module_name(class_path):
        modules = class_path.split('.')
        if len(modules) > 1:
            return ".".join(modules[:-1]), modules[-1]
        else:
            return class_path, None

    try:
        module_name, object_name = _split_into_class_and_module_name(reference_as_str)
        module_reference = import_module(module_name)
        if object_name is None:
            return module_reference
        else:
            return getattr(module_reference, object_name)
    except Exception as ex:
        traceback.print_exc()  # Shows additional traceback for why imports fail
        raise TypeError(f"Unable to resolve the string {reference_as_str} to a fully qualified class path") from ex


# modified from
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )


# modified from
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name: str = field(
        default="tydiqa", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "local file(s) to train on."}
    )
    eval_file: Optional[str] = field(
        default=None, metadata={"help": "local file(s) to test on."}
    )
    data_file_format: str = field(
        default="json", metadata={"help": "the format of the local dataset files (json, csv, text, pandas)"}
    )
    dataset_config_name: str = field(
        default="primary_task", metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        }
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_q_char_len: int = field(
        default=128, metadata={"help": "Max length per question in characters"}
    )
    single_context_multiple_passages: bool = field(
        default=False, metadata={
            "help": "Allow multiple passages in the same input feature. "
                    "For an example with question q and context c_{1..n} setting this to True"
                    "will allow q|c_{i}c_{i+1}; whereas setting this to False enforces q|c_{i} q|c_{i+1}. "
                    "Note that not all datasets/preprocessors support both values of this parameter. "
                    "Some preprocessors may override this value."
            },
    )
    max_contexts: Optional[int] = field(
        default=None, metadata={"help": "Max contexts per consider"}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
                    "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                    "value if set."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    n_best_logits: int = field(
        default=20,
        metadata={"help": "The number of logits to consider when searching for start and end position of an answer"}
    )
    max_answer_length: int = field(
        default=32,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        },
    )
    negative_sampling_prob_when_has_answer: float = field(
        default=0.01,
        metadata={
            "help": "Only used when preparing training features, not for decoding. "
                    "For an example with question q and context c_{1..n} where ∃ answer a ∈ c"
                    "an input feature span q|c_{i} where a ∉ c_{i} will be kept with this probability."
                    "Otherwise it will be discarded."
        },
    )
    negative_sampling_prob_when_no_answer: float = field(
        default=0.04,
        metadata={
            "help": "Only used when preparing training features, not for decoding. "
                    "For an example with question q and context c_{1..n} where ∄ answer a ∈ c"
                    "an input feature span q|c_{i} will be kept with this probability."
                    "Otherwise it will be discarded."
        },
    )


@dataclass
class TaskArguments:
    """
    Task specific arguments.
    """
    scorer_type: str = field(
        default='weighted_sum_target_type_and_score_diff',
        metadata={"help": "The name of the scorer to compute answer score.",
                  "choices": SupportedSpanScorers.get_supported()
                  }
    )
    task_heads: object_reference = field(
        default=None,
        metadata={"help": "The name of the task head to use.",
                  "choices": [EXTRACTIVE_HEAD]
                  }
    )
    preprocessor: object_reference = field(
        default=TyDiQAPreprocessor,
        metadata={"help": "The name of the preprocessor to use.",
                  "choices": [TyDiQAPreprocessor,SQUADPreprocessor]
                  }
    )
    postprocessor: object_reference = field(
        default=ExtractivePostProcessor,
        metadata={"help": "The name of the postprocessor to use.",
                  "choices": [ExtractivePostProcessor, SQUADPostProcessor]
                  }
    )
    eval_metrics: str = field(
        default="TyDiF1",
        metadata={"help": "The name of the evaluation metric function implemented in oneqa (e.g. TyDiF1)," 
                          "or the name of a metric as defined in datasets.list_metrics() (e.g. squad)",
                  "choices": ["TyDiF1","squad"]
                 }
    )
    passage_non_null_threshold: int = field(
        default=2,
        metadata={"help": "The passage level non-null threshold (number of annotators to indicate no answer). This should be set to 1 if there is only one annotation"}
    )
    minimal_non_null_threshold: int = field(
        default=2,
        metadata={"help": "The minimal level non-null threshold (number of annotators to indicate no answer). This should be set to 1 if there is only one annotation"}
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Prints evaluation output if true"}
    )

    def __post_init__(self):
        if not self.task_heads:
            self.task_heads = EXTRACTIVE_HEAD  # cannot directly set mutable value as default


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, TaskArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, task_args = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, task_args = parser.parse_args_into_dataclasses()

    logger = logging.getLogger(__name__)
    scorer_type = task_args.scorer_type
    set_seed(training_args.seed)

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    task_heads = task_args.task_heads
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        config=config,
    )
    model = ModelForDownstreamTasks.from_config(
        config,
        model_args.model_name_or_path,
        task_heads=task_heads,
        cache_dir=model_args.cache_dir,
    )
    model.set_task_head(next(iter(task_heads)))

    # load data
    logger.info('Loading dataset')
    if data_args.train_file is not None or data_args.eval_file is not None:
        data_files = {}

        if data_args.train_file is not None: 
            train_files = glob.glob(data_args.train_file)
            data_files['train'] = train_files 
        if data_args.eval_file is not None: 
            eval_files = glob.glob(data_args.eval_file)
            data_files['validation'] = eval_files

        raw_datasets = datasets.load_dataset(data_args.data_file_format, 
            data_files=data_files,
            cache_dir=model_args.cache_dir)
    else:
        raw_datasets = datasets.load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
        )

    # load preprocessor
    preprocessor_class = task_args.preprocessor
    preprocessor = preprocessor_class(
        stride=data_args.doc_stride,
        tokenizer=tokenizer,
        negative_sampling_prob_when_has_answer=data_args.negative_sampling_prob_when_has_answer,
        negative_sampling_prob_when_no_answer=data_args.negative_sampling_prob_when_no_answer,
        load_from_cache_file=not data_args.overwrite_cache,
        max_seq_len=data_args.max_seq_length,
        num_workers=data_args.preprocessing_num_workers,
        max_q_char_len=data_args.max_q_char_len,
        single_context_multiple_passages=data_args.single_context_multiple_passages,
        max_contexts=data_args.max_contexts,
    )

    # process train data
    if training_args.do_train:
        train_dataset = raw_datasets["train"]
        max_train_samples = data_args.max_train_samples
        if max_train_samples is not None:
            # We will select sample from whole data if argument is specified
            train_dataset = train_dataset.select(range(max_train_samples))
        # Train Feature Creation
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            _, train_dataset = preprocessor.process_train(train_dataset)

    # process val data
    if training_args.do_eval:
        eval_examples = raw_datasets["validation"]
        max_eval_samples = data_args.max_eval_samples
        if max_eval_samples is not None:
            # We will select sample from whole data if argument is specified
            eval_examples = eval_examples.select(range(max_eval_samples))
        # Validation Feature Creation
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_examples, eval_dataset = preprocessor.process_eval(eval_examples)

    # If using mixed precision we pad for efficient hardware acceleration
    using_mixed_precision = any(attrgetter('fp16', 'bf16')(training_args))
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if using_mixed_precision else None)

    postprocessor_class = task_args.postprocessor

    # noinspection PyProtectedMember
    postprocessor = postprocessor_class(
        k=data_args.n_best_logits,
        n_best_size=data_args.n_best_size,
        max_answer_length=data_args.max_answer_length,
        scorer_type=SupportedSpanScorers(scorer_type),
        single_context_multiple_passages=preprocessor._single_context_multiple_passages,
    )

    if task_args.eval_metrics in datasets.list_metrics():
        eval_metrics = datasets.load_metric(task_args.eval_metrics)
    else:
        eval_metrics = getattr(sys.modules[__name__], task_args.eval_metrics)()

    def compute_metrics(p: EvalPredictionWithProcessing):
        return eval_metrics.compute(predictions=p.processed_predictions, references=p.label_ids,
            passage_non_null_threshold=task_args.passage_non_null_threshold, 
            minimal_non_null_threshold=task_args.minimal_non_null_threshold,verbose=task_args.verbose)

    trainer = MRCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=postprocessor.process_references_and_predictions,  # see QATrainer in Huggingface
        compute_metrics=compute_metrics,
    )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    # training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # validation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
    main()
