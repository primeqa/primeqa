import logging
import os
import sys

import datasets
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type
from importlib import import_module
import traceback
from transformers import HfArgumentParser, TrainingArguments, DataCollatorWithPadding, AutoConfig, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint, set_seed

from oneqa.mrc.models.task_model import ModelForDownstreamTasks
from oneqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from oneqa.mrc.processors.preprocessors.default import DefaultPreProcessor
from oneqa.mrc.processors.preprocessors.tydiqa import TyDiQAPreprocessor
from oneqa.mrc.trainers.default import MRCTrainer
from oneqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from oneqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from oneqa.mrc.metrics.tydi_f1.tydi_f1 import TyDiF1

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

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input test data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
                    "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
                    "be faster on GPU but will be slower on TPU)."
        },
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
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
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
        metadata={"help": "Only used when preparing training features, not for decoding. "
                          "This ratio will be used when the example has a short answer, but "
                          "the span does not. Specifically we will keep the span with "
                          "probability negative_sampling_prob_when_has_answer."
                 },
    )
    negative_sampling_prob_when_no_answer: float = field(
        default=0.04,
        metadata={"help": "Only used when preparing training features, not for decoding. "
                          "This ratio will be used when the example has NO short answer. "
                          "Specifically we will keep spans from this example with "
                          "probability negative_sampling_prob_when_has_answer."
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
                  "choices": ["score_diff_based", "target_type_weighted_score_diff", "weighted_sum_target_type_and_score_diff"]
                  }
    )
    task_heads: str = field(
        default='EXTRACTIVE_HEAD',
        metadata={"help": "The name of the task head to use.",
                  "choices": ["oneqa.mrc.models.heads.extractive.EXTRACTIVE_HEAD"]
                  }
    )
    preprocessor: str = field(
        default='TyDiQAPreprocessor',
        metadata={"help": "The name of the preprocessor to use.",
                  "choices": ["oneqa.mrc.processors.preprocessors.tydiqa.TyDiQAPreprocessor"]
                  }
    )
    postprocessor: str = field(
        default='ExtractivePostProcessor',
        metadata={"help": "The name of the postprocessor to use.",
                  "choices": ["oneqa.mrc.processors.postprocessors.extractive.ExtractivePostProcessor"]
                  }
    )
    eval_metrics: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the evaluation metric function."
                  }
    )

def class_reference(class_reference_as_str: str) -> Type:
    """
    Given a fully qualified path to a class reference, return a pointer to the class reference
    :param str class_reference_as_str: the fully qualified path (expects the fully qualified
        path in dot notation, e.g. 
        oneqa.mrc.processors.postprocessors.extractive.ExtractivePostProcessor)
    :return: class
    :rtype: Type
    """
    def _split_into_class_and_module_name(class_path):
        modules = class_path.split('.')
        if len(modules) > 1:
            return ".".join(modules[:-1]), modules[-1]
        else:
            return class_path, None

    try:
        module_name, class_name = _split_into_class_and_module_name(class_reference_as_str)
        module_reference = import_module(module_name)
        if class_name is None:
            return module_reference
        else:
            return getattr(module_reference, class_name)
    except Exception as ex:
        traceback.print_exc()  # Shows additional traceback for why imports fail
        raise TypeError("Unable to resolve the string {} to a fully qualified class path".format(class_reference_as_str))

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

#    training_args = TrainingArguments(
#        output_dir='/dccstor/zhrong-nmt/QA/oneqa/exp/03212022/',
#        do_train=True,
#        do_eval=True,
#        num_train_epochs=0.1,
#        fp16=False,
#    )

    checkpoint_for_eval = model_args.model_name_or_path
        #'/dccstor/bsiyer6/OneQA/test-model/pytorch_model.bin'
    scorer_type = task_args.scorer_type
        #weighted_sum_target_type_and_score_diff'

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

    model_name = model_args.model_name_or_path
        #'xlm-roberta-base'
    task_heads = class_reference(task_args.task_heads)
        #EXTRACTIVE_HEAD  # TODO parameterize
    config = AutoConfig.from_pretrained(
        model_name,
        # cache_dir=model_args.cache_dir,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        # cache_dir=model_args.cache_dir,
        use_fast=True,
        # revision=model_args.model_revision,
        # use_auth_token=True if model_args.use_auth_token else None,
        config=config,
    )
    
    # TODO: remove during parameterization 
    if not training_args.do_train:
        model_name = checkpoint_for_eval
    model = ModelForDownstreamTasks.from_config(
        config,
        model_name,
        task_heads=task_heads,
    )
    model.set_task_head(next(iter(task_heads)))

    # load data
    logger.info('Loading dataset')
    raw_datasets = datasets.load_dataset(data_args.dataset_name, data_args.dataset_config_name)
        #("tydiqa", "primary_task")

    # load preprocessor
    preprocessor_class = class_reference(task_args.preprocessor)
        #TyDiQAPreprocessor  # TODO parameterize
    preprocessor = preprocessor_class(
        stride=data_args.doc_stride, #128,
        tokenizer=tokenizer,
        negative_sampling_prob_when_has_answer=data_args.negative_sampling_prob_when_has_answer,
        negative_sampling_prob_when_no_answer=data_args.negative_sampling_prob_when_no_answer,
        load_from_cache_file=data_args.overwrite_cache,
    )

    # process train data
    train_dataset = raw_datasets["train"]
    max_train_samples = data_args.max_train_samples
        #1000
    if max_train_samples is not None:  # if data_args.max_train_samples is not None:
        # We will select sample from whole data if argument is specified
        train_dataset = train_dataset.select(range(max_train_samples))
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        # train_dataset = preprocessor.adapt_dataset(train_dataset)
        # train_dataset = train_dataset.map(  # TODO debug
        #     preprocessor.process_train,
        #     batched=True,
        #     num_proc=1,  # data_args.preprocessing_num_workers,
        #     remove_columns=train_dataset.column_names,
        #     # load_from_cache_file=not data_args.overwrite_cache,
        #     load_from_cache_file=False,
        #     desc="Running tokenizer on train dataset",
        # )
        # train_dataset = preprocessor.subsample_features(train_dataset)
        _, train_dataset = preprocessor.process_train(train_dataset)

    # process val data
    eval_examples = raw_datasets["validation"]
    max_eval_samples = data_args.max_eval_samples
        #10 #250
    if max_eval_samples is not None:  # data_args.max_eval_samples is not None:
        # We will select sample from whole data
        eval_examples = eval_examples.select(range(max_eval_samples))
    # Validation Feature Creation
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        # eval_examples = preprocessor.adapt_dataset(eval_examples)
        # eval_dataset = eval_examples.map(
        #     preprocessor.process_eval,
        #     batched=True,
        #     num_proc=1,  # data_args.preprocessing_num_workers,
        #     remove_columns=eval_examples.column_names,
        #     # load_from_cache_file=not data_args.overwrite_cache,
        #     load_from_cache_file=False,
        #     desc="Running tokenizer on training dataset",
        # )
        eval_examples, eval_dataset = preprocessor.process_eval(eval_examples)

    # process test data

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if training_args.fp16 else None)

    # train

    postprocessor_class = class_reference(task_args.postprocessor)
        #ExtractivePostProcessor  # TODO parameterize
    postprocessor = postprocessor_class(
        k=data_args.n_best_logits, #5,
        n_best_size=data_args.n_best_size, #20,
        max_answer_length=data_args.max_answer_length, #30,
        scorer_type=SupportedSpanScorers(scorer_type))

    if task_args.eval_metrics:
        metrics_fn = class_reference(task_args.eval_metrics)
    else:
        metrics_fn = None
        #None  # TODO metrics

    trainer = MRCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=postprocessor.process,  # see QATrainer in Huggingface
        compute_metrics=metrics_fn,
    )

    
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        # raise ValueError("Nothing implemented beyond this point")
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
    # max_train_samples = (
    #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    # )
    # metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    # run val

    # run test


if __name__ == '__main__':
    main()
