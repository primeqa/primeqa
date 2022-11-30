#!/usr/bin/env python
# coding=utf-8
# This file is an adaptation of run_glue.py from The HuggingFace Inc. team
# which is licensed under the Apache License, Version 2.0 (the "License");
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
# This code is used to fine-tune the library models for n-way classification.
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional, List
from glob import glob
import pandas as pd
import json
from shutil import move
from os import path

import numpy as np
from datasets import load_dataset, load_metric, Dataset, Features, Value, ClassLabel
from numpy.lib.function_base import append
from primeqa.text_classification.processors.postprocessors.text_classifier import TextClassifierPostProcessor
from primeqa.text_classification.processors.preprocessors.text_classifier import TextClassifierPreProcessor
from primeqa.boolqa.processors.dataset.mrc2dataset import create_dataset_from_run_mrc_output
from primeqa.text_classification.metrics.classification import Classification
from primeqa.mrc.data_models.eval_prediction_with_processing import EvalPredictionWithProcessing
from primeqa.text_classification.trainers.nway import NWayTrainer
from primeqa.mrc.run_mrc_utils import object_reference # TODO this should be a utils file

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.6.0")

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: str = field(
        default=None,
        metadata={"help": "The name of the task to train on: "},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
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
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    balanced: Optional[bool] = field(
        default=False,
        metadata={
            "help": "balance the data if true "
        },
    )
    label_list: List[str] = field(
        default=None,
        metadata={"help": "the labels used by the classifier (order is significant)"}
    )
    sentence2_key: str = field(
        default=None,
        metadata={"help": "the field in the input dataset to use as sentence2" }
    )        
    drop_label: str = field(
        default=None,
        metadata={"help": "dropping label 'no_answer' converts ternary classifier into binary"}  
    )
    sentence1_key: str = field(
        default="question",
        metadata={"help": "the field in the input dataset to use as sentence1" }
    )
    example_id_key: str = field(
        default="example_id",
        metadata={"help": "a field that can be used as a unique identifier of input records"}
    )
    language_key: str = field(
        default=None,
        metadata={"help": "a field that can be used as a unique identifier of input records"}
    )
    output_label_prefix: str = field(
        default=None,
        metadata={"help": "a prefix used in the output file eval_predictions.json to distinguish fields created by this invocation of the classifier"}
    )

    def __post_init__(self):
        if self.train_file is None and (self.validation_file is None and self.test_file is None):
            raise ValueError("Need a training and test/validation file.")
        else:
            if self.train_file is not None:
                train_extension = self.train_file.split(".")[-1]
                assert train_extension in ["csv", "json","tsv", "jsonl"], "`train_file` should be a csv, tsv or a json(l) file."
            if self.validation_file is not None:
                validation_extension = self.validation_file.split(".")[-1]
                if self.train_file is not None:
                    assert train_extension == validation_extension
                else:
                    assert validation_extension in ["csv", "json", "tsv", "jsonl"], "`validation_file` should be a csv, tsv or a json(l) file."
            if self.test_file is not None:
                test_extension = self.test_file.split(".")[-1]
                assert test_extension in ["csv", "json", "tsv", "jsonl"], "`test_file` should be a csv, tsv or a json(l) file."


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
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class TaskArguments:
    """
    Task specific arguments.
    """
    preprocessor: object_reference = field(
        default=TextClassifierPreProcessor,
        metadata={"help": "The name of the preprocessor to use.",
                  "choices": [TextClassifierPreProcessor]
                  }
    )
    postprocessor: object_reference = field(
        default=TextClassifierPostProcessor,
        metadata={"help": "The name of the postprocessor to use.",
                  "choices": [TextClassifierPostProcessor]
                  }
    )
    do_mrc_pipeline: bool = field(
        default=False,
        metadata={
            "help": "if true, then the predict dataset will be loaded from json format produced by run_mrc.py"
        }
    )

def create_a_backup_file_if_file_exists(original_file: str):
    if path.isfile(original_file):
        backup_file = '%s.bak' % original_file
        logging.debug('Found a pre-existing file at location %s.  Backing it up to %s' %
                      (original_file, backup_file))
        move(original_file, backup_file)

def save_to_json_file(obj_to_save, out_file_path, with_backup=True, indent=4, sort_keys=True):
    if with_backup:
        create_a_backup_file_if_file_exists(out_file_path)

    logging.debug('Writing %s as json to file: %s' % (type(obj_to_save), out_file_path))
    with open(out_file_path, 'w') as outfile:
        json.dump(obj_to_save, outfile, indent=indent, sort_keys=sort_keys)


def restrict_labels(dataset : Dataset, sentence1_key : str, label_list : List[str]) -> Dataset:
    ''' discard instances whose label is not in the label list '''
    examples_with_label = []
    for example in dataset:
        if example['label'] in label_list and example[sentence1_key] != None:
            examples_with_label.append(example)
    df = pd.DataFrame(examples_with_label)
    return Dataset.from_pandas(df)
# TODO - something like this is more idiomatic for pandas?
# df_all=datasets['train'].to_pandas()
# mask=~df_all[data_args.sentence1_key].isna() & df_all['label'].isin(data_args.label_list)
# df=df_all[mask]


def balance_dataset(data_args, datasets):
    '''balance the number of labels in a dataset'''
    examples_by_label = {}

    for label in data_args.label_list:
        examples_by_label[label] = []

    for example in datasets['train']:
        examples_by_label[example['label']].append(example)
            
        # min class size is smaller than dataset size
    min_count = len(datasets['train'])

    for label in examples_by_label:
        if len(examples_by_label[label]) < min_count:
            min_count = len(examples_by_label[label])
        
    balanced_examples = []

    if data_args.max_train_samples is not None and data_args.max_train_samples < min_count * len(examples_by_label):
        min_count = int(data_args.max_train_samples / len(examples_by_label))

    for label in examples_by_label:
        def seed():
          return 0.1
            # should we have something like this? its possible the data is sorted in some manner (eg alphabetically)
        random.shuffle(examples_by_label[label], seed)
        balanced_examples.extend(examples_by_label[label][:min_count])
    random.shuffle(balanced_examples)
    df = pd.DataFrame(balanced_examples)
    logger.info("balanced (down sampling) dataset to " + str(min_count) + " per class.")
    return Dataset.from_pandas(df)


def main(raw_args):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, TaskArguments))
    if len(raw_args) == 2 and raw_args[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, task_args = parser.parse_json_file(json_file=os.path.abspath(raw_args[1]))
    elif len(raw_args) == 1:
        model_args, data_args, training_args, task_args = parser.parse_dict(raw_args[0])
    else:
        model_args, data_args, training_args, task_args = parser.parse_args_into_dataclasses()

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

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: Provide your own CSV/JSON training and evaluation files (see below)
    # This script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    
    # Loading a dataset from your local files.
    # CSV/JSON training and evaluation files are needed.

    # Possible options:
    # Train, Train+Eval, Train+Predict, Eval, Predict
    data_files = {}
    non_label_column_names = None
    if training_args.do_train:
        if data_args.train_file is None:
            raise ValueError("Missing train file for `do_train`.")
        train_files = [infile for infile in glob(data_args.train_file)]
        data_files["train"] = train_files

    if training_args.do_eval:
        if data_args.validation_file is None:
            raise ValueError("Missing validation file for `do_eval`.")
        validation_files = [infile for infile in glob(data_args.validation_file)]
        data_files["validation"] = validation_files
    # Get the test dataset: you can provide your own CSV/JSON test file (see below)
    if training_args.do_predict:
        if data_args.test_file is None:
            raise ValueError("Missing test file for `do_predict`.")
        test_files = [infile for infile in glob(data_args.test_file)]
        data_files["test"] = test_files     

    for key in data_files.keys():
        logger.info(f"load a local file for {key}: {data_files[key]}")

    cache_dir = None
    if not data_args.overwrite_cache:
        cache_dir=model_args.cache_dir

    features_input={
        data_args.example_id_key: Value('string'), 
        data_args.sentence1_key: Value('string'), 
        'label': Value('string')
    }
    if data_args.language_key is not None:
        features_input[ data_args.language_key ] = Value('string')
    if data_args.sentence2_key is not None:
        features_input[ data_args.sentence2_key ] = Value('string')
    features = Features( features_input )

    if task_args.do_mrc_pipeline:
        datasets={'test':create_dataset_from_run_mrc_output(data_args.test_file, unpack=False)}
    else:
        if data_args.train_file is not None and data_args.train_file.endswith(".csv") or data_args.test_file is not None and data_args.test_file.endswith(".csv"):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files, cache_dir=cache_dir, delimiter=",", features=features)
        elif (data_args.train_file is not None and data_args.train_file.endswith(".tsv")) or (data_args.validation_file is not None \
        and data_args.validation_file.endswith(".tsv")) or (data_args.test_file is not None and data_args.test_file.endswith(".tsv")):
            # Loading a dataset from local csv files
            datasets = load_dataset("csv", data_files=data_files, cache_dir=cache_dir, delimiter="\t", features=features)
        else:
            # Loading a dataset from local json files
            datasets = load_dataset("json", data_files=data_files, cache_dir=cache_dir, features=features)

    # Labels - load from file
    num_labels = len(data_args.label_list)

    logger.info("The following labels are being used, all other instances will be discarded.")
    logger.info(data_args.label_list)

    # discard instances whose label is not in the label list
    if training_args.do_train:
        logger.info("Discard training instance not in label list, if any.")
        datasets['train']=restrict_labels(datasets['train'], data_args.sentence1_key, data_args.label_list)
        if len(datasets['train']) == 0:
            raise ValueError("No training data left with labels provided in label list")

    if training_args.do_eval:
        logger.info("Discard eval instance not in label list, if any.")
        datasets['validation']=restrict_labels(datasets['validation'], data_args.sentence1_key, data_args.label_list)
        if len(datasets['validation']) == 0:
            raise ValueError("No validation data left with labels provided in label list")

    # balance the training dataset
    if training_args.do_train and data_args.balanced:
        datasets['train'] = balance_dataset(data_args, datasets)
        
            
    # See more about loading any type of standard or custom dataset at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    
    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=data_args.task_name,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    # Padding strategy
    if data_args.pad_to_max_length:
        padding = "max_length"
    else:
        # We will pad later, dynamically at batch creation, to the max sequence length in each batch
        padding = False

    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    preprocessor_class = task_args.preprocessor
    preprocessor = preprocessor_class(
        example_id_key=data_args.example_id_key,
        sentence1_key=data_args.sentence1_key,
        sentence2_key=data_args.sentence2_key,
        language_key=data_args.language_key,
        tokenizer=tokenizer,
        load_from_cache_file=not data_args.overwrite_cache,
        max_seq_len=max_seq_length,
        padding=padding,
        label_list=data_args.label_list
    )

    if training_args.do_train:
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_examples = datasets["train"]
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_examples)
        )
        if max_train_samples is not None and len(train_examples) > max_train_samples:
            train_examples = train_examples.select(range(max_train_samples))
        train_examples, train_dataset = preprocessor.process_train(train_examples)

    if training_args.do_eval:
        if "validation" not in datasets and "validation_matched" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = datasets["validation"]

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_examples)

        if max_eval_samples is not None:
            eval_examples = eval_examples.select(range(max_eval_samples))
        eval_examples, eval_dataset = preprocessor.process_eval(eval_examples)

    if training_args.do_predict or data_args.test_file is not None:
        if "test" not in datasets and "test_matched" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_examples = datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_examples = predict_examples.add_faiss_indexselect(range(data_args.max_predict_samples))

        predict_examples, predict_dataset = preprocessor.process_eval(predict_examples)

    # Log a few random samples from the training set:
    if training_args.do_train:
        for index in random.sample(range(len(train_dataset)), 3):
            logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")

    # Get the metric function
    #if data_args.task_name is not None:
    eval_metrics = Classification()
    def compute_metrics(p: EvalPredictionWithProcessing):
        return eval_metrics.compute(predictions=p.processed_predictions, references=p.label_ids)

    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None

    postprocessor_class = task_args.postprocessor
    postprocessor = postprocessor_class(
        k=10, 
        drop_label=data_args.drop_label,
        label_list = data_args.label_list,
        id_key=data_args.example_id_key,
        output_label_prefix=data_args.output_label_prefix
    )

    training_args.metric_for_best_model='eval_all_avg_f1'

    # Initialize our Trainer
    trainer = NWayTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=postprocessor.process if task_args.do_mrc_pipeline else postprocessor.process_references_and_predictions, 
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        metrics = train_result.metrics
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
    # Evaluation - have gold answers
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = min(max_eval_samples, len(eval_examples))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict on unlabeled data
    if training_args.do_predict or task_args.do_mrc_pipeline:
        logger.info("*** Predict ***")
        predict_dataset=predict_dataset.remove_columns('label')
        predictions = trainer.predict(predict_dataset, predict_examples)
        # TODO I don't think process_reference_and_predictions should be called here

        with open(os.path.join(training_args.output_dir, 'predictions.json'), 'w') as f:
            json.dump(predictions.predictions, f, indent=4)
        with open(os.path.join(training_args.output_dir, 'predictions_processed.json'), 'w') as f:
            json.dump(predictions.processed_predictions, f, indent=4)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main(sys.argv)
