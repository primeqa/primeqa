#!/usr/bin/env python
# based loosely on https://github.com/huggingface/transformers/blob/main/examples/pytorch/text-classification/run_glue.py
#
#
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for n-way classification."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import dataclass, field
from sysconfig import is_python_build
from typing import Optional, List
import json
from shutil import move
from os import path

import numpy as np
from datasets import Dataset, DatasetDict


import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
)
from transformers.trainer_utils import is_main_process

from primeqa.boolqa.processors.postprocessors.boolqa_classifier import BoolQAClassifierPostProcessor
from primeqa.boolqa.processors.preprocessors.boolqa_classifier import BoolQAClassifierPreProcessor
from primeqa.boolqa.processors.dataset.mrc2dataset import create_dataset_from_run_mrc_output

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




@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
    id_key: str = field(
        default="example_id",
        metadata={"help": "a field that can be used as a unique identifier of input records"}
    )
    label_list: List[str] = field(
        default=None,
        metadata={"help": "the labels used by the classifier (order is significant)"}
    )
    output_label_prefix: str = field(
        default=None,
        metadata={"help": "a prefix used in the output file eval_predictions.json to distinguish fields created by this invocation of the classifier"}
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


def main(raw_args):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(raw_args) == 2 and raw_args[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(raw_args[1]))
    elif len(raw_args) == 1:
        model_args, data_args, training_args = parser.parse_dict(raw_args[0])
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

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

    num_labels = len(model_args.label_list)        
    logger.info("The following labels are being used, all other instances will be discarded.")
    logger.info(model_args.label_list)




    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=model_args.output_label_prefix,
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


    raw_datasets={}
    raw_datasets['validation']=create_dataset_from_run_mrc_output(data_args.test_file, unpack=False)

    #max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)
    # load preprocessor
    preprocessor_class = BoolQAClassifierPreProcessor # TODO task_args.preprocessor
    preprocessor = preprocessor_class(
        sentence1_key=model_args.sentence1_key,
        sentence2_key=model_args.sentence2_key,
        tokenizer=tokenizer,
        load_from_cache_file=not data_args.overwrite_cache,
        max_seq_len=tokenizer.model_max_length,
        padding=padding
    )

    # process eval data
    eval_examples = raw_datasets["validation"]
    max_eval_samples = data_args.max_eval_samples
    if max_eval_samples is not None:  # data_args.max_eval_samples is not None:
        # We will select sample from whole data
        eval_examples = eval_examples.select(range(max_eval_samples))
    # Validation Feature Creation
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_examples, eval_dataset = preprocessor.process_eval(eval_examples)


    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )


    # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    elif training_args.fp16:
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)
    else:
        data_collator = None


    
    postprocessor_class = BoolQAClassifierPostProcessor  # TODO # taskargs.
    postprocessor = postprocessor_class(
        k=10, 
        drop_label=model_args.drop_label,
        label_list = model_args.label_list,
        id_key=model_args.id_key,
        output_label_prefix=model_args.output_label_prefix
    )


    # Initialize our Trainer
    #trainer = NWayTrainer( 
    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=eval_dataset,
        compute_metrics=None, #compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator,
    
    )
    # Predict on unlabeled data

    logger.info("*** Predict ***")
    predictions = trainer.predict(eval_dataset, metric_key_prefix="predict").predictions

    eval_preds = postprocessor.process_references_and_predictions(eval_examples, eval_dataset, predictions)

    with open(os.path.join(training_args.output_dir, 'eval_predictions.json'), 'w') as f:
        json.dump(eval_preds.predictions, f, indent=4)
    with open(os.path.join(training_args.output_dir, 'eval_predictions_processed.json'), 'w') as f:
        json.dump(eval_preds.processed_predictions, f, indent=4)


if __name__ == "__main__":
    main(sys.argv)
