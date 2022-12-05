import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import List, Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import pandas as pd
from datasets import load_dataset

from pickle import NONE
from primeqa.tableqa.tapex.utils.argument_utils_for_tapex import DataTrainingArguments,ModelArguments
import pandas as pd
import transformers
from filelock import FileLock
import os
from primeqa.tableqa.models.tableqa_seq2seq_model import TableQASeq2SeqModel
from transformers import (
    AutoConfig,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TapexTokenizer,
    set_seed,
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from primeqa.tableqa.tapex.preprocessors.wikisql import preprocess_tableqa_function_wikisql
from primeqa.tableqa.tapex.preprocessors.wikitablequestions import preprocess_tableqa_function_wtq

logger = logging.getLogger(__name__)

class TapexModel():
    def __init__(self,path_to_config_json): 
        print("reading the config from ",path_to_config_json)
        self._config_json = path_to_config_json

    @property
    def model(self):
        """ Propery of TableQA model.
        Returns:
            Sequence to sequence model object (based on model name)
        """
        return self._model

    @property
    def tokenizer(self):
        """ Property of Tapex model.
        Returns:
            Tokenizer class object based on the model name/ path
        """
        return self._tokenizer
        

        

    def predict(self,data_dict,queries_list):
        """This function takes a table dictionary and a list of queries as input and returns the answer to the queries using the TableQA model.

        Args:
            data_dict (Dict): Table in dict format
            queries_list (List): List of queries

        Returns:
            Dict: Returns a dictionary of query and the predicted answer.
        """
        print("in predict for TapexModel with data: ",data_dict , " ,queries:", queries_list)
        self.load_model_from_config(self._config_json)
        table = pd.DataFrame.from_dict(data_dict)
        inputs = self._tokenizer(table, queries_list, padding='max_length', return_tensors="pt")
        outputs = self._model.generate(**inputs)
        answers = self._tokenizer.batch_decode(outputs, skip_special_tokens=True)

        query_answer_dict = {}
        for query, answer in zip(queries_list, answers):
            query_answer_dict[query] = answer
        return query_answer_dict


    def load_model_from_config(self,config_json) :
        print("loading from config at ",config_json)
        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
        
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(config_json))     

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
        logger.info(f"Training/evaluation parameters {training_args}")

        # Set seed before initializing model.
        set_seed(training_args.seed)                                                      
    
        # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        # Load pretrained model and tokenizer
        #
        # Distributed training:
        # The .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.

        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        # IMPORTANT: the initial BART model's decoding is penalized by no_repeat_ngram_size, and thus
        # we should disable it here to avoid problematic generation
        config.no_repeat_ngram_size = 0
        config.max_length = 1024
        config.early_stopping = False

        # load tapex tokenizer
        self._tokenizer = TapexTokenizer.from_pretrained(
            model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
            add_prefix_space=True,
        )

        # load Bart based Tapex model (default tapex-large)
        self._model = BartForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )

        if self._model.config.decoder_start_token_id is None:
            raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    def train(self):
        
        print("loading from config")
        self.load_model_from_config(self._config_json)

        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(self._config_json))  
        
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

        if data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
        else:
            data_files = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
                extension = data_args.train_file.split(".")[-1]
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
                extension = data_args.validation_file.split(".")[-1]
            if data_args.test_file is not None:
                data_files["test"] = data_args.test_file
                extension = data_args.test_file.split(".")[-1]
            datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
        
        column_names = datasets["train"].column_names
        max_target_length = data_args.max_target_length
        padding = "max_length" if data_args.pad_to_max_length else False
        if training_args.label_smoothing_factor > 0 and not hasattr(self._model, "prepare_decoder_input_ids_from_labels"):
            logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{self._model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )

        if data_args.dataset_name == 'wikisql':
            preprocess_tableqa_function = partial(preprocess_tableqa_function_wikisql, model_args=model_args, data_args=data_args,is_training=False)
            
        elif data_args.dataset_name == 'wikitablequestions':
            preprocess_tableqa_function = partial(preprocess_tableqa_function_wtq, model_args=model_args, data_args=data_args,is_training=False)
        else:
            raise ValueError("Only wikisql and wikitablequestions are supported")

        preprocess_tableqa_function_training = partial(preprocess_tableqa_function, is_training=True)
        
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = datasets["train"]
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_tableqa_function_training,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        # Data collator
        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else self._tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            self._tokenizer,
            model=self._model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            return preds, labels

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = self._tokenizer.batch_decode(preds, skip_special_tokens=True)
            if data_args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = self._tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

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

        # Initialize our Trainer
        trainer = Seq2SeqTrainer(
            model=self._model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self._tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        )

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
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

    def eval(self):
        
        print("loading from config")
        self.load_model_from_config(self._config_json)

        parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(self._config_json))  
        
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

        if data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            datasets = load_dataset(data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir)
        else:
            data_files = {}
            if data_args.test_file is not None:
                data_files["test"] = data_args.test_file
                extension = data_args.test_file.split(".")[-1]
            datasets = load_dataset(extension, data_files=data_files, cache_dir=model_args.cache_dir)
        
        column_names = datasets["test"].column_names
        max_target_length = data_args.max_target_length
        padding = "max_length" if data_args.pad_to_max_length else False
        if training_args.label_smoothing_factor > 0 and not hasattr(self._model, "prepare_decoder_input_ids_from_labels"):
            logger.warning(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
                f"`{self._model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
            )

        if data_args.dataset_name == 'wikisql':
            preprocess_tableqa_function = partial(preprocess_tableqa_function_wikisql, model_args=model_args, data_args=data_args,is_training=False)
            
        elif data_args.dataset_name == 'wikitablequestions':
            preprocess_tableqa_function = partial(preprocess_tableqa_function_wtq, model_args=model_args, data_args=data_args,is_training=False)
        else:
            raise ValueError("Only wikisql and wikitablequestions are supported")

        preprocess_tableqa_function_training = partial(preprocess_tableqa_function, is_training=True)
        
        if "test" not in datasets:
            raise ValueError("eval requires a train dataset")
        test_dataset = datasets["test"]

        test_dataset = test_dataset.map(
            preprocess_tableqa_function_training,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )
        # Data collator
        label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else self._tokenizer.pad_token_id
        data_collator = DataCollatorForSeq2Seq(
            self._tokenizer,
            model=self._model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

        def postprocess_text(preds, labels):
            preds = [pred.strip() for pred in preds]
            labels = [label.strip() for label in labels]

            return preds, labels

        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
            decoded_preds = self._tokenizer.batch_decode(preds, skip_special_tokens=True)
            if data_args.ignore_pad_token_for_loss:
                # Replace -100 in the labels as we can't decode them.
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
            decoded_labels = self._tokenizer.batch_decode(labels, skip_special_tokens=True)

            # Some simple post-processing
            decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

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
        training_args.predict_with_generate=True
        # Initialize our Trainer
        trainer = Seq2SeqTrainer(
            model=self._model,
            args=training_args,
            valid_dataset=test_dataset,
            tokenizer=self._tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics if training_args.predict_with_generate else None,
        )

        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        metrics["test_samples"] = min(len(test_dataset))

        trainer.log_metrics("test", metrics)
        trainer.save_metrics("test", metrics)
        trainer.save_state()




