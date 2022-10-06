"""This script allows to train & evaluate a data generation model or apply Active Learning in combination with an RC model if appropriate."""

from copy import copy
import dataclasses
from functools import partial
import json
import logging
import math
from operator import itemgetter
import os
from re import search
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Tuple

import datasets
from datasets import get_dataset_config_names, load_dataset, Dataset
from examples.datagen_with_al.utils.data import get_datasets, expand_answers
from primeqa.al.models.al import ActiveLearner, GenALScorer, TrainerConfig
from primeqa.qg.metrics.generation_metrics import rouge_metrics
from primeqa.qg.models.qg_model import QGModel
from primeqa.qg.processors.data_loader import QGDataLoader
from primeqa.qg.trainers.qg_trainer import GenTrainer, QGTrainer
from primeqa.qg.utils.data import dicts_to_feature_dict, prepare_labelled_data, select_unique
from primeqa.qg.utils.data_collator import (
    DataCollatorForSeq2SeqWithDecoderInputs,
    T2TDataCollator,
)
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, TrainingArguments, set_seed

logger = logging.getLogger()

# unique IDs for the trainer, also used for metric record
GEN_TRAINER_ID = 'gen_trainer'
RC_TRAINER_ID = 'rc_trainer'


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="t5-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models; should be a seq-to-seq model",
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do we want to store the pretrained models downloaded from s3"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dataset: str = field(
        default=None,
        metadata={
            "help": "Name of the dataset to train the qg model",
        },
    )
    eval_dataset: str = field(
        default=None,
        metadata={
            "help": "Name of the dataset to evaluate the qg model",
        },
    )
    max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )
    num_worker: Optional[int] = field(
        default=1,
        metadata={"help": "The number of workers for processing datasets"},
    )


@dataclass
class ALArguments:
    """
    Arguments for performing Active Learning.
    """
    do_al: Optional[bool] = field(
        default=False, metadata={"help": "Whether to perform Active Learning"}
    )


@dataclass
class InferenceArguments:
    do_generate: Optional[bool] = field(
        default=False, metadata={"help": "Whether to generate questions"}
    )
    gen_output_path: Optional[str] = field(
        default="generated_data",
        metadata={"help": "path to dir where generated questions will be saved"},
    )
    predict_dataset: Optional[str] = field(
        default=None, metadata={"help": "The dataset used for generating data"}
    )
    exclude_dataset: Optional[str] = field(
        default_factory=list,
        metadata={"help": "the dataset(s) for which the contexts are excluded"}
    )
    max_gen_length: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum number of tokens generated"}
    )
    skip_context_length_above: Optional[int] = field(
        default=None,
        metadata={"help": "The maximum context length in tokens (contexts with more tokens will be discarded)"}
    )
    skip_context_length_below: Optional[int] = field(
        default=100,
        metadata={"help": "The minimum context length in tokens (contexts with less tokens will be discarded); default 100 - 0 disables it"}
    )
    num_shards: Optional[int] = field(
        default=1,
        metadata={
            "help": "The number of shards which will be used to predict data (1 means no sharding)"
        },
    )
    shard_size: Optional[int] = field(
        default=None, metadata={"help": "The size of the shards"}
    )
    shards: Optional[List[int]] = field(
        default=None,
        metadata={
            "help": "The indices of the shards to process, where the shards are computed according to `num_shards` and `shard_size`; useful for splitting generation into several processes"
        },
    )

def main(raw_args):
    parser = HfArgumentParser(
        (
            ModelArguments,
            Seq2SeqTrainingArguments,
            DataTrainingArguments,
            ALArguments,
            InferenceArguments,
        )
    )

    # type annotations
    model_args: ModelArguments
    training_args: Seq2SeqTrainingArguments
    data_args: DataTrainingArguments
    al_args: ALArguments
    inference_args: InferenceArguments

    if len(raw_args) == 2 and raw_args[1].endswith(".json"):
        model_args, training_args, data_args, al_args, inference_args = parser.parse_json_file(
            json_file=raw_args[1]
        )
    elif len(raw_args) == 1:
        model_args, training_args, data_args = parser.parse_dict(raw_args[0])
    else:
        (
            model_args,
            training_args,
            data_args,
            al_args,
            inference_args,
        ) = parser.parse_args_into_dataclasses()
    
    # some arguments have to be hardcoded in order for HF Trainer to work
    training_args.predict_with_generate = True
    training_args.prediction_loss_only = False

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    qg_model = QGModel(model_args.model_name_or_path, modality='passage_qa2s')

    qgdl = QGDataLoader(
        tokenizer=qg_model.tokenizer,
        modality='passage_qa2s',
        input_max_len=data_args.max_len,
        target_max_len=data_args.target_max_len,
    )

    # load datasets
    if al_args.do_al or training_args.do_train:
        train_dataset = get_datasets(data_args.train_dataset)
        train_dataset = expand_answers(train_dataset, separate_answers=False)
    if training_args.do_eval:
        validation_dataset = get_datasets(data_args.eval_dataset)
        validation_dataset = expand_answers(validation_dataset, separate_answers=False)

    # process data
    validation_dataset = qgdl.create(validation_dataset) if training_args.do_eval else None
    compute_metrics = rouge_metrics(qg_model.tokenizer)

    gen_train_class = GenTrainer
    gen_trainer_kwargs = dict(
        max_gen_length=30,
        model=qg_model.model,
        tokenizer=qg_model.tokenizer,
        args=training_args,
        eval_dataset=validation_dataset,
        data_collator=DataCollatorForSeq2SeqWithDecoderInputs(qg_model.tokenizer),
        compute_metrics=compute_metrics,
    )

    if training_args.do_eval or training_args.do_train:
        # for evaluation and training we need to create the trainer first
        # for training we also need to preprocess training data
        trainer = gen_train_class(**gen_trainer_kwargs, train_dataset=qgdl.create(train_dataset) if training_args.do_train else None)

    # evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)

    # training
    if training_args.do_train:
        train_result = trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # AL
    if al_args.do_al:
        # the ActiveLearner gets passed trainer configs which contain the trainer class and the trainer kwargs
        
        def preprocess_fn_wrapper(process_fn: Callable, add_examples_to_output: bool = False) -> Callable[[Dataset], Tuple[Dataset, Dataset]]:
            # wrapper to have access to training_args
            def wrapped_process_fn(examples) -> Tuple[Dataset, Dataset]:
                with training_args.main_process_first(desc="Dataset pre-processing"):
                    result = process_fn(examples)
                if add_examples_to_output:
                    # the pre-processing function returns only processed examples but we need the examples as well
                    return examples, result
                # otherwise return the result, should be examples and processed examples
                return result
            return wrapped_process_fn
        gen_trainer_config = TrainerConfig(GEN_TRAINER_ID, gen_train_class, gen_trainer_kwargs, preprocess_fn_wrapper(qgdl.create, add_examples_to_output=True), preprocess_fn_wrapper(qgdl.create, add_examples_to_output=True), None)

        # TODO add rc trainer with separate arguments, maybe prefix them on the command line?
        # rc_trainer_config = TrainerConfig(RC_TRAINER_ID, rc_trainer_class, rc_trainer_kwargs, preprocess_fn_wrapper(preprocessor.process_train), preprocess_fn_wrapper(preprocessor.process_eval), None)
        # rc_trainer_class = MRCTrainer
        #     rc_trainer_kwargs = dict(
        #         model=model,
        #         args=training_args,
        #         train_dataset=None,
        #         eval_dataset=eval_dataset if training_args.do_eval else None,
        #         eval_examples=eval_examples if training_args.do_eval else None,
        #         tokenizer=tokenizer,
        #         data_collator=data_collator,
        #         post_process_function=postprocessor.process_references_and_predictions,  # see QATrainer in Huggingface
        #         compute_metrics=compute_metrics,
        #     )

        al = ActiveLearner(training_args.output_dir, (gen_trainer_config,))
        # TODO choose strategy via command line arguments
        # strategy = RCALScorer(rc_trainer_id=0)
        # a special token id map is needed to generate sequences - these are valid for the QA2S model
        special_token_id_map = dict(
            bos_token_id=qg_model.tokenizer.convert_tokens_to_ids('<q>'),
            eos_token_id=qg_model.tokenizer.convert_tokens_to_ids('</q>'),
            bos_token_id_2=qg_model.tokenizer.convert_tokens_to_ids('<a>'),
            eos_token_id_2=qg_model.tokenizer.convert_tokens_to_ids('</a>'),
        )
        strategy = GenALScorer(strategy=GenALScorer.Strategy.SENTENCE_PROBABILITY_DROPOUT, max_gen_length=30, special_token_id_map=special_token_id_map, gen_trainer_id=GEN_TRAINER_ID)

        # perform training via active learning
        # TODO allow to set arguments via command line
        logger.info("***** Running Active Learning *****")
        metrics = al.run(examples=train_dataset, strategy=strategy, num_iterations=4, num_samples_per_iteration=50, feature_id_column='id')
        if metrics:
            logger.info("***** Active Learning results *****")
            for trainer_id in sorted(metrics.keys()):
                logger.info(" Trainer id = %s", trainer_id)
                if isinstance(next(iter(metrics[trainer_id].values())), Mapping):
                    for iteration in sorted(metrics[trainer_id].keys()):
                        logger.info("  Iteration = %s", iteration)
                        for metric, value in sorted(metrics[trainer_id][iteration].items(), key=itemgetter(0)):
                            logger.info("   %s = %s", metric, value)
                else:
                    for metric, value in sorted(metrics[trainer_id].items(), key=itemgetter(0)):
                        logger.info("   %s = %s", metric, value)
        logger.info("***** Finished Active Learning *****")
                    
    # inference
    if inference_args.do_generate:
        # get dataset
        dataset = get_datasets(inference_args.predict_dataset)

        # same setup as Shakeri et al.: Only contexts with >= 100 tokens, contexts truncated to 550 tokens, 100000 contexts randomly drawn

        dataset = select_unique(dataset, 'context')

        if inference_args.skip_context_length_above:
            logging.info(f"Skipping contexts > {inference_args.skip_context_length}")
            try:
                dataset = dataset.filter(lambda x: len(qg_model.tokenizer.tokenize(x['context'], add_special_tokens=False)) <= inference_args.skip_context_length, num_proc=data_args.num_worker)
            except IndexError:
                logging.info("No data left after filtering for context length, exiting.")
                exit()

        if inference_args.exclude_dataset:
            # exclude contexts for generation
            logging.info(f"Excluding contexts from specified data")
            exclude_dataset = load_dataset('datasets/shared-task', name=inference_args.exclude_dataset)
            exclude_contexts = exclude_dataset.flatten_indices().unique('context')
            dataset = dataset.filter(lambda x: x['context'] not in exclude_contexts, num_proc=data_args.num_worker)

        if inference_args.skip_context_length_below:
            logging.info(f"Discarding documents with less than {inference_args.skip_context_length_below} tokens")
            dataset = dataset.filter(lambda x: inference_args.skip_context_length_below <= len(qg_model.tokenizer.tokenize(x['context'], add_special_tokens=False)), num_proc=data_args.num_worker)
        
        num_samples = min(100000, len(dataset))
        logging.info(f"Randomly selecting {num_samples} documents (from {len(dataset)} available documents)")
        # shuffle data using seed to make sure that we have always the same documents
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(num_samples))
        logging.info(f"Truncating documents to {550} tokens")
        # NOTE somehow this part doesn't work with multiprocessing
        dataset = dataset.map(lambda x: {'context': qg_model.tokenizer.convert_tokens_to_string(qg_model.tokenizer.tokenize(x['context'], add_special_tokens=False)[:550])}, num_proc=data_args.num_worker)
        dataset = dataset.select(range(10))
        
        # create trainer
        gen_trainer = GenTrainer(
            model=qg_model.model,
            args=training_args,
            tokenizer=qg_model.tokenizer,
            data_collator=DataCollatorForSeq2SeqWithDecoderInputs(
                qg_model.tokenizer
            ),
            max_gen_length=inference_args.max_gen_length,
        )

        # process dataset in shards
        if inference_args.shard_size is not None:
            # ceil so that there is a maximum of `shard_size` samples in each shard
            inference_args.num_shards = max(
                1, math.ceil(len(dataset) / inference_args.shard_size)
            )
        if inference_args.shards is not None:
            assert 0 <= max(inference_args.shards) < inference_args.num_shards
            shard_indices = inference_args.shards
        else:
            shard_indices = range(inference_args.num_shards)

        for i in shard_indices:
            if inference_args.num_shards == 1:
                shard = dataset
                logger.info(f"Processing {len(shard)} samples.")
            else:
                shard = dataset.shard(inference_args.num_shards, i, contiguous=True)
                logger.info(
                    f"Processing shard {i} ({inference_args.num_shards} in total) with {len(shard)} samples."
                )

            # move answers and question columns so that we might use them later
            shard = prepare_labelled_data(shard)

            # process data
            shard = qgdl.create(shard)

            # predict
            predictions = gen_trainer.predict(test_dataset=shard)
            shard = datasets.Dataset.from_dict(dicts_to_feature_dict(predictions))

            if shard:
                # save generated data to disk - `flatten_indices` makes sure that only data related to this shard is stored
                shard.flatten_indices().save_to_disk(
                    inference_args.gen_output_path if inference_args.num_shards == 1 else os.path.join(inference_args.gen_output_path, str(i))
                )
                logger.info(
                    f"Saved {len(shard)} rows to {inference_args.gen_output_path}"
                )


if __name__ == "__main__":
    main(sys.argv)
