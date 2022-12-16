import json
import logging
import math
import os
import sys
import glob
from dataclasses import dataclass, field
from typing import List, Optional

import datasets
from datasets import list_datasets, load_dataset
from primeqa.qg.metrics.generation_metrics import rouge_metrics
from primeqa.qg.models.qg_model import QGModel
from primeqa.qg.processors.data_loader import QGDataLoader
from primeqa.qg.trainers.qg_trainer import GenTrainer, QGTrainer
from primeqa.qg.utils.data import dicts_to_feature_dict, prepare_labelled_data
from primeqa.qg.utils.data_collator import (
    DataCollatorForSeq2SeqWithDecoderInputs,
    T2TDataCollator,
)
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, set_seed

logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default="t5-base",
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models such as t5-base, google/mt5-base"
        },
    )
    modality: str = field(
        default="table",
        metadata={
            "help": "Whether to generate questions from tables or passages",
            "choices": ["table", "passage"],
        },
    )
    gen_config: str = field(
        default="qg",
        metadata={
            "help": "Which method to use for generating question-answer pairs",
            "choices": ["qg", "qa2s"],
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
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

    dataset_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name of the dataset to train the qg model",
            "choices": list_datasets(),
        },
    )
    dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Config of the dataset loaded, e.g. 'secondary_task' for TyDiQA"
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "local file(s) in .jsonl to train on."},
    )
    eval_file: Optional[str] = field(
        default=None,
        metadata={"help": "local file(s) in .jsonl to evaluate on."},
    )
    max_len: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    target_max_len: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )


@dataclass
class InferenceArguments:
    do_generate: Optional[bool] = field(
        default=False, metadata={"help": "Whether to generate questions"}
    )
    num_questions_per_instance: Optional[int] = field(
        default=5,
        metadata={"help": "Number of questions to generate per table/passage"},
    )
    max_where_clauses: Optional[int] = field(
        default=1, metadata={"help": "Max number of filters in generated question"}
    )
    data_path: str = field(
        default="primeqa/qg/sample_table.json",
        metadata={
            "help": "Path to JSON file with LIST of tables/passages. Each table \
                              should be a dict with keys 'header' and 'rows', and passages should be str"
        },
    )
    generate_aggregate: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to generate aggregate questions with max, min, sum, etc."
        },
    )
    gen_output_path: Optional[str] = field(
        default="sample_generation.json",
        metadata={"help": "path to JSON file where generated questions will be saved"},
    )
    predict_dataset: Optional[str] = field(
        default=None, metadata={"help": "The dataset used for generating data"}
    )
    predict_dataset_config: Optional[str] = field(
        default=None,
        metadata={
            "help": "Config of the dataset loaded, e.g. 'secondary_task' for TyDiQA"
        },
    )
    predict_dataset_split: Optional[str] = field(
        default=None,
        metadata={"help": "The split of the dataset used for generating data"},
    )
    max_gen_length: Optional[int] = field(
        default=None, metadata={"help": "The maximum number of tokens generated"}
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
    print(raw_args)
    parser = HfArgumentParser(
        (
            ModelArguments,
            DataTrainingArguments,
            Seq2SeqTrainingArguments,
            InferenceArguments,
        )
    )

    if len(raw_args) == 2 and raw_args[1].endswith(".json"):
        model_args, data_args, training_args, inference_args = parser.parse_json_file(
            json_file=raw_args[1]
        )
    elif len(raw_args) == 1:
        model_args, data_args, training_args = parser.parse_dict(raw_args[0])
    else:
        (
            model_args,
            data_args,
            training_args,
            inference_args,
        ) = parser.parse_args_into_dataclasses()

    # These arguments has to be hardcoded in order for Trainer to work
    training_args.predict_with_generate = True
    training_args.remove_unused_columns = (
        True if model_args.modality == "passage" and model_args.gen_config == "qa2s" else False
    )
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
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
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
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    qg_model = QGModel(model_args.model_name_or_path, modality=model_args.modality)



    qgdl = QGDataLoader(
        tokenizer=qg_model.tokenizer,
        dataset_name=data_args.dataset_name,
        modality=model_args.modality,
        gen_config=model_args.gen_config,
        input_max_len=data_args.max_len,
        target_max_len=data_args.target_max_len,
    )

    train_dataset = None
    valid_dataset = None
    data_files={}
    
    if data_args.train_file is not None: 
        data_files['train'] = glob.glob(data_args.train_file)
    if data_args.eval_file is not None: 
        data_files['validation'] = glob.glob(data_args.eval_file)
    
    dataset = load_dataset("json", data_files=data_files)
    
    if training_args.do_train:
        if data_args.train_file is not None:
            #dataset = load_dataset("json", data_files=data_args.train_file)
            dataset = dataset['train']
            train_dataset = qgdl.create(dataset=dataset)
        else:
            train_dataset = qgdl.create(
                dataset_split="train", dataset_config=data_args.dataset_config
            )
    
    if training_args.do_eval:
        if data_args.eval_file is not None:
            # dataset = load_dataset("json", data_files=data_args.eval_file)
            # this is not a bug, by default huggingface datasets library loads any data as train split
            dataset = dataset['validation']
            valid_dataset = qgdl.create(dataset=dataset)
        else:
            valid_dataset = qgdl.create(
                dataset_split="validation", dataset_config=data_args.dataset_config
            )

    compute_metrics = rouge_metrics(qg_model.tokenizer)
   
    if training_args.do_train or training_args.do_eval:
        trainer = QGTrainer(
            model=qg_model.model,
            tokenizer=qg_model.tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=DataCollatorForSeq2SeqWithDecoderInputs(qg_model.tokenizer)
            if model_args.modality == "passage" and model_args.gen_config == "qa2s"
            else T2TDataCollator(),
            compute_metrics=compute_metrics,
        )
        compute_metrics = rouge_metrics(qg_model.tokenizer)

    if training_args.do_train:
        train_result =trainer.train(
            model_path=model_args.model_name_or_path
            if os.path.isdir(model_args.model_name_or_path)
            else None
        )
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Inference
    if inference_args.do_generate:
        if model_args.modality == "passage" and model_args.gen_config == "qa2s":
            # this configuration uses a custom trainer for prediction

            # get dataset
            dataset = load_dataset(
                inference_args.predict_dataset,
                name=inference_args.predict_dataset_config,
                split=inference_args.predict_dataset_split,
            )
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
                        os.path.join(inference_args.gen_output_path, str(i))
                    )
                    logger.info(
                        f"Saved {len(shard)} entries to {inference_args.gen_output_path}"
                    )
        else:
            # There are some arguments to control the type of questions generated such as probability of aggregations, number of where clauses etc. (contd.)
            # These arguments can optionally be provided by the user as inference arguments.
            # Check out the notebook at primeqa/notebooks/qg/tableqginference.ipynb for more details.

            # aggregation proobablities
            agg_prob = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            if inference_args.generate_aggregate:
                agg_prob = [0.0, 0.2, 0.2, 0.2, 0.2, 0.2]

            # where clauses
            where_prob = [0.0] * 5
            for i in range(1, 5):
                if i <= inference_args.max_where_clauses:
                    where_prob[i] = 1.0
            where_prob = [w / sum(where_prob) for w in where_prob]

            with open(inference_args.data_path) as fp:
                data_list = json.load(fp)

            generated_questions = qg_model.generate_questions(
                data_list,
                inference_args.num_questions_per_instance,
                agg_prob,
                where_prob,
            )
            with open(inference_args.gen_output_path, "w") as fp:
                json.dump(generated_questions, fp)

    # Evaluation
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(metrics.keys()):
                logger.info("  %s = %s", key, str(metrics[key]))
                writer.write("%s = %s\n" % (key, str(metrics[key])))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main(sys.argv)
