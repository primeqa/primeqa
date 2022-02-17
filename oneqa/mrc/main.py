import logging
import os

import datasets
from transformers import TrainingArguments, DataCollatorWithPadding, AutoConfig, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint, set_seed

from oneqa.mrc.processors.default import DefaultProcessor
from oneqa.mrc.trainers.default import MRCTrainer


def main():
    logger = logging.getLogger(__name__)
    training_args = TrainingArguments(
        output_dir='/dccstor/aferritt3/oneqa/test-model',
        do_train=True,
        do_eval=True,
        num_train_epochs=1.,
        fp16=True,
    )

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

    model_name = 'xlm-roberta-base'
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
    )

    # load data
    raw_datasets = datasets.load_dataset("tydiqa", "primary_task")

    # load processor
    processor = DefaultProcessor()

    # process train data
    train_dataset = raw_datasets["train"]
    max_train_samples = 2000
    if max_train_samples is not None:  #if data_args.max_train_samples is not None:
        # We will select sample from whole data if argument is specified
        train_dataset = train_dataset.select(range(max_train_samples))
    with training_args.main_process_first(desc="train dataset map pre-processing"):
        train_dataset = train_dataset.map(
            processor._process,
            batched=True,
            num_proc=8,  #data_args.preprocessing_num_workers,
            #remove_columns=column_names,
            #load_from_cache_file=not data_args.overwrite_cache,
            load_from_cache_file=False,
            desc="Running tokenizer on train dataset",
        )

    # process val data
    eval_examples = raw_datasets["validation"]
    max_eval_samples = 500
    if max_eval_samples is not None:  # data_args.max_eval_samples is not None:
        # We will select sample from whole data
        eval_examples = eval_examples.select(range(max_eval_samples))
    # Validation Feature Creation
    with training_args.main_process_first(desc="validation dataset map pre-processing"):
        eval_dataset = eval_examples.map(
            processor._process,
            batched=True,
            num_proc=8,  #data_args.preprocessing_num_workers,
            # remove_columns=column_names,
            #load_from_cache_file=not data_args.overwrite_cache,
            load_from_cache_file=False,
            desc="Running tokenizer on validation dataset",
        )

    # process test data

    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if training_args.fp16 else None)

    # train

    trainer = MRCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=post_processing_function,
        compute_metrics=compute_metrics,
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


    logger.info("*** Evaluate ***")
    metrics = trainer.evaluate()

    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)



    # run val

    # run test


if __name__ == '__main__':
    main()