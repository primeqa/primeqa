import logging
from enum import Enum
from operator import attrgetter, itemgetter
from typing import List

from examples.datagen_with_al.utils.data import LMFilter, RTFilter, unpack_samples
from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from primeqa.mrc.models.task_model import ModelForDownstreamTasks
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from primeqa.mrc.processors.postprocessors.squad import SQUADPostProcessor
from transformers import AutoConfig, AutoTokenizer, DataCollatorWithPadding, TrainingArguments

from datasets import concatenate_datasets, load_from_disk

format_str = "[%(levelname)s - %(name)s - %(asctime)s] %(message)s"
datefmt_str = "%d-%m-%Y %H:%M:%S"
logging.basicConfig(format=format_str, datefmt=datefmt_str, level=logging.INFO)
logger = logging.getLogger()


class Strategy(Enum):
    LM = "lm"
    RT = "rt"

    def __str__(self) -> str:
        return self.value


def filter_samples(
    dataset_paths: List[str],
    output_path: str,
    strategy: Strategy,
    cache_dir: str = None,
    rt_model_path: str = None,
    num_workers: int = None,
):
    # load dataset from disk
    dataset = concatenate_datasets([load_from_disk(dataset_path) for dataset_path in dataset_paths])

    # filter dataset
    if strategy is None:
        logger.info("Filtering disabled, only unpacking samples")
        filter_fn = unpack_samples()
    elif strategy == Strategy.LM:
        logger.info("Filtering samples using LM scoring (num_keep=5)")
        filter_fn = LMFilter(num_keep=5)
    elif strategy == Strategy.RT:
        logger.info("Filtering samples using RTcons")
        task_heads = EXTRACTIVE_HEAD
        config = AutoConfig.from_pretrained(
            rt_model_path,
            cache_dir=cache_dir,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            rt_model_path,
            cache_dir=cache_dir,
            use_fast=True,
            config=config,
        )

        config.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
        model = ModelForDownstreamTasks.from_config(
            config,
            rt_model_path,
            task_heads=task_heads,
            cache_dir=cache_dir,
        )
        model.set_task_head("qa_head")

        training_args = TrainingArguments(output_dir="tmp")

        # If using mixed precision we pad for efficient hardware acceleration
        using_mixed_precision = any(attrgetter("fp16", "bf16")(training_args))
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if using_mixed_precision else None)

        # noinspection PyProtectedMember
        postprocessor = SQUADPostProcessor(
            k=1,
            n_best_size=20,
            max_answer_length=32,
            scorer_type=SupportedSpanScorers("weighted_sum_target_type_and_score_diff"),
            single_context_multiple_passages=True,
        )

        filter_fn = RTFilter(
            model=model,
            args=training_args,
            tokenizer=tokenizer,
            data_collator=data_collator,
            post_process_function=postprocessor.process_references_and_predictions,
            num_workers=num_workers,
        )

    dataset = filter_fn(dataset)

    # save dataset to disk
    dataset.save_to_disk(output_path)


if __name__ == "__main__":

    def main():
        import argparse

        parser = argparse.ArgumentParser(description="This program allows to filter synthetic data.")
        parser.add_argument("dataset_path", type=str, nargs="+", help="The dataset(s) for filtering")
        parser.add_argument(
            "output_dir",
            type=str,
            help="The directory where the filtered dataset will be saved",
        )
        parser.add_argument(
            "--filter",
            type=Strategy,
            default=Strategy.LM,
            choices=[Strategy.LM, Strategy.RT],
            help="Specifies the model architecture to use for generation",
        )
        parser.add_argument(
            "--rt_model_name_or_path",
            help="Set the transformer model for answer prediction for RT filtering",
        )
        parser.add_argument(
            "--disable_filtering",
            action="store_true",
            help="Don't filter samples and unpack them",
        )
        parser.add_argument("--cache", help="The directory used as cache")
        parser.add_argument("--num_workers", type=int, help="The number of workers for processing tasks")
        args = parser.parse_args()

        filter_samples(
            args.dataset_path,
            args.output_dir,
            None if args.disable_filtering else args.filter,
            cache_dir=args.cache,
            rt_model_path=args.rt_model_name_or_path,
            num_workers=args.num_workers,
        )

    main()
