"""This script allows to train & evaluate a data generation model or apply Active Learning in combination with an RC model if appropriate."""

import dataclasses
import logging
import math
import os
import sys
import traceback
from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass, field
from importlib import import_module
from operator import attrgetter, itemgetter
from typing import Any, Callable, List, Optional, Tuple

from examples.datagen_with_al.utils.data import expand_answers, get_datasets
from primeqa.al.models.al import ActiveLearner, GenALScorer, TrainerConfig
from primeqa.boolqa.processors.postprocessors.extractive import (
    ExtractivePipelinePostProcessor,
)
from primeqa.mrc.data_models.eval_prediction_with_processing import (
    EvalPredictionWithProcessing,
)
from primeqa.mrc.metrics.mlqa.mlqa import MLQA
from primeqa.mrc.metrics.nq_f1.nq_f1 import NQF1
from primeqa.mrc.metrics.squad.squad import SQUAD
from primeqa.mrc.metrics.tydi_f1.tydi_f1 import TyDiF1
from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from primeqa.mrc.models.task_model import ModelForDownstreamTasks
from primeqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from primeqa.mrc.processors.postprocessors.natural_questions import (
    NaturalQuestionsPostProcessor,
)
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from primeqa.mrc.processors.postprocessors.squad import SQUADPostProcessor
from primeqa.mrc.processors.preprocessors.natural_questions import (
    NaturalQuestionsPreProcessor,
)
from primeqa.mrc.processors.preprocessors.squad import SQUADPreprocessor
from primeqa.mrc.processors.preprocessors.tydiqa import TyDiQAPreprocessor
from primeqa.mrc.processors.preprocessors.tydiqa_google import TyDiQAGooglePreprocessor
from primeqa.mrc.trainers.mrc import MRCTrainer
from primeqa.qg.metrics.generation_metrics import rouge_metrics
from primeqa.qg.models.qg_model import QGModel
from primeqa.qg.processors.data_loader import QGDataLoader
from primeqa.qg.trainers.qg_trainer import GenTrainer
from primeqa.qg.utils.data import (
    dicts_to_feature_dict,
    prepare_labelled_data,
    select_unique,
)
from primeqa.qg.utils.data_collator import DataCollatorForSeq2SeqWithDecoderInputs
from transformers import (
    AutoConfig,
    AutoTokenizer,
    DataCollatorWithPadding,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import datasets
from datasets import Dataset, load_dataset

logger = logging.getLogger()

# unique IDs for the trainer, also used for metric record
GEN_TRAINER_ID = "gen_trainer"
RC_TRAINER_ID = "rc_trainer"


def object_reference(reference_as_str: str) -> object:
    """
    Given a fully qualified path to a class reference, return a pointer to the reference.
    This will work with types, functions, methods, and other objects (e.g. dict).

    Args:
        reference_as_str: the fully qualified path (expects the fully qualified path in dot notation,
                          e.g. primeqa.mrc.processors.postprocessors.extractive.ExtractivePostProcessor).

    Returns:
        reference to path given by input

    Raises:
        TypeError: Unable to resolve input path
    """

    def _split_into_class_and_module_name(class_path):
        modules = class_path.split(".")
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
        raise TypeError(
            f"Unable to resolve the string {reference_as_str} to a fully qualified class path"
        ) from ex


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models",
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_dataset: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "Name of the dataset to train the qg model",
        },
    )
    eval_dataset: Optional[str] = field(
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
        default=False,
        metadata={
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
    doc_stride: int = field(
        default=128,
        metadata={
            "help": "When splitting up a long document into chunks, how much stride to take between chunks."
        },
    )
    n_best_size: int = field(
        default=20,
        metadata={
            "help": "The total number of n-best predictions to generate when looking for an answer."
        },
    )
    n_best_logits: int = field(
        default=20,
        metadata={
            "help": "The number of logits to consider when searching for start and end position of an answer"
        },
    )
    max_answer_length: int = field(
        default=32,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )
    num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "The number of workers for processing datasets"},
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do we want to store the pretrained models downloaded from s3"
        },
    )


@dataclass
class ALArguments:
    """
    Arguments for performing Active Learning.
    """

    do_al: Optional[bool] = field(
        default=False, metadata={"help": "Whether to perform Active Learning"}
    )
    output_dir: Optional[str] = field(
        default=None, metadata={"help": "The output directory for AL"}
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
        metadata={"help": "the dataset(s) for which the contexts are excluded"},
    )
    max_gen_length: Optional[int] = field(
        default=None, metadata={"help": "The maximum number of tokens generated"}
    )
    skip_context_length_above: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum context length in tokens (contexts with more tokens will be discarded)"
        },
    )
    skip_context_length_below: Optional[int] = field(
        default=100,
        metadata={
            "help": "The minimum context length in tokens (contexts with less tokens will be discarded); default 100 - 0 disables it"
        },
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


@dataclass
class TaskArguments:
    """
    Task specific arguments.
    """

    scorer_type: str = field(
        default="weighted_sum_target_type_and_score_diff",
        metadata={
            "help": "The name of the scorer to compute answer score.",
            "choices": SupportedSpanScorers.get_supported(),
        },
    )
    preprocessor: object_reference = field(
        default=SQUADPreprocessor,
        metadata={
            "help": "The name of the preprocessor to use.",
            "choices": [
                TyDiQAPreprocessor,
                SQUADPreprocessor,
                TyDiQAGooglePreprocessor,
                NaturalQuestionsPreProcessor,
            ],
        },
    )
    postprocessor: object_reference = field(
        default=ExtractivePostProcessor,
        metadata={
            "help": "The name of the postprocessor to use.",
            "choices": [
                ExtractivePostProcessor,
                ExtractivePipelinePostProcessor,
                SQUADPostProcessor,
                NaturalQuestionsPostProcessor,
            ],
        },
    )
    eval_metrics: str = field(
        default="SQUAD",
        metadata={
            "help": "The name of the evaluation metric function implemented in primeqa (e.g. TyDiF1).",
            "choices": ["TyDiF1", "SQUAD", "MLQA", "NQF1"],
        },
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Prints logging info if true (including evaluation output)"},
    )


class TrainingArgumentsMetaClass(type):
    def __call__(cls, *args: Any, **kwds: Any) -> Any:
        if not hasattr(cls, "prefix"):
            raise AttributeError(
                f"Class {cls} does not have attribute `prefix`. Make sure to create class with attribute `prefix`, e.g. TrainingArgumentsMetaClass('class_name', (Seq2SeqTrainingArguments,), dict(prefix='prefix'))"
            )
        prefix = cls.prefix
        return super().__call__(
            *args,
            **{
                key[3:] if key.startswith(prefix) else key: value
                for key, value in kwds.items()
            },
        )

    def __getattribute__(cls, __name: str) -> Any:
        # add prefix to field names, note that this only modifies the class fields, not the instance fields which stay original
        if __name == dataclasses._FIELDS:
            fields = super().__getattribute__(__name)
            new_fields = {}
            for key, value in fields.items():
                value: dataclasses.Field
                new_value = copy(value)
                new_value.name = cls.prefix + new_value.name
                new_fields[cls.prefix + key] = new_value
            return new_fields
        return super().__getattribute__(__name)


RCSeq2SeqTrainingArguments = TrainingArgumentsMetaClass(
    "RCSeq2SeqTrainingArguments", (Seq2SeqTrainingArguments,), dict(prefix="rc_")
)
RCModelArguments = TrainingArgumentsMetaClass(
    "RCModelArguments", (ModelArguments,), dict(prefix="rc_")
)
QGSeq2SeqTrainingArguments = TrainingArgumentsMetaClass(
    "QGSeq2SeqTrainingArguments", (Seq2SeqTrainingArguments,), dict(prefix="qg_")
)
QGModelArguments = TrainingArgumentsMetaClass(
    "QGModelArguments", (ModelArguments,), dict(prefix="qg_")
)
RCTaskArguments = TrainingArgumentsMetaClass(
    "RCTaskArguments", (TaskArguments,), dict(prefix="rc_")
)


def main(raw_args):
    parser = HfArgumentParser(
        (
            RCModelArguments,
            QGModelArguments,
            RCSeq2SeqTrainingArguments,
            QGSeq2SeqTrainingArguments,
            RCTaskArguments,
            DataTrainingArguments,
            ALArguments,
            InferenceArguments,
        )
    )

    # type annotations
    rc_model_args: ModelArguments
    qg_model_args: ModelArguments
    rc_training_args: Seq2SeqTrainingArguments
    qg_training_args: Seq2SeqTrainingArguments
    rc_task_args: TaskArguments
    data_args: DataTrainingArguments
    al_args: ALArguments
    inference_args: InferenceArguments

    if len(raw_args) == 2 and raw_args[1].endswith(".json"):
        (
            rc_model_args,
            qg_model_args,
            rc_training_args,
            qg_training_args,
            rc_task_args,
            data_args,
            al_args,
            inference_args,
        ) = parser.parse_json_file(json_file=raw_args[1])
    elif len(raw_args) == 1:
        (
            rc_model_args,
            qg_model_args,
            rc_training_args,
            qg_training_args,
            rc_task_args,
            data_args,
            al_args,
            inference_args,
        ) = parser.parse_dict(raw_args[0])
    else:
        (
            rc_model_args,
            qg_model_args,
            rc_training_args,
            qg_training_args,
            rc_task_args,
            data_args,
            al_args,
            inference_args,
        ) = parser.parse_args_into_dataclasses()

    # check for conditions
    if inference_args.do_generate:
        if inference_args.predict_dataset is None:
            raise ValueError(
                "Predict dataset cannot be None, please specify one usin `--predict_dataset`."
            )

    # some arguments have to be hardcoded in order for HF Trainer to work
    qg_training_args.predict_with_generate = True
    qg_training_args.prediction_loss_only = False

    # Setup logging
    logging.basicConfig(
        format="[%(asctime)s - %(levelname)s - %(name)s] %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
        if any(
            (
                rc_training_args.local_rank in [-1, 0],
                qg_training_args.local_rank in [-1, 0],
            )
        )
        else logging.WARN,
    )
    # logger.warning(
    #     "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
    #     training_args.local_rank,
    #     training_args.device,
    #     training_args.n_gpu,
    #     bool(training_args.local_rank != -1),
    #     training_args.fp16,
    # )
    logger.info("RC Training parameters: %s", rc_training_args.to_dict())
    logger.info("QG Training parameters: %s", qg_training_args.to_dict())

    # Set seed
    set_seed(rc_training_args.seed)
    set_seed(qg_training_args.seed)

    # load datasets
    if al_args.do_al or rc_training_args.do_train or qg_training_args.do_train:
        train_dataset = get_datasets(data_args.train_dataset, concatenate=True)
        train_dataset = expand_answers(train_dataset, separate_answers=False)
    if rc_training_args.do_eval or qg_training_args.do_eval:
        validation_dataset = get_datasets(data_args.eval_dataset)
        validation_dataset = expand_answers(validation_dataset, separate_answers=False)

    def preprocess_fn_wrapper(
        training_args: TrainingArguments,
        process_fn: Callable,
        add_examples_to_output: bool = False,
    ) -> Callable[[Dataset], Tuple[Dataset, Dataset]]:
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

    ## RC setup

    if rc_model_args.model_name_or_path is not None:
        # detecting last checkpoint
        last_checkpoint = None
        if (
            os.path.isdir(rc_training_args.output_dir)
            and rc_training_args.do_train
            and not rc_training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(rc_training_args.output_dir)
            if (
                last_checkpoint is None
                and len(os.listdir(rc_training_args.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({rc_training_args.output_dir}) already exists and is not empty. "
                    "Use --rc_overwrite_output_dir to overcome."
                )
            elif (
                last_checkpoint is not None
                and rc_training_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--rc_output_dir` or add `--rc_overwrite_output_dir` to train from scratch."
                )

        task_heads = EXTRACTIVE_HEAD
        rc_config = AutoConfig.from_pretrained(
            rc_model_args.config_name
            if rc_model_args.config_name
            else rc_model_args.model_name_or_path,
            cache_dir=data_args.cache_dir,
        )
        rc_tokenizer = AutoTokenizer.from_pretrained(
            rc_model_args.tokenizer_name
            if rc_model_args.tokenizer_name
            else rc_model_args.model_name_or_path,
            cache_dir=data_args.cache_dir,
            use_fast=True,
            config=rc_config,
        )

        rc_config.sep_token_id = rc_tokenizer.convert_tokens_to_ids(
            rc_tokenizer.sep_token
        )
        rc_model = ModelForDownstreamTasks.from_config(
            rc_config,
            rc_model_args.model_name_or_path,
            task_heads=task_heads,
            cache_dir=data_args.cache_dir,
        )
        rc_model.set_task_head("qa_head")

        # load preprocessor
        rc_preprocessor_class = rc_task_args.preprocessor
        rc_preprocessor = rc_preprocessor_class(
            stride=data_args.doc_stride,
            tokenizer=rc_tokenizer,
            max_seq_len=data_args.max_seq_length,
            num_workers=data_args.num_workers,
            max_q_char_len=data_args.max_q_char_len,
            single_context_multiple_passages=data_args.single_context_multiple_passages,
            max_contexts=data_args.max_contexts,
        )

        # process eval data
        rc_eval_examples, rc_eval_dataset = (
            preprocess_fn_wrapper(rc_training_args, rc_preprocessor.process_eval)(
                validation_dataset
            )
            if rc_training_args.do_eval
            else None
        )

        # set up metric
        # If using mixed precision we pad for efficient hardware acceleration
        using_mixed_precision = any(attrgetter("fp16", "bf16")(rc_training_args))
        rc_data_collator = DataCollatorWithPadding(
            rc_tokenizer, pad_to_multiple_of=64 if using_mixed_precision else None
        )

        rc_postprocessor_class = rc_task_args.postprocessor
        # noinspection PyProtectedMember
        rc_postprocessor = rc_postprocessor_class(
            k=data_args.n_best_logits,
            n_best_size=data_args.n_best_size,
            max_answer_length=data_args.max_answer_length,
            scorer_type=SupportedSpanScorers("weighted_sum_target_type_and_score_diff"),
            single_context_multiple_passages=rc_preprocessor._single_context_multiple_passages,
        )

        eval_metrics = getattr(sys.modules[__name__], rc_task_args.eval_metrics)()

        def rc_compute_metrics(p: EvalPredictionWithProcessing):
            return eval_metrics.compute(
                predictions=p.processed_predictions,
                references=p.label_ids,
                dataset_config_name=rc_eval_dataset.config_name,
            )

        rc_trainer_class = MRCTrainer
        rc_trainer_kwargs = dict(
            model=rc_model,
            args=rc_training_args,
            eval_dataset=rc_eval_dataset if rc_training_args.do_eval else None,
            eval_examples=rc_eval_examples if rc_training_args.do_eval else None,
            tokenizer=rc_tokenizer,
            data_collator=rc_data_collator,
            post_process_function=rc_postprocessor.process_references_and_predictions,
            compute_metrics=rc_compute_metrics,
        )

    ## QG setup

    if qg_model_args.model_name_or_path is not None:
        # detecting last checkpoint
        last_checkpoint = None
        if (
            os.path.isdir(qg_training_args.output_dir)
            and qg_training_args.do_train
            and not qg_training_args.overwrite_output_dir
        ):
            last_checkpoint = get_last_checkpoint(qg_training_args.output_dir)
            if (
                last_checkpoint is None
                and len(os.listdir(qg_training_args.output_dir)) > 0
            ):
                raise ValueError(
                    f"Output directory ({qg_training_args.output_dir}) already exists and is not empty. "
                    "Use --qg_overwrite_output_dir to overcome."
                )
            elif (
                last_checkpoint is not None
                and qg_training_args.resume_from_checkpoint is None
            ):
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--qg_output_dir` or add `--qg_overwrite_output_dir` to train from scratch."
                )

        qg_model = QGModel(qg_model_args.model_name_or_path, modality="passage_qa2s")

        qgdl = QGDataLoader(
            tokenizer=qg_model.tokenizer,
            modality="passage_qa2s",
            input_max_len=None,
            target_max_len=data_args.target_max_len,
        )

        # process eval data
        qg_eval_dataset = (
            qgdl.create(validation_dataset) if qg_training_args.do_eval else None
        )

        # set up metric
        qg_compute_metrics = rouge_metrics(qg_model.tokenizer)

        gen_trainer_class = GenTrainer
        gen_trainer_kwargs = dict(
            max_gen_length=30,
            model=qg_model.model,
            tokenizer=qg_model.tokenizer,
            args=qg_training_args,
            eval_dataset=qg_eval_dataset,
            data_collator=DataCollatorForSeq2SeqWithDecoderInputs(qg_model.tokenizer),
            compute_metrics=qg_compute_metrics,
        )

    ## training, evaluation and AL

    # for evaluation and training we need to create the trainer first
    # for training we also need to preprocess training data
    if qg_training_args.do_eval or qg_training_args.do_train:
        qg_trainer = gen_trainer_class(
            **gen_trainer_kwargs,
            train_dataset=qgdl.create(train_dataset)
            if qg_training_args.do_train
            else None,
        )
    if rc_training_args.do_eval or rc_training_args.do_train:
        rc_trainer = rc_trainer_class(
            **rc_trainer_kwargs,
            train_dataset=preprocess_fn_wrapper(
                rc_training_args, rc_preprocessor.process_train
            )(train_dataset)[1]
            if rc_training_args.do_train
            else None,
        )

    # evaluation
    if qg_training_args.do_eval:
        metrics = qg_trainer.evaluate()
        qg_trainer.log_metrics("eval", metrics)
    if rc_training_args.do_eval:
        metrics = rc_trainer.evaluate()
        rc_trainer.log_metrics("eval", metrics)

    # training
    if qg_training_args.do_train:
        train_result = qg_trainer.train()
        metrics = train_result.metrics
        qg_trainer.log_metrics("train", metrics)
        qg_trainer.save_metrics("train", metrics)
        qg_trainer.save_state()
    if rc_training_args.do_train:
        train_result = rc_trainer.train()
        metrics = train_result.metrics
        rc_trainer.log_metrics("train", metrics)
        rc_trainer.save_metrics("train", metrics)
        rc_trainer.save_state()

    # AL
    if al_args.do_al:
        # the ActiveLearner gets passed trainer configs which contain the trainer class and the trainer kwargs

        gen_trainer_config = TrainerConfig(
            GEN_TRAINER_ID,
            gen_trainer_class,
            gen_trainer_kwargs,
            preprocess_fn_wrapper(
                qg_training_args, qgdl.create, add_examples_to_output=True
            ),
            preprocess_fn_wrapper(
                qg_training_args, qgdl.create, add_examples_to_output=True
            ),
            None,
        )
        rc_trainer_config = TrainerConfig(
            RC_TRAINER_ID,
            rc_trainer_class,
            rc_trainer_kwargs,
            preprocess_fn_wrapper(
                rc_training_args,
                rc_preprocessor.process_train,
                add_examples_to_output=True,
            ),
            preprocess_fn_wrapper(
                rc_training_args,
                rc_preprocessor.process_train,
                add_examples_to_output=False,
            ),
            None,
        )

        al = ActiveLearner(al_args.output_dir, (gen_trainer_config, rc_trainer_config))
        # TODO choose strategy via command line arguments
        # strategy = RCALScorer(rc_trainer_id=0)
        # a special token id map is needed to generate sequences - these are valid for the QA2S model
        special_token_id_map = dict(
            bos_token_id=qg_model.tokenizer.convert_tokens_to_ids("<q>"),
            eos_token_id=qg_model.tokenizer.convert_tokens_to_ids("</q>"),
            bos_token_id_2=qg_model.tokenizer.convert_tokens_to_ids("<a>"),
            eos_token_id_2=qg_model.tokenizer.convert_tokens_to_ids("</a>"),
        )
        strategy = GenALScorer(
            strategy=GenALScorer.Strategy.SENTENCE_PROBABILITY_DROPOUT,
            max_gen_length=30,
            special_token_id_map=special_token_id_map,
            gen_trainer_id=GEN_TRAINER_ID,
        )

        # perform training via active learning
        # TODO allow to set arguments via command line
        logger.info("***** Running Active Learning *****")
        metrics = al.run(
            examples=train_dataset,
            strategy=strategy,
            num_iterations=4,
            num_samples_per_iteration=50,
            feature_id_column="id",
        )
        if metrics:
            logger.info("***** Active Learning results *****")
            for trainer_id in sorted(metrics.keys()):
                logger.info(" Trainer id = %s", trainer_id)
                if isinstance(next(iter(metrics[trainer_id].values())), Mapping):
                    for iteration in sorted(metrics[trainer_id].keys()):
                        logger.info("  Iteration = %s", iteration)
                        for metric, value in sorted(
                            metrics[trainer_id][iteration].items(), key=itemgetter(0)
                        ):
                            logger.info("   %s = %s", metric, value)
                else:
                    for metric, value in sorted(
                        metrics[trainer_id].items(), key=itemgetter(0)
                    ):
                        logger.info("   %s = %s", metric, value)
        logger.info("***** Finished Active Learning *****")

    # inference
    if inference_args.do_generate:
        # get dataset
        dataset = get_datasets(inference_args.predict_dataset)

        # same setup as Shakeri et al.: Only contexts with >= 100 tokens, contexts truncated to 550 tokens, 100000 contexts randomly drawn

        dataset = select_unique(dataset, "context")

        if inference_args.skip_context_length_above:
            logging.info(f"Skipping contexts > {inference_args.skip_context_length}")
            try:
                dataset = dataset.filter(
                    lambda x: len(
                        qg_model.tokenizer.tokenize(
                            x["context"], add_special_tokens=False
                        )
                    )
                    <= inference_args.skip_context_length,
                    num_proc=data_args.num_workers,
                )
            except IndexError:
                logging.info(
                    "No data left after filtering for context length, exiting."
                )
                exit()

        if inference_args.exclude_dataset:
            # exclude contexts for generation
            logging.info(f"Excluding contexts from specified data")
            exclude_dataset = load_dataset(
                "datasets/shared-task", name=inference_args.exclude_dataset
            )
            exclude_contexts = exclude_dataset.flatten_indices().unique("context")
            dataset = dataset.filter(
                lambda x: x["context"] not in exclude_contexts,
                num_proc=data_args.num_workers,
            )

        if inference_args.skip_context_length_below:
            logging.info(
                f"Discarding documents with less than {inference_args.skip_context_length_below} tokens"
            )
            dataset = dataset.filter(
                lambda x: inference_args.skip_context_length_below
                <= len(
                    qg_model.tokenizer.tokenize(x["context"], add_special_tokens=False)
                ),
                num_proc=data_args.num_workers,
            )

        num_samples = min(100000, len(dataset))
        logging.info(
            f"Randomly selecting {num_samples} documents (from {len(dataset)} available documents)"
        )
        # shuffle data using seed to make sure that we have always the same documents
        dataset = dataset.shuffle(seed=42)
        dataset = dataset.select(range(num_samples))
        logging.info(f"Truncating documents to {550} tokens")
        # NOTE somehow this part doesn't work with multiprocessing
        dataset = dataset.map(
            lambda x: {
                "context": qg_model.tokenizer.convert_tokens_to_string(
                    qg_model.tokenizer.tokenize(x["context"], add_special_tokens=False)[
                        :550
                    ]
                )
            },
            num_proc=data_args.num_workers,
        )

        # create trainer
        gen_trainer = GenTrainer(
            model=qg_model.model,
            args=qg_training_args,
            tokenizer=qg_model.tokenizer,
            data_collator=DataCollatorForSeq2SeqWithDecoderInputs(qg_model.tokenizer),
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
                    inference_args.gen_output_path
                    if inference_args.num_shards == 1
                    else os.path.join(inference_args.gen_output_path, str(i))
                )
                logger.info(
                    f"Saved {len(shard)} rows to {inference_args.gen_output_path}"
                )


if __name__ == "__main__":
    main(sys.argv)
