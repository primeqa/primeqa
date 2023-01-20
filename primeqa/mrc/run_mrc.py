import logging
import os
import sys
import json
import torch
import gc
import glob
from dataclasses import dataclass, field
from operator import attrgetter
from typing import Optional, Type, List

from torch.utils.data import ConcatDataset

import datasets
from datasets import concatenate_datasets

import apache_beam as beam
from transformers import HfArgumentParser, Seq2SeqTrainingArguments, DataCollatorWithPadding, AutoConfig, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint, set_seed

from primeqa.mrc.data_models.eval_prediction_with_processing import EvalPredictionWithProcessing
from primeqa.mrc.metrics.tydi_f1.tydi_f1 import TyDiF1
from primeqa.mrc.metrics.mlqa.mlqa import MLQA
from primeqa.mrc.metrics.squad.squad import SQUAD
from primeqa.mrc.metrics.nq_f1.nq_f1 import NQF1
from primeqa.mrc.metrics.rouge.rouge import ROUGE
from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD, EXTRACTIVE_WITH_CONFIDENCE_HEAD
from primeqa.mrc.models.heads.generative import FID_HEAD
from primeqa.mrc.models.task_model import ModelForDownstreamTasks
from primeqa.mrc.models.fid_task_model import FiDModelForDownstreamTasks
from primeqa.mrc.processors.postprocessors.extractive import ExtractivePostProcessor
from primeqa.text_classification.processors.postprocessors.extractive import ExtractivePipelinePostProcessor
from primeqa.mrc.processors.postprocessors.eli5_fid import ELI5FiDPostProcessor
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from primeqa.mrc.processors.preprocessors.tydiqa import TyDiQAPreprocessor
from primeqa.mrc.processors.preprocessors.squad import SQUADPreprocessor
from primeqa.mrc.processors.preprocessors.base import BasePreProcessor
from primeqa.mrc.processors.postprocessors.squad import SQUADPostProcessor
from primeqa.mrc.processors.preprocessors.natural_questions import NaturalQuestionsPreProcessor
from primeqa.mrc.processors.postprocessors.natural_questions import NaturalQuestionsPostProcessor
from primeqa.mrc.processors.preprocessors.tydiqa_google import TyDiQAGooglePreprocessor
from primeqa.mrc.processors.preprocessors.eli5_fid import ELI5FiDPreprocessor
from primeqa.mrc.data_models.data_collator import FiDDataCollator
from primeqa.mrc.processors.preprocessors.tydiboolqa_bpes import TyDiBoolQAPreprocessor
from primeqa.mrc.processors.preprocessors.mrqa import MRQAPreprocessor
from primeqa.mrc.trainers.mrc import MRCTrainer
from primeqa.mrc.trainers.mrc_mskd import MSKD_MRCTrainer
from primeqa.text_classification.run_nway_classifier import main as cls_main
from primeqa.mrc.trainers.seq2seq_mrc import MRCSeq2SeqTrainer
from primeqa.boolqa.run_score_normalizer import main as sn_main
from run_mrc_utils import object_reference, get_raw_datasets, process_raw_datasets
from primeqa.tableqa.run_tableqa import run_table_qa




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
    confidence_model_path: str = field(
        default=None,
        metadata={"help": "Path to the confidence calibration model"}
    )

@dataclass
class DistillationArguments:
    """
    Arguments pertaining to knowledge distillation.
    """
    kd_teacher_model_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained teacher model for knowledge distillation"}
    )
    kd_teacher_config_path: str = field(
        default=None,
        metadata={"help": "Path to pretrained teacher config for knowledge distillation"}
    )
    kd_temperature: float = field(
        default=1.,
        metadata={"help": "Temperature for knowledge distillation"}
    )

# modified from
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    tableqa_config_file: Optional[str] = field(
        default=None, metadata={"help": "TableQA additional arguments"}
    )
    dataset_name: str = field(
        default="tydiqa", metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "local file(s) to train on."}
    )
    eval_file: Optional[str] = field(
        default=None, metadata={"help": "local file(s) to test on."}
    )
    train_fof: Optional[str] = field(
        default=None, metadata={"help": "file of local file(s) to train on, for multi-dataset training"}
    )
    eval_fof: Optional[str] = field(
        default=None, metadata={"help": "file of local file(s) to test on, for multi-dataset evaluation"}
    )    
    data_file_format: str = field(
        default="json", metadata={"help": "the format of the local dataset files (json, jsonl, csv, text, pandas)"}
    )
    dataset_config_name: str = field(
        default="primary_task", metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        }
    )
    dataset_filter_column_name: str = field(
        default=None, metadata={
            "help": "Dataset column name to filter on, e.g. 'subset'"
        }
    )
    dataset_filter_column_values: List[str] = field(
        default=None, metadata={
            "help": "Dataset column values to match when filtering e.g. 'SQuAD HotpotQA'"
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
    beam_runner: str = field(
        default=None,
        metadata={"help": "The beam runner for loading large dataset.",
                  "choices": ['DirectRunner'],
                  }
    )


@dataclass
class TaskArguments:
    """
    Task specific arguments.
    """
    modality: str = field(
        default='text',
        metadata={"help": "whether modality is table or text",
        "choices": ["text", "table"]
                  }
    )

    scorer_type: str = field(
        default='weighted_sum_target_type_and_score_diff',
        metadata={"help": "The name of the scorer to compute answer score.",
                  "choices": SupportedSpanScorers.get_supported()
                  }
    )
    task_heads: object_reference = field(
        default=None,
        metadata={"help": "The name of the task head to use.",
                  "choices": [EXTRACTIVE_HEAD, EXTRACTIVE_WITH_CONFIDENCE_HEAD, FID_HEAD]
                  }
    )
    task_model: object_reference = field(
        default=ModelForDownstreamTasks,
        metadata={"help": "The name of the preprocessor to use.",
                  "choices": [ModelForDownstreamTasks, FiDModelForDownstreamTasks]
                  }
    )
    task_data_collator: object_reference = field(
        default=DataCollatorWithPadding,
        metadata={"help": "The name of the preprocessor to use.",
                  "choices": [DataCollatorWithPadding, FiDDataCollator]
                  }
    )
    task_trainer: object_reference = field(
        default=MRCTrainer,
        metadata={"help": "The name of the preprocessor to use.",
                  "choices": [MRCTrainer, MRCSeq2SeqTrainer, MSKD_MRCTrainer]
                  }
    )
    preprocessor: object_reference = field(
        default=TyDiQAPreprocessor,
        metadata={"help": "The name of the preprocessor to use.",
                  "choices": [MRQAPreprocessor, BasePreProcessor,TyDiQAPreprocessor,SQUADPreprocessor,TyDiQAGooglePreprocessor,NaturalQuestionsPreProcessor,ELI5FiDPreprocessor,TyDiBoolQAPreprocessor]
                }
    )
    postprocessor: object_reference = field(
        default=ExtractivePostProcessor,
        metadata={"help": "The name of the postprocessor to use.",
                  "choices": [ExtractivePostProcessor,ExtractivePipelinePostProcessor,SQUADPostProcessor, NaturalQuestionsPostProcessor, ELI5FiDPostProcessor]
                }
    )
    eval_metrics: str = field(
        default="TyDiF1",
        metadata={"help": "The name of the evaluation metric function implemented in primeqa (e.g. TyDiF1).",
                  "choices": ["TyDiF1","SQUAD","MLQA","NQF1","ROUGE"]
                 }
    )
    do_boolean: bool = field(
        default=False, metadata={"help": "Enable processing of boolean questions.  If activated,"
                                        "--do_eval will be forced also, and --postprocessor will be "
                                        "defaulted to ExtractivePipelinePostProcessor unless overridden"
                                        "by a postprocessor that subclasses ExtractivePipelinePostProcessor"}
    )
    boolean_config: str = field(
        default=None, metadata={"help": "The configuration name file for the boolean task in json format"}
    )    
    passage_non_null_threshold: int = field(
        default=2,
        metadata={"help": "The passage level non-null threshold (number of annotators to indicate no answer). This should be set to 1 if there is only one annotation"}
    )
    span_non_null_threshold: int = field(
        default=2,
        metadata={"help": "The span level non-null threshold (number of annotators to indicate no answer). This should be set to 1 if there is only one annotation"}
    )
    verbose: bool = field(
        default=False,
        metadata={"help": "Prints logging info if true (including evaluation output)"}
    )
    output_dropout_rate: float = field(
        default=0.25,
        metadata={"help": "The dropout probability applied to LM output in "
                          "order to generate confidence calibration features."
                  },
    )
    decoding_times_with_dropout: int = field(
        default=5,
        metadata={"help": "The number of decoding times to generate confidence "
                          "calibration features with dropout."
                  },
    )

    def __post_init__(self):
        if not self.task_heads:
            self.task_heads = EXTRACTIVE_HEAD  # cannot directly set mutable value as default


def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, TaskArguments, DistillationArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, task_args, kd_args = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, task_args, kd_args = parser.parse_args_into_dataclasses()

    if training_args.do_train and data_args.train_fof is not None:
        # add knowledge distillation arguments to training_args        
        kd_args.task_heads = task_args.task_heads
        for k in kd_args.__dict__:
            setattr(training_args, k, getattr(kd_args, k))
            
    # if we are doing the boolean post-processing, require do_eval, because the id's (not included in HF
    # dataset) might have changed
    # we require ExtractivePipelinePostProcessor to populate certain fields for the boolqa classifiers,
    # so force it here - this can't be done in a __post_init__ postprocess is in TaskArguments and
    # do_eval is in TrainingArguments
    if task_args.do_boolean:
        training_args.do_eval = True
        if not isinstance(task_args.postprocessor, ExtractivePipelinePostProcessor):
            task_args.postprocessor = ExtractivePipelinePostProcessor
            
    logger = logging.getLogger(__name__)
    if task_args.verbose:
        logging.basicConfig(level = logging.INFO)
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

    # Run Table Question Answering        
    if task_args.modality=="table":
        run_table_qa(data_args,model_args,training_args)
        sys.exit(0)

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

    config.sep_token_id = tokenizer.convert_tokens_to_ids(tokenizer.sep_token)
    config.output_dropout_rate = task_args.output_dropout_rate
    config.decoding_times_with_dropout = task_args.decoding_times_with_dropout

    model_class = task_args.task_model
    model = model_class.from_config(
        config,
        model_args.model_name_or_path,
        task_heads=task_heads,
        cache_dir=model_args.cache_dir,
    )
    model.set_task_head(next(iter(task_heads)))

    # load data
    if data_args.train_fof is not None or data_args.eval_fof is not None:
        logger.info('Loading datasets')
        raw_datasets = {}
        if data_args.train_fof is not None:
            raw_train_datasets, train_preprocessors = get_raw_datasets(data_args.train_fof, data_args,
                                                                       task_args, model_args.cache_dir,
                                                                       split='train')
            raw_datasets['train'] = raw_train_datasets
        if data_args.eval_fof is not None:            
            raw_validation_datasets, validation_preprocessors = get_raw_datasets(data_args.eval_fof, data_args,
                                                                                 task_args, model_args.cache_dir,
                                                                                 split='validation')
            raw_datasets['validation'] = raw_validation_datasets
    else:
        logger.info('Loading dataset')        
        if data_args.train_file is not None or data_args.eval_file is not None:
            data_files = {}
            raw_datasets = {}
            # Load train and validation datasets separately because they might have different columns
            if data_args.train_file is not None: 
                data_files['train'] = glob.glob(data_args.train_file)
                raw_datasets["train"] = datasets.load_dataset(
                    data_args.data_file_format, 
                    data_files={"train": data_files["train"]}, 
                    split="train",
                    cache_dir=model_args.cache_dir
                 )
            if data_args.eval_file is not None: 
                data_files['validation'] = glob.glob(data_args.eval_file)
                raw_datasets["validation"] = datasets.load_dataset(
                    data_args.data_file_format, 
                    data_files={"validation": data_files["validation"]}, 
                    split="validation",
                    cache_dir=model_args.cache_dir
                 )        
        else:
            if data_args.dataset_name == "natural_questions":
                raw_datasets = datasets.load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    cache_dir=model_args.cache_dir,
                    beam_runner=data_args.beam_runner,
                    revision="main"
                )
            else: 
                raw_datasets = datasets.load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    cache_dir=model_args.cache_dir
                )

    # load preprocessor
    if not data_args.train_fof:
        train_preprocessors = [task_args.preprocessor]
    if not data_args.eval_fof:
        validation_preprocessors = [task_args.preprocessor]
    for i, p in enumerate(train_preprocessors):
        if isinstance(p, str):
            p = object_reference(p)
        train_preprocessors[i] = p(
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
            max_answer_len=data_args.max_answer_length
        )
    for i, p in enumerate(validation_preprocessors):
        if isinstance(p, str):
            p = object_reference(p)
        validation_preprocessors[i] = p(
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
            max_answer_len=data_args.max_answer_length
        )

    # if filtering, check that both column name and column values are provided
    if data_args.dataset_filter_column_values is not None:
        if data_args.dataset_filter_column_name is None:
            raise ValueError(f"Filtering on --dataset_filter_column_values ({data_args.dataset_filter_column_values}) "
                      "requires --dataset_filter_column_name to be provided.")

    # process train data
    if training_args.do_train:
        train_examples = raw_datasets['train']
        if data_args.train_fof is None:
            if data_args.dataset_filter_column_values is not None:
                logger.info(f"Filter TRAIN dataset {data_args.dataset_filter_column_name} {data_args.dataset_filter_column_values}")
                train_examples = train_examples.filter(lambda example: example[data_args.dataset_filter_column_name] in (data_args.dataset_filter_column_values))
                train_examples = train_examples.shuffle(seed=training_args.seed)
                logger.info(f"Filtered TRAIN dataset size {train_examples.num_rows}")
            train_examples = [train_examples]
        # Train feature creation
        _, train_datasets = process_raw_datasets(train_examples, train_preprocessors, training_args,
                                                 split='train', max_samples=data_args.max_train_samples)
        if data_args.train_fof is not None:
            if task_args.task_trainer == MSKD_MRCTrainer:
                train_dataset = ConcatDataset(train_datasets)
            else:
                train_dataset = concatenate_datasets(train_datasets).shuffle(seed=training_args.seed)
        else:
            train_dataset = train_datasets[0]
    # process val data
    if training_args.do_eval:
        eval_examples = raw_datasets['validation']
        if data_args.eval_fof is None:
            if data_args.dataset_filter_column_values is not None:
                logger.info(f"Filter EVAL dataset {data_args.dataset_filter_column_name} {data_args.dataset_filter_column_values}")
                eval_examples = eval_examples.filter(lambda example: example[data_args.dataset_filter_column_name] in (data_args.dataset_filter_column_values))
                logger.info(f"Filtered EVAL dataset size {eval_examples.num_rows}")
            eval_examples = [eval_examples]
        # Validation feature creation
        eval_examples, eval_datasets = process_raw_datasets(eval_examples, validation_preprocessors, training_args,
                                                            split='validation', max_samples=data_args.max_eval_samples)
        if data_args.eval_fof is not None and task_args.task_trainer == MSKD_MRCTrainer:
            eval_dataset = ConcatDataset(eval_datasets)
            setattr(eval_dataset, 'config_name', getattr(eval_dataset.datasets[0], 'config_name'))
        else:
            eval_examples, eval_dataset = eval_examples[0], eval_datasets[0]        

    # If using mixed precision we pad for efficient hardware acceleration
    using_mixed_precision = any(attrgetter('fp16', 'bf16')(training_args))
    data_collator_class = task_args.task_data_collator
    data_collator = data_collator_class(tokenizer, pad_to_multiple_of=64 if using_mixed_precision else None)

    postprocessor_class = task_args.postprocessor

    # noinspection PyProtectedMember
    postprocessor = postprocessor_class(
        k=data_args.n_best_logits,
        n_best_size=data_args.n_best_size,
        max_answer_length=data_args.max_answer_length,
        scorer_type=SupportedSpanScorers(scorer_type),
        single_context_multiple_passages=train_preprocessors[0]._single_context_multiple_passages,
        confidence_model_path=model_args.confidence_model_path,
        output_confidence_feature=True if task_args.task_heads == EXTRACTIVE_WITH_CONFIDENCE_HEAD else False,
        tokenizer=tokenizer,
    )
    
    eval_metrics = getattr(sys.modules[__name__], task_args.eval_metrics)()

    def compute_metrics(p: EvalPredictionWithProcessing):
        return eval_metrics.compute(predictions=p.processed_predictions, references=p.label_ids,
            passage_non_null_threshold=task_args.passage_non_null_threshold, 
            span_non_null_threshold=task_args.span_non_null_threshold,verbose=task_args.verbose,
            dataset_config_name = eval_dataset.config_name)

    trainer_class = task_args.task_trainer
    trainer = trainer_class(
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
        if data_args.train_fof is None:
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

        if data_args.eval_fof is None:
            max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
            metrics["eval_samples"] = min(max_eval_samples, len(eval_examples))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if task_args.do_boolean:
        logger.info("Processing of boolean questions")
        if not os.path.exists(os.path.join(training_args.output_dir,"eval_predictions.json")):
            raise Exception(f"No MRC predictions were found at {training_args.output_dir}")
        with open(task_args.boolean_config, 'r') as f:
            boolean_config = json.load(f)

        boolean_config['qtc']['output_dir'] = training_args.output_dir+"/qtc"
        boolean_config['qtc']['test_file'] = training_args.output_dir + "/eval_predictions.json"
        boolean_config['qtc']['do_mrc_pipeline']='True'
        boolean_config['evc']['output_dir'] = training_args.output_dir+"/evc"
        boolean_config['evc']['test_file'] = training_args.output_dir + "/qtc/predictions.json"
        boolean_config['evc']['do_mrc_pipeline']='True'
        boolean_config['sn']['output_dir'] = training_args.output_dir+"/sn"
        boolean_config['sn']['test_file'] = training_args.output_dir + "/evc/predictions.json"

        if model: del model
        gc.collect()
        torch.cuda.empty_cache()
        logger.info(f"torch memory allocated {torch.cuda.memory_allocated()} \
            max memory {torch.cuda.max_memory_allocated()}")

        cls_main([boolean_config['qtc']])
        cls_main([boolean_config['evc']])
        sn_main([boolean_config['sn']])

        with open(os.path.join(boolean_config['sn']['output_dir'], 'eval_predictions_processed.json'), 'r') as f:
            processed_predictions = json.load(f)
            
        references = postprocessor.prepare_examples_as_references(eval_examples)
        boolean_eval_metric = eval_metrics.compute(predictions=processed_predictions, references=references)
        boolean_eval_metric["eval_samples"] = min(max_eval_samples, len(eval_examples))
        trainer.log_metrics("eval", boolean_eval_metric)
        path = os.path.join(boolean_config['sn']['output_dir'], f"all_results.json")
        with open(path, "w") as f:
            json.dump(boolean_eval_metric, f, indent=4, sort_keys=True)        


if __name__ == '__main__':
    main()
