import logging
import os
import sys
import traceback
import json
import torch
import gc
import glob
from dataclasses import dataclass, field
from importlib import import_module
from operator import attrgetter
from typing import Optional, Type

import datasets
import apache_beam as beam
from torch.utils.data import ConcatDataset
from transformers import HfArgumentParser, TrainingArguments, DataCollatorWithPadding, AutoConfig, AutoTokenizer
from transformers.trainer_utils import get_last_checkpoint, set_seed

from primeqa.mrc.data_models.eval_prediction_with_processing import EvalPredictionWithProcessing
from primeqa.mrc.metrics.squad.squad import SQUAD
from primeqa.mrc.models.heads.extractive import EXTRACTIVE_HEAD
from primeqa.mrc.models.task_model import ModelForDownstreamTasks
from primeqa.mrc.processors.postprocessors.scorers import SupportedSpanScorers
from primeqa.mrc.processors.preprocessors.squad import SQUADPreprocessor
from primeqa.mrc.processors.postprocessors.squad import SQUADPostProcessor
from trainer import MSKD_MRCTrainer

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
        modules = class_path.split('.')
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
        raise TypeError(f"Unable to resolve the string {reference_as_str} to a fully qualified class path") from ex


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
    confidence_model_path: str = field(
        default=None,
        metadata={"help": "Path to the confidence calibration model"}
    )


# modified from
# https://github.com/huggingface/transformers/blob/main/examples/pytorch/question-answering/run_qa.py
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_fof: Optional[str] = field(
        default=None, metadata={"help": "file of local file(s) to train on."}
    )
    eval_fof: Optional[str] = field(
        default=None, metadata={"help": "file of local file(s) to test on."}
    )
    data_file_format: str = field(
        default="json", metadata={"help": "the format of the local dataset files (json, csv, text, pandas)"}
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
    

@dataclass
class TaskArguments:
    """
    Task specific arguments.
    """
    modality: str = field(
        default='text',
        metadata={
            "help": "whether modality is table or text",
            "choices": ["text"]
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
                  "choices": [EXTRACTIVE_HEAD]
                  }
    )
    preprocessor: object_reference = field(
        default=SQUADPreprocessor,
        metadata={"help": "The name of the preprocessor to use.",
                  "choices": [SQUADPreprocessor]
                  }
    )
    postprocessor: object_reference = field(
        default=SQUADPostProcessor,
        metadata={"help": "The name of the postprocessor to use.",
                  "choices": [SQUADPostProcessor]
                  }
    )
    eval_metrics: str = field(
        default="SQUAD",
        metadata={"help": "The name of the evaluation metric function implemented in primeqa (e.g. TyDiF1).",
                  "choices": ["SQUAD"]
                 }
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
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, TaskArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, task_args = \
            parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, task_args = parser.parse_args_into_dataclasses()

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
    model = ModelForDownstreamTasks.from_config(
        config,
        model_args.model_name_or_path,
        task_heads=task_heads,
        cache_dir=model_args.cache_dir,
    )
    model.set_task_head(next(iter(task_heads)))

    # load data
    if data_args.train_fof is not None or data_args.eval_fof is not None:
        logger.info('Loading datasets')
        def get_raw_datasets(fof):
            data_files = []
            raw_datasets = []
            with open(fof, 'r') as infile:
                for line in infile:
                    if line.startswith('#'):
                        continue
                    filename = line.strip()
                    if not filename:
                        continue
                    data_files.append(filename)                    
                    raw_dataset = datasets.load_dataset(
                        data_args.data_file_format, 
                        data_files=filename,
                        cache_dir=model_args.cache_dir
                    )['train']
                    raw_datasets.append(raw_dataset)
            return raw_datasets, data_files
        if data_args.train_fof is not None:
            raw_train_datasets, _ = get_raw_datasets(data_args.train_fof)
        if data_args.eval_fof is not None:            
            raw_validation_datasets, validation_data_files = get_raw_datasets(data_args.eval_fof)
    
    # load preprocessor
    preprocessor_class = task_args.preprocessor
    preprocessor = preprocessor_class(
        stride=data_args.doc_stride,
        tokenizer=tokenizer,
        negative_sampling_prob_when_has_answer=data_args.negative_sampling_prob_when_has_answer,
        load_from_cache_file=not data_args.overwrite_cache,
        max_seq_len=data_args.max_seq_length,
        num_workers=data_args.preprocessing_num_workers,
        max_q_char_len=data_args.max_q_char_len,
        single_context_multiple_passages=data_args.single_context_multiple_passages,
        max_contexts=data_args.max_contexts,
    )

    if data_args.train_fof is not None or data_args.eval_fof is not None:
        # multi-dataset training and/or evaluation
        def preprocess_raw_datasets(raw_datasets, max_samples, training_args, preprocess_fn, split):
            examples, datasets = [], []
            for dataset in raw_datasets:
                if max_samples is not None:
                    # We will select sample from whole data if argument is specified
                    dataset = dataset.select(range(max_samples))
                # Feature Creation
                with training_args.main_process_first(
                        desc=f"{split} dataset map pre-processing"
                ):
                    examples_ds, dataset = preprocess_fn(dataset)
                    examples.append(examples_ds)                    
                    datasets.append(dataset)
            return examples, datasets
        if data_args.train_fof is not None:
            # process train data
            if training_args.do_train:
                _, train_datasets = preprocess_raw_datasets(
                    raw_train_datasets, data_args.max_train_samples, training_args,
                    preprocessor.process_train, 'train'
                )
                train_dataset = ConcatDataset(train_datasets)
        if data_args.eval_fof is not None:
            # process val data
            if training_args.do_eval:        
                eval_examples, eval_datasets = preprocess_raw_datasets(
                    raw_validation_datasets, data_args.max_eval_samples, training_args,
                    preprocessor.process_eval, 'validation'
                )
                eval_dataset = eval_datasets
                               
    # If using mixed precision we pad for efficient hardware acceleration
    using_mixed_precision = any(attrgetter('fp16', 'bf16')(training_args))
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=64 if using_mixed_precision else None)

    postprocessor_class = task_args.postprocessor

    # noinspection PyProtectedMember
    postprocessor = postprocessor_class(
        k=data_args.n_best_logits,
        n_best_size=data_args.n_best_size,
        max_answer_length=data_args.max_answer_length,
        scorer_type=SupportedSpanScorers(scorer_type),
        single_context_multiple_passages=preprocessor._single_context_multiple_passages,
        confidence_model_path=model_args.confidence_model_path,
        output_confidence_feature=False
    )

    eval_metrics = getattr(sys.modules[__name__], task_args.eval_metrics)()

    if data_args.train_fof is not None or data_args.eval_fof is not None:
        def compute_metrics(p: EvalPredictionWithProcessing):
            return eval_metrics.compute(predictions=p.processed_predictions, references=p.label_ids,
                passage_non_null_threshold=task_args.passage_non_null_threshold, 
                span_non_null_threshold=task_args.span_non_null_threshold,verbose=task_args.verbose,
                dataset_config_name = eval_datasets[0].config_name)

        kd_args = None
        if training_args.do_train and model_args.kd_teacher_config_path is not None:
            kd_args = {
                'teacher_model_path': model_args.kd_teacher_model_path,
                'teacher_config_path': model_args.kd_teacher_config_path,
                'temperature': model_args.kd_temperature,
                'task_heads': task_heads
            }
        trainer = MSKD_MRCTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            kd_args=kd_args if training_args.do_train else None, # the only extra argument at this point
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
        if data_args.train_fof is not None:
            len_train_datasets = sum([len(dataset) for dataset in train_datasets])
            max_train_samples = (
                data_args.max_train_samples * len(train_datasets) if data_args.max_train_samples is not None else len_train_datasets
            )
            metrics["train_samples"] = min(max_train_samples, len_train_datasets)

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

if __name__ == '__main__':
    main()
