from dataclasses import dataclass, field
import torch
from typing import Any, Dict, List, Optional, Union
import os
import sys
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
@dataclass
class RRArguments():
   per_device_train_batch_size_rr: int = field(
       default=8, metadata={"help": "Train batch size"}
    )
   per_device_eval_batch_size_rr: int = field(
       default=8, metadata={"help": "Train batch size"}
    )
   rr_model_name: str = field(
       default='bert-base-uncased', metadata={"help": "Which model to use for RR training/testing"}
    )
   row_retriever_model_name_path: str = field(
       default=None, metadata={"help": "Row retriever configuration file"}
    )
   pos_frac_per_epoch: List[float] = field(
      default_factory=lambda: [0.3, 0.3, 0.1, 0.0001, 0.0001], metadata={"help": "Positive fraction per epoch"}
   )
   group_frac_per_epoch: List[float]  = field(
      default_factory=lambda: [0.0, 0.5, 1.0, 1.0, 1.0], metadata={"help": "Group fraction per epoch"}
   )
   num_train_epochs_rr: int = field(
      default=2,metadata={"help": "Number of epochs to train the row retriever"}
   )
   save_every_niter_rr: int = field(
      default=100,metadata={"help": "Save model after how many iterations"}
   )
   save_model_path_rr: str = field(
      default='data/hybridqa/models/rr.bin',metadata={"help": "Path to save row retrieval model"}
   )

@dataclass
class AEArguments(TrainingArguments):
   max_seq_length: int = field(
        default=512,metadata={"help": "Input Sequence Length"}
    )
   per_gpu_train_batch_size: int = field(
        default=8,metadata={"help": "Per GPU train batch size"}
    )
   train_batch_size: int = field(
        default=8,metadata={"help": "Train batch size"}
    )
   per_gpu_eval_batch_size: int = field(
        default=8,metadata={"help": "Per GPU eval batch size"}
    )
   max_query_length: int = field(
        default=64,metadata={"help": "Maximum length of the query"}
    )
   threads: int = field(
        default=1,metadata={"help": "Number of preprocessing threads"}
    )
   top_k_rr: int = field(
        default=5,metadata={"help": "Top K "}
    )
   null_score_diff_threshold: float = field(
        default=0.0,metadata={"help": "If null_score - best_non_null is greater than the threshold predict null"}
    )
   eval_batch_size: int = field(
        default=8,metadata={"help": "Evaluation batch size"}
    )
   n_best_size: int = field(
        default=20,metadata={"help": "The total number of n-best predictions to generate in the nbest_predictions.json output file."}
    )
   do_lower_case: bool = field(
      default=True,metadata={"help": "Do lowercase the input"}
   )
   do_train_ae: bool = field(
      default=False,metadata={"help": "Do training"}
   )
   verbose_logging: bool = field(
      default=False,metadata={"help": "Log everything"}
   )
   do_predict_ae: bool = field(
      default=False,metadata={"help": "Do predict"}
   )
   version_2_with_negative: bool = field(
      default=False,metadata={"help": "Squad 2.0"}
   )
   do_eval_ae: bool = field(
      default=False,metadata={"help": "Do evaluation"}
   )
   device: torch.device = field(
        default=torch.device("cpu"),metadata={"help": "Whether to use cpu or gpu"}
    )
   n_gpu: int = field(
        default=1,metadata={"help": "Number of GPUs"}
    )
   max_answer_length: int = field(
        default=30,metadata={"help": "Maximum length of the query"}
    )
   model_name_or_path_ae: str = field(
      default="bert-base-uncased",metadata={"help":"Model name or path"}
   )
   model_type: str = field(
       default='bert', metadata={"help": "Type of model to be train"}
    )
   config_name: str = field(
       default='', metadata={"help": "Config name"}
    )
   tokenizer_name: str = field(
       default='', metadata={"help": "Tokenizer name"}
    )
   cache_dir: str = field(
       default='/tmp/', metadata={"help": "Temp directory for caching"}
    )
   doc_stride: int = field(
       default=128, metadata={"help": "Doc Stride"}
    )
   train_file: str = field(
       default='data/hybridqa/ae_input_test.json', metadata={"help": "Type of model to be train"}
    )
   eval_file: str = field(
       default='data/hybridqa/ae_input_test.json', metadata={"help": "Type of model to be train"}
    )
   pred_ans_file: str = field(
      default='data/hybridqa/predictions/answer_extractor_output_test.json', metadata={"help": "Row retriever configuration file"}
   )
   
   
@dataclass
class LinkPredictorArguments:
   """
    Arguments pertaining to the link prediction module
   """
   model: str = field(
       default='gpt2', metadata={"help": "Pre-trained link prediction model"}
    )
   top_k: int = field(
       default=0, metadata={"help": "Top k links to predict"}
    )
   top_p: float = field(
       default=0.9, metadata={"help": "Top p value"}
    )
   seed_lg: int = field(
       default=42, metadata={"help": "Random seed"}
    )
   num_epoch_lg: int = field(
       default=2, metadata={"help": "Number of epochs"}
    )
   dataset: str = field(
       default=None, metadata={"help": "Which dataset to use"}
    )
   batch_size_lg: int = field(
       default=64, metadata={"help": "Batch size"}
    )
   linker_model: str = field(
       default=None, metadata={"help": "Load from the checkpoint"}
    )
   every: int = field(
       default=50, metadata={"help": "Batch size"}
    )
   max_source_len: int = field(
       default=32, metadata={"help": "Maximum source length"}
    )
   max_target_len: int = field(
       default=16, metadata={"help": "Maximum target length"}
    )
   do_train_lg: bool = field(
        default=False, metadata={"help": "Do_training"}
    )
   do_val_lg: bool = field(
        default=False, metadata={"help": "Do validation"}
    )
   do_all_lg: bool = field(
        default=False, metadata={"help": "Generate links for all the tables"}
    )
   learning_rate_lg: float = field(
       default=5e-6, metadata={"help": "Learning rate for training"})
   shard: str = field(
       default=None, metadata={"help": "Which shard"}
    )
   device_lg: torch.device = field(
        default=torch.device("cuda"),metadata={"help": "Whether to use cpu or gpu"}
    )
   
    
   

@dataclass
class HybridQAArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    data_path_root: str = field(
       default='data/ottqa/', metadata={"help": "Root path to store the preprocessed dataset"}
    )
    dataset_name: str = field(
       default='hybridqa', metadata={"help": "Name of the dataset"}
    )
    train_data_path: str = field(
       default='data/hybridqa/train.json', metadata={"help": "Train data path for training on user's own dataset"}
    )
    dev_data_path: str = field(
       default='data/hybridqa/toy.json', metadata={"help": "Dev data path for training on user's own dataset"}
    )
    test_data_path: str = field(
       default='data/hybridqa/test.json', metadata={"help": "Test data path"}
    )
    collections_file: str = field(
       default='linearized_tables.tsv', metadata={"help": "Collection/corpus file for table retriever"}
    )
    
    test: Optional[bool] = field(
        default=False, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    train_tr: Optional[bool] = field(
        default=False, metadata={"help": "Whether to train the table retriever or not"}
    )
    train_lp: Optional[bool] = field(
        default=False, metadata={"help": "Whether to train the link generator or not"}
    )