from transformers import (
    DataCollator,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from primeqa.qg.processors.data_loader import QGDataLoader
import torch
from dataclasses import dataclass,field
from typing import Optional, List, Dict


@dataclass
class T2TDataCollator:
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
        

        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'labels': lm_labels, 
            'decoder_attention_mask': decoder_attention_mask
        }


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
       default='t5-base', metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    modality: str = field(
       default='table', metadata={"help": "If to work on tables or passages",
                                  "choices":["table", "passage"]}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do we want to store the pretrained models downloaded from s3"}
    )


@dataclass
class QGTrainingArguments(TrainingArguments):
    """
    Arguments pertraining to model training hyperparameters
    """
    num_cores: Optional[int] = field(
        default=1, metadata={"help":"Number of cpu cores to use"}
    )
    n_gpu: Optional[int] = field(
        default=1, metadata={"help": "Number of gpus to train on"}
    )
    per_gpu_train_batch_size: int = field(
        default=8, metadata={"help": "Train Batch size per gpu"}
    )
    per_gpu_eval_batch_size: int = field(
        default=8, metadata={"help": "Dev batch size per gpu"}
    ) 
    gradient_accumulation_steps: Optional[int] = field(
        default=4, metadata={"help": "Gradient acculation steps "}
    )
    learning_rate: Optional[float]= field(
        default=0.0001, metadata={"help": "Learning rate to be used during training the model"}
    )
    num_train_epochs:int = field(
        default=4, metadata={"help": "Numbers of epochs to train the model"}
    )
    do_train:Optional[bool] = field(
        default=False, metadata={"help": "Whether to train the model or not"}
    )
    do_eval:Optional[bool] = field(
        default=False,metadata={"help": "run evaluation on dev set"}
    )
    do_predict:Optional[bool] = field(
        default=False, metadata={"help": "Generate model prediction on test set"}
    )
    remove_unused_columns:Optional[bool] = field(
        default=False, metadata={"help": ""}
    )
    output_dir:Optional[str] = field(
        default='./models/qg/sample_run/', metadata={"help": "Models and checkpoints will be saved here"}
    )
    args_file_path:Optional[str] = field(
        default='', metadata={"help": "Path to JSON file with all arguments needed"}
    )
    prediction_loss_only:Optional[bool] = field(
        default=True, metadata={"help": ""}
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name:Optional[str] = field(
        default="wikisql", metadata={"help": "Name of the dataset to train the qg model",
                                     "choices": ["wikisql", "squad", "squad_v2"]}
    )
    train_file_path: Optional[str] = field(
        default='train_data.pt',
        metadata={"help": "Path for cached train dataset"},
    )
    valid_file_path: Optional[str] = field(
        default='valid_data.pt',
        metadata={"help": "Path for cached valid dataset"},
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
        default=False, metadata={"help": "Whether to generate."}
    )
    num_questions_per_instance: Optional[int] = field(
        default=5, metadata={"help": "Number of questions to generate per table/passage"}
    )
    max_where_clauses: Optional[int] = field(
        default=1, metadata={"help": "Max number of filters in generated question"}
    )
    data_path:str = field(
        default='examples/qg/sample_table.json', metadata={"help": "Path to JSON file with LIST of tables/passages. Each table \
                              should be a dict with keys 'header' and 'rows', and passages should be str"}
    )
    generate_aggregate: Optional[bool] = field(
        default=False, metadata={"help": "Whether to generate aggregate questions with max, min, sum, etc."}
    )
    gen_output_path: Optional[str] = field(
        default='examples/qg/sample_generation.json', metadata={"help": "path to JSON fiel where generated questions will be saved"} 
    )