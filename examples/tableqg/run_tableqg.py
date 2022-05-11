from transformers import (
    DataCollator,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from oneqa.tableqg.processors.table_data import QGDataLoader
import torch
from dataclasses import dataclass,field
from oneqa.tableqg.models.tableqg_model import TableQG
from oneqa.tableqg.trainers.qg_trainer import QGTrainer
from typing import Optional, List, Dict

import logging

import os

logger = logging.getLogger(__name__)

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
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do we want to store the pretrained models downloaded from s3"}
    )


@dataclass
class TableQGTrainingArguments(TrainingArguments):
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
        default=True, metadata={"help": "Whether to train the model or not"}
    )
    do_eval:Optional[bool] = field(
        default=True,metadata={"help": "run evaluation on dev set"}
    )
    do_predict:Optional[bool] = field(
        default=False, metadata={"help": "Generate model prediction on test set"}
    )
    remove_unused_columns:Optional[bool] = field(
        default=False, metadata={"help": ""}
    )
    prediction_loss_only:Optional[bool] = field(
        default=True, metadata={"help": ""}
    )
    output_dir:Optional[str] = field(
        default='./models/', metadata={"help": "Models and checkpoints will be saved here"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    dataset_name:Optional[str] = field(
        default="wikisql", metadata={"help": "Name of the dataset to train the tableqg model"}
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



def main():

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TableQGTrainingArguments))

    # We need to load the config hyperparameter from args file path
    if os.path.exists(os.path.abspath('args.json')):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath('args.json'))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
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

    tqg = TableQG(model_args.model_name_or_path)
    model = tqg.model
    tokenizer = tqg.tokenizer

    qgdl = QGDataLoader(tokenizer,data_args)
    train_dataset = qgdl.create("train")
    valid_dataset = qgdl.create("validation")

    trainer = QGTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        data_collator=T2TDataCollator()
    )
    if training_args.do_train:
        trainer.train(
        model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
    )
    trainer.save_model()

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))
    
        results.update(eval_output)
    
    return results


if __name__ == "__main__":
    main()

