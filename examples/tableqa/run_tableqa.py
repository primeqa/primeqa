#from transformers import TapasConfig, TapasForQuestionAnswering, AdamW
from primeqa.tableqa.models.tableqa_model import TableQAModel
from primeqa.tableqa.preprocessors.dataset import TableQADataset
from primeqa.tableqa.trainer.tableqa_trainer import TableQATrainer
from dataclasses import dataclass, field
from transformers import TapasConfig
from transformers import (
    DataCollator,
    HfArgumentParser,
    TrainingArguments,
    set_seed,default_data_collator,
)
import pandas as pd
from primeqa.tableqa.utils.data_collator import TapasCollator
@dataclass
class TapasArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    dataset_name: str = field(
       default='wikisql', metadata={"help": "Name of the dataset to train the tapas model on"}
    )
    num_aggregation_labels: int = field(
       default=4, metadata={"help": "Total number of aggregation labels"}
    )
    use_answer_as_supervision: bool = field(
        default=True, metadata={"help": "Whether to use answer as supervision or not"}
    )
    answer_loss_cutoff: float = field(
        default=0.664694, metadata={"help": "Answer loss cutoff"}
    )
    cell_selection_preference: float = field(
        default=0.207951, metadata={"help": "Cell selection preference"}
    )

    huber_loss_delta: float = field(
        default=0.121194, metadata={"help": "Huber loss delta"}
    )
    init_cell_selection_weights_to_zero: bool = field(
        default=True, metadata={"help": "Init cell selection weights to zero or not"}
    )
    select_one_column: bool = field(
        default=True, metadata={"help": "select one column"}
    )
    allow_empty_column_selection: bool = field(
        default=True, metadata={"help": "Allow empty column selection"}
    )
    temperature: float = field(
        default=0.0352513, metadata={"help": "temperature"}
    )


def main():
    parser = HfArgumentParser((TapasArguments, TrainingArguments))
    tapas_args,training_args = parser.parse_args_into_dataclasses()
    print(tapas_args)
    config = TapasConfig(tapas_args)
    tableqa_model = TableQAModel("google/tapas-base",config=config)
    model = tableqa_model.model
    if training_args.do_train or training_args.do_eval:
        if tapas_args.dataset_name=="sqa":
            sqa_dataset = TableQADataset(tapas_args.dataset_name,tableqa_model.tokenizer)
            train_dataset = sqa_dataset.load_data("dev")
            eval_dataset = sqa_dataset.load_data("dev")
        elif tapas_args.dataset_name=="wikisql":
            print("loading wikisql dataset")
            wikisql_dataset =  TableQADataset(tapas_args.dataset_name,tableqa_model.tokenizer)
            train_dataset = wikisql_dataset.load_data("train")
            eval_dataset = wikisql_dataset.load_data("dev")
        trainer = TableQATrainer(model=model,
                                args=training_args,
                                train_dataset=train_dataset if training_args.do_train else None,
                                eval_dataset=eval_dataset if training_args.do_eval else None,
                                tokenizer=tableqa_model.tokenizer,
                                data_collator=TapasCollator(),
                                )
        if training_args.do_train:
            train_result = trainer.train()
            trainer.save_model()
            metrics = train_result.metrics
            trainer.log_metrics("train", metrics)
            trainer.save_metrics("train", metrics)
            trainer.save_state()
        if training_args.do_eval:
            print("*** Evaluate ***")
            metrics = trainer.evaluate()
            trainer.log_metrics("eval", metrics)
            trainer.save_metrics("eval", metrics)


if __name__ == '__main__':
       main()
    