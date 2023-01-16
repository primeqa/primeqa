from dataclasses import dataclass, field

@dataclass
class TableQAArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    data_path_root: str = field(
       default='primeqa/tableqa/preprocessors/data/wikisql/', metadata={"help": "root path to store the preprocessed dataset"}
    )
    train_data_path: str = field(
       default='primeqa/tableqa/preprocessors/data/wikisql/', metadata={"help": "Train data path for training on user's own dataset"}
    )
    dev_data_path: str = field(
       default='primeqa/tableqa/preprocessors/data/wikisql/', metadata={"help": "Dev data path for training on user's own dataset"}
    )

    # dataset_name: str = field(
    #    default='wikisql', metadata={"help": "Name of the dataset to train the tapas model on"}
    # )
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