from dataclasses import dataclass, field


@dataclass
class XTRTrainingArguments:
    """
    Arguments used in training
    Note:
        here and in the other DPRXArguments classes:
        (1) Some of the argument name (e.g. "bsize") are chosen to match between the training/indexing/search modalities, and also other IR engines in PrimeQA.
        (2) The argument names are sorted alphabetically, with the mandatory arguments listed first.
    """


    collection: str = field(
        metadata={"help": "Training collection file path"}
    )

    bsize: int = field(default=8, metadata={"help": "Batch size"})

    model_name_or_path: str = field(
        default="t5-small",
        metadata={"help": "encoder model name or path"},
    )

    learning_rate: float = field(
        default=1e-03, metadata={"help": "Learing rate"}
    )

    max_grad_norm: float = field(
        default=2.0, metadata={"help": "Max gradient norm"}
    )

    max_hard_negatives: int = field(
        default=1,
        metadata={
            "help": 'Max hard negatives (only applies with the "xtr" training data type)'
        },
    )

    epochs: int = field(
        default=3, metadata={"help": "Number of training epochs"}
    )

    resume_from_checkpoint: str = field(
        default="",
        metadata={
            "help": "Path to the checkpoint file used to resume training"
        },
    )

    sample_negative_from_top_k: int = field(
        default=1,
        metadata={
            "help": "Take the first negative if <= 0, otherwise sample a negative from the top-k"
        },
    )

    train_dir: str = field(
        default="None", metadata={"help": "Path to the training directory"}
    )
    

@dataclass
class XTRIndexingArguments:
    """
    Arguments used in indexing
    """

    collection: str = field(
        default="None", metadata={"help": "Collection file path"}
    )

    model_name_or_path: str = field(
        default="t5-small",
        metadata={"help": "Query model name or path"},
    )

    index_name: str = field(
        default="dummy",
        metadata={"help": 'index directory'},
    )

    dim: int = field(
        default=128,
        metadata={
            "name": "Dimension",
        },
    )
    query_maxlen: int = field(
        default=32,
        metadata={
            "name": "Maximum query length",
            "range": [8, 64, 8],
        },
    )
    doc_maxlen: int = field(
        default=180,
        metadata={
            "name": "Maximum document length",
            "range": [32, 256, 4],
        },
    )
    bsize: int = field(
        default=128,
        metadata={"name": "Dimension", "range": [8, 256, 8]},
    )
    rank: int = field(
        default=-1,
        metadata={"name": "Rank"},
    )
    nranks: int = field(
        default=1,
        metadata={"name": "nRanks"},
    )
    nbits: int = field(
        default=1,
        metadata={"name": "nbits", "options": [1, 2, 4]},
    )
    kmeans_niters: int = field(
        default=4,
        metadata={"name": "Number of iterations (kmeans)", "range": [1, 8, 1]},
    )
    num_partitions_max: int = field(
        default=10000000,
        metadata={
            "name": "Maximum number of partitions",
        },
    )

@dataclass
class XTRSearchArguments:
    """
    Arguments used in search
    """

    index_name: str = field(
        metadata={
            "name": "Index name",
        },
    )
    model_name_or_path: str = field(
        default=None,
        metadata={
            "name": "Model",
            "description": "Path to checkpoint",
        },
    )
    collection: str = field(
        default=None,
        metadata={
            "name": "Collection",
            "description": "Path to collection",
        },
    )
    ncells: int = field(
        default=None,
        metadata={
            "name": "Number of cells",
        },
    )
    ndocs: int = field(
        default=None,
        metadata={
            "name": "Number of documents in PLAID Stage 1",
        },
    )

    dim: int = field(
        default=128,
        metadata={
            "name": "Bottleneck Dimension",
        },
    )

    bsize: int = field(default=10, metadata={"help": "Batch size"})

    topK: int = field(
        default=10, metadata={"help": "Number of hits to return"}
    )
    nbits: int = field(
        default=1,
        metadata={"name": "nbits", "options": [1, 2, 4]},
    )

    query_maxlen: int = field(
        default=32,
        metadata={
            "name": "Maximum query length",
            "range": [8, 64, 8],
        },
    )
