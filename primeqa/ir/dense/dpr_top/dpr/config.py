from dataclasses import dataclass, field


@dataclass
class CommonArguments:
    """
    Arguments used in multiple operational modes (training, indexing, search) with the same default value
    """

    output_dir: str = field(
        default=None,
        metadata={"help": "Output directory to write out search results"},
    )

    queries: str = field(
        default=None,
        metadata={
            "help": "Path to the tsv file where each line is in format 'id\tquery'"
        },
    )


@dataclass
class TrainingArguments:
    """
    Arguments used in training
    """

    bsize: int = field(default=8, metadata={"help": "Batch size"})

    collection: str = field(
        default="", metadata={"help": "Training collection file path"}
    )

    force_confict_free_batches: bool = field(
        default=False,
        metadata={
            "help": "Check that batches do not contain instances s.t. batch negatives will actually be positives"
        },
    )

    encoder_gpu_train_limit: int = field(
        default=8,
        metadata={
            "help": "Max number of instances to encode (-1) to disable gradient checkpointing"
        },
    )

    learning_rate: float = field(
        default=2e-05, metadata={"help": "Learing rate"}
    )

    max_grad_norm: float = field(
        default=2.0, metadata={"help": "Max gradient norm"}
    )

    num_train_epochs: int = field(
        default=3, metadata={"help": "Number of training epochs"}
    )

    positive_pids: str = field(
        default="None", metadata={"help": "Path to the positive passage IDs file"}
    )

    sample_negative_from_top_k: int = field(
        default=5,
        metadata={
            "help": "Take the first negative if <= 0, otherwise sample a negative from the top-k"
        },
    )

    train_dir: str = field(
        default="None", metadata={"help": "Path to the training directory"}
    )

    training_data_type: str = field(
        default="None",
        metadata={
            "help": "Training data type",
            "choices": ["dpr", "kgi_jsonl", "num_triples", "text_triples", "text_triples_with_title"],
        },
    )


@dataclass
class IndexingArguments:
    """
    Arguments used in indexing
    """

    bsize: int = field(default=16, metadata={"help": "Batch size"})

    corpus: str = field(default="None", metadata={"help": "Corpus file path"})

    dpr_ctx_encoder_model_name: str = field(
        default="facebook/dpr-ctx_encoder-multiset-base",
        metadata={"help": "Context encoder model name"},
    )

    dpr_ctx_encoder_path: str = field(
        default="None",
        metadata={
            "help": "Context encoder model path (takes precedence over dpr_ctx_encoder_model_name if specified)"
        },
    )

    embed: str = field(
        default="1of1",
        metadata={"help": 'Embedding shard ID (<n>of<total>, e.g. "2of10")'},
    )

    sharded_index: bool = field(
        default=False, metadata={"help": "Use sharded index"}
    )


@dataclass
class SearchArguments:
    """
    Arguments used in search
    """

    bsize: int = field(default=10, metadata={"help": "Batch size"})

    index_location: str = field(
        default=None, metadata={"help": "Path to the index directory location"}
    )

    model_name_or_path: str = field(
        default="", metadata={"help": "Query encoder model name or path"}
    )

    queries: str = field(
        default=None,
        metadata={
            "help": "Path to the tsv file where each line is in format 'id\tquery'"
        },
    )

    top_k: int = field(
        default=10, metadata={"help": "Number of hits to return"}
    )


@dataclass
class DPRTrainingConfig(CommonArguments, TrainingArguments):
    pass


@dataclass
class DPRIndexingConfig(CommonArguments, IndexingArguments):
    pass


@dataclass
class DPRSearchConfig(CommonArguments, SearchArguments):
    pass
