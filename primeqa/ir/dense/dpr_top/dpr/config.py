from dataclasses import dataclass, field


@dataclass
class DPRTrainingArguments:
    """
    Arguments used in training
    Note:
        here and in the other DPRXArguments classes:
        (1) Some of the argument name (e.g. "bsize") are chosen to match between the training/indexing/search modalities, and also other IR engines in PrimeQA.
        (2) The argument names are sorted alphabetically, with the mandatory arguments listed first.
    """

    output_dir: str = field(
        metadata={"help": "Output directory to write results"},
    )

    bsize: int = field(default=8, metadata={"help": "Batch size"})

    collection: str = field(
        default="", metadata={"help": "Training collection file path"}
    )

    ctx_encoder_name_or_path: str = field(
        default="facebook/dpr-ctx_encoder-multiset-base",
        metadata={"help": "Query model name or path"},
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

    max_negatives: int = field(
        default=0,
        metadata={
            "help": 'Max non-hard negatives (only applies with the "dpr" training data type)'
        },
    )

    max_hard_negatives: int = field(
        default=1,
        metadata={
            "help": 'Max hard negatives (only applies with the "dpr" training data type)'
        },
    )

    epochs: int = field(
        default=3, metadata={"help": "Number of training epochs"}
    )

    positive_pids: str = field(
        default="None",
        metadata={"help": "Path to the positive passage IDs file"},
    )

    qry_encoder_name_or_path: str = field(
        default="facebook/dpr-question_encoder-multiset-base",
        metadata={"help": "Query model name or path"},
    )

    queries: str = field(
        default=None,
        metadata={
            "help": 'Path to the tsv file where each line is in format "id<TAB>query"'
        },
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

    training_data_type: str = field(
        default="None",
        metadata={
            "help": "Training data type",
            "choices": [
                "dpr",
                "kgi_jsonl",
                "num_triples",
                "text_triples",
                "text_triples_with_title",
            ],
        },
    )

    warmup_instances: int = field(
        default=0, metadata={"help": "Number of warm-up instances"}
    )

    warmup_fraction: float = field(
        default=0.0,
        metadata={
            "help": "Warm-up instances fraction, only applies if warmup_instances <= 0"
        },
    )


@dataclass
class DPRIndexingArguments:
    """
    Arguments used in indexing
    """

    output_dir: str = field(
        metadata={"help": "Output directory to write results"},
    )

    bsize: int = field(default=16, metadata={"help": "Batch size"})

    collection: str = field(
        default="None", metadata={"help": "Collection file path"}
    )

    ctx_encoder_name_or_path: str = field(
        default="facebook/dpr-ctx_encoder-multiset-base",
        metadata={"help": "Query model name or path"},
    )

    embed: str = field(
        default="1of1",
        metadata={"help": 'Embedding shard ID (<n>of<total>, e.g. "2of10")'},
    )

    sharded_index: bool = field(
        default=True, metadata={"help": "Use sharded index"}
    )


@dataclass
class DPRSearchArguments:
    """
    Arguments used in search
    """

    output_dir: str = field(
        default="",
        metadata={"help": "Output directory to write results"},
    )

    queries: str = field(
        default="",
        metadata={
            "help": "Path to the tsv file where each line is in format 'id\tquery'"
        },
    )

    bsize: int = field(default=10, metadata={"help": "Batch size"})

    index_location: str = field(
        default=None, metadata={"help": "Path to the index directory location"}
    )

    model_name_or_path: str = field(
        default="",
        metadata={"help": "Query encoder model name or path"},
    )

    qry_encoder_name_or_path: str = field(
        default="facebook/dpr-question_encoder-multiset-base",
        metadata={"help": "Query model name or path"},
    )

    ctx_encoder_name_or_path: str = field(
        default="facebook/dpr-ctx_encoder-multiset-base",
        metadata={"help": "Query model name or path"},
    )

    top_k: int = field(
        default=10, metadata={"help": "Number of hits to return"}
    )

    rescore_only: bool = field(
        default=False, metadata={"help": "Running in the rescoring mode"}
    )

    max_doc_length: int = field(
        default=128, metadata={"help": "Maximum number of tokens in a document"}
    )