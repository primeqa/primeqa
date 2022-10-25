from typing import Union, List
from dataclasses import dataclass, field

from primeqa.pipelines.components.base import IndexerComponent
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.indexer import Indexer


@dataclass
class ColBERTIndexer(IndexerComponent):
    """_summary_

    Args:
        index_root (str): Path to root directory where index to be stored.
        index_name (str): Index name.
        checkpoint (str): Model to load.
        similarity (str, optional): Similarity. Defaults to "cosine"
        dim (int, optional): Dimension. Defaults to 128
        query_maxlen (int, optional): Maxium query length. Defaults to 32.
        doc_maxlen (int, optional): Maximum document length. Defaults to 180.
        mask_punctuation (bool, optional): If set to "True", will mask punctuation. Defaults to True.
        bsize (int, optional): Batch size. Defaults to 128.
        amp (int, optional): amp. Defaults to False.
        nbits (int, optional): Number of bits. Defaults to 1.
        kmeans_niters (int, optional): Number of iterations (kmeans). Defaults to 4.
        num_partitions_max (int, optional): Maximum partions size. Defaults to 10000000.

    Raises:
        TypeError: _description_
    """

    checkpoint: str = field(
        metadata={
            "name": "Checkpoint",
            "description": "Path to checkpoint",
        },
    )
    similarity: str = field(
        default="cosine",
        metadata={"name": "Similarity", "options": ["cosine", "l2"]},
    )
    dim: int = field(
        default=128,
        metadata={
            "name": "Dimension",
        },
    )
    query_maxlen: int = field(
        default=32,
        metadata={"name": "Maxium query length", "range": [8, 64, 8]},
    )
    doc_maxlen: int = field(
        default=180,
        metadata={"name": "Maxium document length", "range": [32, 256, 4]},
    )
    mask_punctuation: bool = field(
        default=True,
        metadata={"name": "Mask punctuation", "options": [True, False]},
    )
    bsize: int = field(
        default=128,
        metadata={"name": "Dimension", "range": [8, 256, 8]},
    )
    amp: bool = field(
        default=False,
        metadata={"name": "Amp", "options": [True, False]},
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

    def __post_init__(self):
        self._config = ColBERTConfig(
            index_root=self.index_root,
            index_name=self.index_name,
            index_path=f"{self.index_root}/{self.index_name}",
            similarity=self.similarity,
            dim=self.dim,
            query_maxlen=self.query_maxlen,
            doc_maxlen=self.doc_maxlen,
            mask_punctuation=self.mask_punctuation,
            bsize=self.bsize,
            amp=self.amp,
            nbits=self.nbits,
            kmeans_niters=self.kmeans_niters,
            num_partitions_max=self.num_partitions_max,
        )

        # Placeholder variables
        self._indexer = None

    def load(self, *args, **kwargs):
        self._indexer = Indexer(self.checkpoint, config=self._config)

    def index(self, collection: Union[List[dict], str], *args, **kwargs):
        if not isinstance(collection, str):
            raise TypeError(
                "ColBERT indexer expects path to `documents.tsv` as value for `collection` argument."
            )
        self._indexer.index(
            self.index_name,
            collection,
            overwrite="overwrite" in kwargs and kwargs["overwrite"],
        )
