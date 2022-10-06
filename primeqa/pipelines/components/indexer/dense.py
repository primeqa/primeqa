import logging
from typing import Union, List
from dataclasses import dataclass

from primeqa.pipelines.components.base import IndexerComponent
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.indexer import Indexer


@dataclass
class ColBERTIndexer(IndexerComponent):
    """_summary_

    Args:
        index_root: str
        checkpoint: str
        index_name (str, optional): Index name. Defaults to "index"
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
        ncells (int, optional): Number of cells. Defaults to None.
        logger (logging.Logger, optional) = logger object. Defaults to logging.getLogger(ColBERTIndexer).

    Raises:
        TypeError: _description_
    """

    index_root: str
    checkpoint: str
    index_name: str = "index"
    similarity: str = "cosine"
    dim: int = 128
    query_maxlen: int = 32
    doc_maxlen: int = 180
    mask_punctuation: bool = True
    bsize: int = 128
    amp: bool = False
    nbits: int = 1
    kmeans_niters: int = 4
    num_partitions_max: int = 10000000
    logger: logging.Logger = logging.getLogger("ColBERTIndexer")

    def __post_init__(self):
        self.name = "ColBERT Indexer"
        self.type = IndexerComponent.__name__

        self._config = ColBERTConfig(
            index_root=self.index_root,
            index_name=self.index_name,
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
        self._indexer.index(self.index_name, collection)
