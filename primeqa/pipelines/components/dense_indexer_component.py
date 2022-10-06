import logging
from typing import Union, List

from primeqa.pipelines.components.base import IndexerComponent
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.indexer import Indexer


class ColBERTIndexerComponent(IndexerComponent):
    def __init__(
        self,
        index_root: str,
        checkpoint: str,
        index_name: str = "index",
        similarity: str = "cosine",
        dim: int = 128,
        query_maxlen: int = 32,
        doc_maxlen: int = 180,
        mask_punctuation: bool = True,
        bsize: int = 128,
        amp: bool = False,
        nbits: int = 1,
        kmeans_niters: int = 4,
        num_partitions_max: int = 10000000,
        logger: Union[logging.Logger, None] = None,
    ) -> None:
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger

        # Default object variables
        self.name = "ColBERT Indexer"
        self.type = IndexerComponent.__name__

        # Custom object variable
        self.index_root = index_root
        self.checkpoint = checkpoint
        self.index_name = index_name
        self.similarity = similarity
        self.dim = dim
        self.query_maxlen = query_maxlen
        self.doc_maxlen = doc_maxlen
        self.mask_punctuation = mask_punctuation
        self.bsize = bsize
        self.amp = amp
        self.nbits = nbits
        self.kmeans_niters = kmeans_niters
        self.num_partitions_max = num_partitions_max

        # Create configuration
        self.config = ColBERTConfig(
            index_root=self.index_root,
            index_name=index_name,
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
        self.indexer = None

    def load(self, *args, **kwargs):
        self.indexer = Indexer(self.checkpoint, config=self.config)

    def index(self, collection: Union[List[dict], str], *args, **kwargs):
        if not isinstance(collection, str):
            raise TypeError(
                "ColBERT indexer expects path to `documents.tsv` as value for `collection` argument."
            )
        self.indexer.index(self.index_name, collection)
