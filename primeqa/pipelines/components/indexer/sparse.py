import logging
from typing import Union, List
from dataclasses import dataclass

from primeqa.pipelines.components.base import IndexerComponent


@dataclass
class BM25Indexer(IndexerComponent):
    """_summary_

    Args:
        logger (logging.Logger, optional) = logger object. Defaults to logging.getLogger(BM25Indexer).
    """

    logger: logging.Logger = logging.getLogger("BM25Indexer")

    def __post_init__(self):
        self.name = "BM25 Indexer"
        self.type = IndexerComponent.__name__

    def load(self, *args, **kwargs):
        pass

    def index(self, collection: Union[List[dict], str], *args, **kwargs):
        pass
