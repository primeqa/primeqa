import logging
from typing import List
from dataclasses import dataclass

from primeqa.pipelines.components.base import RetrieverComponent


@dataclass
class BM25Retriever(RetrieverComponent):
    """_summary_

    Args:
        index_root: str
        index_name: str
        max_num_documents (int, optional): Maximum number of document. Defaults to 100.
        logger (logging.Logger, optional): logger object. Defaults to logging.getLogger(BM25Retriever).

    Returns:
        _type_: _description_
    """

    index_root: str
    index_name: str
    max_num_documents: int = 100
    logger: logging.Logger = logging.getLogger("BM25Retriever")

    def __post_init__(self):
        self.name = "BM25 Retriever"
        self.type = RetrieverComponent.__name__

        # Placeholder variables
        self._searcher = None

    def load(self, *args, **kwargs):
        pass

    def retrieve(self, input_texts: List[str], *args, **kwargs):
        pass
