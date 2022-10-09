from typing import List
from dataclasses import dataclass, field

from primeqa.pipelines.components.base import RetrieverComponent


@dataclass
class BM25Retriever(RetrieverComponent):
    """_summary_

    Args:
        index_root: str
        index_name: str
        max_num_documents (int, optional): Maximum number of document. Defaults to 100.

    Returns:
        _type_: _description_
    """

    index_root: str = field(
        metadata={
            "name": "Index root",
            "description": "Path to root directory where index is stored",
        },
    )
    index_name: str = field(
        metadata={
            "name": "Index name",
        },
    )
    max_num_documents: int = field(
        default=100,
        metadata={"name": "Maximum number of documents", "range": [1, 100, 1]},
    )

    def __post_init__(self):
        # Placeholder variables
        self._searcher = None

    def load(self, *args, **kwargs):
        pass

    def retrieve(self, input_texts: List[str], *args, **kwargs):
        pass
