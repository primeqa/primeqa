from typing import List
from dataclasses import dataclass, field

from primeqa.Components.base import Retriever


@dataclass
class BM25Retriever(Retriever):
    """_summary_

    Args:
        index_root: str
        index_name: str
        max_num_documents (int, optional): Maximum number of document. Defaults to 5.

    Important:
    1. Each field has metadata property which can carry additional information for other downstream usages.
    2. Two special keys (api_support and exclude_from_hash) are defined in "metadata" property.
        a. api_support (bool, optional): If set to True, that parameter is exposed via service layer. Defaults to False.
        b. exclude_from_hash (bool,optional): If set to True, that parameter is not considered while building the hash representation for the object. Defaults to False.

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
        default=5,
        metadata={"name": "Maximum number of documents", "range": [1, 100, 1]},
    )

    def __post_init__(self):
        # Placeholder variables
        self._searcher = None

    def load(self, *args, **kwargs):
        pass

    def retrieve(self, input_texts: List[str], *args, **kwargs):
        pass
