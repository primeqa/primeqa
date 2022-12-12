from typing import List
from dataclasses import dataclass, field
import json

from primeqa.pipelines.components.base import RetrieverComponent
from primeqa.ir.sparse.retriever import PyseriniRetriever


@dataclass
class BM25Retriever(RetrieverComponent):
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

    max_num_documents: int = field(
        default=5,
        metadata={
            "name": "Maximum number of retrieved documents",
            "range": [1, 100, 1],
            "api_support": True,
            "exclude_from_hash": True,
        },
    )
    
    num_workers: int = field(
        default=1,
        metadata={
            "name": "Num worker threads",
            "range": [1, 100, 1],
            "exclude_from_hash": True,
        },
    )

    def __post_init__(self):
        # Placeholder variables
        self._searcher = None
        
    def __hash__(self) -> int:
        # Step 1: Identify all fields to be included in the hash
        hashable_fields = [
            k
            for k, v in self.__class__.__dataclass_fields__.items()
            if not "exclude_from_hash" in v.metadata
            or not v.metadata["exclude_from_hash"]
        ]

        # Step 2: Run
        return hash(
            f"{self.__class__.__name__}::{json.dumps({k: v for k, v in vars(self).items() if k in hashable_fields}, sort_keys=True)}"
        )

    def load(self, *args, **kwargs):
        self._searcher = PyseriniRetriever(self.index_root)

    def retrieve(self, input_texts: List[str], *args, **kwargs):
        qids = [str(idx) for  idx, query in enumerate(input_texts) ]
        hits = self._searcher.batch_retrieve(input_texts, qids, topK=self.max_num_documents, threads=self.num_workers)
        return [
            [(result['doc_id'], result['score']) for result in results_per_query]
            for results_per_query in hits.values()
        ]
