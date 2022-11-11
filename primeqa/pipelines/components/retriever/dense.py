from typing import List
from dataclasses import dataclass, field
import json

from primeqa.pipelines.components.base import RetrieverComponent
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher


@dataclass
class ColBERTRetriever(RetrieverComponent):
    """_summary_

    Args:
        index_root: str
        index_name: str
        checkpoint (str, optional): Model to load. Defaults to checkpoint in index configuration.
        collection (str, optional): collection to load. Defaults to collection in index configuration.
        max_num_documents (int, optional): Maximum number of retrieved document. Defaults to 100.
        ncells (int, optional): Number of cells. Defaults to None.
        centroid_score_threshold (float, optional): Centroid score threshold. Defaults to None.
        ndocs (int, optional): Number of documents in PLAID Stage 1. Defaults to None.

    Returns:
        _type_: _description_

    """

    checkpoint: str = field(
        default=None,
        metadata={
            "name": "Checkpoint",
            "description": "Path to checkpoint",
            "api_support": True,
        },
    )
    collection: str = field(
        default=None,
        metadata={
            "name": "Collection",
            "description": "Path to collection",
        },
    )
    max_num_documents: int = field(
        default=100,
        metadata={
            "name": "Maximum number of retrieved documents",
            "range": [1, 100, 1],
            "api_support": True,
            "exclude_from_hash": True,
        },
    )
    ncells: int = field(
        default=None,
        metadata={
            "name": "Number of cells",
        },
    )
    centroid_score_threshold: float = field(
        default=None,
        metadata={
            "name": "Centroid Score Threshold",
        },
    )
    ndocs: int = field(
        default=None,
        metadata={
            "name": "Number of documents in PLAID Stage 1",
        },
    )

    def __post_init__(self):
        self._config = ColBERTConfig(
            index_root=self.index_root,
            index_name=self.index_name,
            index_path=f"{self.index_root}/{self.index_name}",
            ncells=self.ncells,
            centroid_score_threshold=self.centroid_score_threshold,
            ndocs=self.ndocs,
        )

        # Placeholder variables
        self._searcher = None

    def __hash__(self) -> int:
        return hash(
            f"{self.__class__.__name__}::{json.dumps({k: v.default for k, v in self.__class__.__dataclass_fields__.items() if not 'exclude_from_hash' in v.metadata or not v.metadata['exclude_from_hash']}, sort_keys=True)}"
        )

    def load(self, *args, **kwargs):
        self._searcher = Searcher(
            self.index_name,
            checkpoint=self.checkpoint,
            collection=self.collection,
            config=self._config,
        )

    def retrieve(self, input_texts: List[str], *args, **kwargs):
        # TODO: Add kwarg defining return format (List[List[Tuple(pids, score)]], List[List[<document>]])
        ranking_results = self._searcher.search_all(
            {idx: str(input_text) for idx, input_text in enumerate(input_texts)},
            k=self.max_num_documents,
        )
        return [
            [(result[0], result[-1]) for result in results_per_query]
            for results_per_query in ranking_results.data.values()
        ]
