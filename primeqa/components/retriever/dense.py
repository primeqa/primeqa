from typing import List, Any
import os
from dataclasses import dataclass, field
import json

from primeqa.components.base import Retriever as BaseRetriever
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher
from primeqa.ir.dense.dpr_top.dpr.config import DPRSearchArguments
from primeqa.ir.dense.dpr_top.dpr.searcher import DPRSearcher
from primeqa.components.indexer.dense import DPRIndexer


@dataclass
class ColBERTRetriever(BaseRetriever):
    """_summary_

    Args:
        index_root: str
        index_name: str
        checkpoint (str, optional): Model to load. Defaults to checkpoint in index configuration.
        collection (str, optional): collection to load. Defaults to collection in index configuration.
        max_num_documents (int, optional): Maximum number of retrieved document. Defaults to 5.
        ncells (int, optional): Number of cells. Defaults to None.
        centroid_score_threshold (float, optional): Centroid score threshold. Defaults to None.
        ndocs (int, optional): Number of documents in PLAID Stage 1. Defaults to None.

    Important:
    1. Each field has metadata property which can carry additional information for other downstream usages.
    2. Two special keys (api_support and exclude_from_hash) are defined in "metadata" property.
        a. api_support (bool, optional): If set to True, that parameter is exposed via service layer. Defaults to False.
        b. exclude_from_hash (bool,optional): If set to True, that parameter is not considered while building the hash representation for the object. Defaults to False.

    Returns:
        _type_: _description_

    """

    checkpoint: str = field(
        default=None,
        metadata={
            "name": "Checkpoint",
            "description": "Path to checkpoint",
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
        default=5,
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
        self._searcher = Searcher(
            self.index_name,
            checkpoint=self.checkpoint,
            collection=self.collection,
            config=self._config,
        )

    @classmethod
    def get_engine_type(cls):
        return "ColBERT"

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass

    def predict(self, input_texts: List[str], *args, **kwargs) -> Any:
        """Retrieves relevant documents based on input_texts

        Args:
            input_texts (List[str]): search queries

        Returns:
            Any: List of tuples. Each tuple contains a document indetifier and relevancy score
        """
        # Step 1: Locally update object variable values, if provided
        max_num_documents = (
            kwargs["max_num_documents"]
            if "max_num_documents" in kwargs
            else self.max_num_documents
        )
        ranking_results = self._searcher.search_all(
            {idx: str(input_text) for idx, input_text in enumerate(input_texts)},
            k=max_num_documents,
        )
        return [
            [(result[0], result[-1]) for result in results_per_query]
            for results_per_query in ranking_results.data.values()
        ]

@dataclass
class DPRRetriever(BaseRetriever):
    """_summary_

    Args:
        index_root: str
        index_name: str
        checkpoint (str, optional): Model to load. Defaults to checkpoint in index configuration.
        collection (str, optional): collection to load. Defaults to collection in index configuration.
        max_num_documents (int, optional): Maximum number of retrieved document. Defaults to 5.

    Important:
    1. Each field has metadata property which can carry additional information for other downstream usages.
    2. Two special keys (api_support and exclude_from_hash) are defined in "metadata" property.
        a. api_support (bool, optional): If set to True, that parameter is exposed via service layer. Defaults to False.
        b. exclude_from_hash (bool,optional): If set to True, that parameter is not considered while building the hash representation for the object. Defaults to False.

    Returns:
        _type_: _description_

    """
    indexer: DPRIndexer = field(
        default=None,
        metadata={
            "name": "Indexer",
            "description": "The instance of ColBERTIndexer used to index the search corpus",
        },
    )
    query_encoder_model_name_or_path: str = field(
        default=None,
        metadata={
            "name": "Checkpoint",
            "description": "Path to checkpoint",
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
        default=5,
        metadata={
            "name": "Maximum number of retrieved documents",
            "range": [1, 100, 1],
            "api_support": True,
            "exclude_from_hash": True,
        },
    )

    def __post_init__(self):
        self.checkpoint=None
        if self.query_encoder_model_name_or_path is not None:
            self.checkpoint=self.query_encoder_model_name_or_path
        elif self.indexer is not None:
            self.query_encoder_model_name_or_path=self.indexer.ctx_encoder_model_name_or_path
            self.checkpoint=self.indexer.ctx_encoder_model_name_or_path
            
        self._config = DPRSearchArguments(
            #index_location=self.index_location,
            model_name_or_path=self.checkpoint,
            index_location=self.indexer.output_dir,
            top_k=self.max_num_documents,
        )

        self._searcher = DPRSearcher(
            self._config,
        )



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
        self._searcher = DPRSearcher(
            self._config,
        )
    def get_searcher(self):
        return self._searcher

    @classmethod
    def get_engine_type(cls):
        return "DPR"

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass

    def predict(self, input_texts: List[str], return_passages: bool, *args, **kwargs) -> Any:
        """Retrieves relevant documents based on input_texts

        Args:
            input_texts (List[str]): search queries

        Returns:
            Any: List of tuples. Each tuple contains a document indetifier and relevancy score
        """
        # Step 1: Locally update object variable values, if provided
        max_num_documents = (
            kwargs["max_num_documents"]
            if "max_num_documents" in kwargs
            else self.max_num_documents
        )

        retrieved_doc_ids, passages = self._searcher.search(
            list(input_texts), max_num_documents, mode="query_list"
        )
        if return_passages:
            return retrieved_doc_ids, passages
        
        retrieved_doc_ids = [list(map(int, doc_ids)) for doc_ids in retrieved_doc_ids]

        # returning: list (per query) of lists (per rank) of tuples (((int) docid, (float)score)
        return [
            list(zip(docids, scores))
            for docids, scores in zip(
                retrieved_doc_ids, [passage["scores"] for passage in passages]
            )
        ]
