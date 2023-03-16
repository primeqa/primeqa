from typing import List, Any
import os
from dataclasses import dataclass, field
import json
import numpy as np

from primeqa.components.base import Reranker as BaseReranker
from sentence_transformers import CrossEncoder



@dataclass
class CrossEncoderReranker(BaseReranker):
    """_summary_

    Args:
        checkpoint (str, optional): Model to load. 
        collection (str, optional): collection to load. Defaults to collection in index configuration.
        max_num_documents (int, optional): Maximum number of reranked document to return. Defaults to 5.

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

    def __post_init__(self):
        # Placeholder variables
        self._model = None

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
        self._model =  CrossEncoder(self.checkpoint)

    @classmethod
    def get_engine_type(cls):
        return "CrossEncoder"

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass
    
    def predict(self, queries: List[str], 
                    doc_indexes:  List[List[int]],
                    texts: List[List[str]],
                    *args, 
                    **kwargs):
        """
        Args:
            queries (List[str]): search queries
            texts (List[List[str]]): list of texts to rerank per query
            doc_indexes:  List[List[int]]

        Returns:
            Any: List of tuples. Each tuple contains a document identifier  and relevancy score
        """
        # Step 1: Locally update object variable values, if provided
        max_num_documents = (
            kwargs["max_num_documents"]
            if "max_num_documents" in kwargs
            else self.max_num_documents
        )
        
        ranking_results = []
        for query, passages in zip(queries, texts):
            model_inputs = [[query, passage] for passage in passages]
            scores = self._model.predict(model_inputs).tolist()
            ranked_passage_indexes = np.array(scores).argsort()[::-1][:max_num_documents].tolist()
            
            results = []
            for idx in ranked_passage_indexes:
                results.append( (idx, scores[idx]) )
            ranking_results.append(results)
        return ranking_results
    