from typing import List, Dict
import os
from dataclasses import dataclass, field
import json
import numpy as np

from primeqa.components.base import Reranker as BaseReranker
from sentence_transformers import CrossEncoder



@dataclass
class STCrossEncoderReranker(BaseReranker):
    """_summary_

    Args:
        model (str, optional): Model to load. 
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

    model: str = field(
        default="cross-encoder/ms-marco-TinyBERT-L-2",
        metadata={
            "name": "Model",
            "api_support": True,
            "description": "Path to model",
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
    include_title: bool = field(
        default=True,
        metadata={
            "name": "Include Title",
            "description": "Whether to concate text and title",
            "choices": "True|False"
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
        self._loaded_model =  CrossEncoder(self.model)


    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass
    
    def predict(self, queries: List[str], 
                    documents:  List[List[Dict]],
                    *args, 
                    **kwargs):
        """
        Args:
            queries (List[str]): search queries
            texts (List[List[Dict]]): For each query, a list of documents to rerank
                where each document is a dictionary with the following structure:
                {
                    "document": {
                        "text": "A man is eating food.",
                        "document_id": "0",
                        "title": "food"
                    },
                    "score": 1.4
                }
        
        Returns:
            List[List[Dict]] For each query a list of reranked documents in the same 
            structure as the input documents with the score replace with the reranker score.
        """
        # Step 1: Locally update object variable values, if provided
        max_num_documents = (
            kwargs["max_num_documents"]
            if "max_num_documents" in kwargs
            else self.max_num_documents
        )
        
        
        include_title = (
            kwargs["include_title"]
            if "include_title" in kwargs
            else self.include_title
        )
        
        ranking_results = []
        for query, docs in zip(queries, documents):
            texts = []                
            for p in docs:
                if include_title and 'title' in p['document'] and len(p['document']['title'].strip()) > 0:
                    texts.append(p['document']['title'] + '\n\n' + p['document']['text'])
                else:
                    texts.append(p['document']['text'])
                    
            model_inputs = [[query, text] for text in texts]
            
            scores = self._loaded_model.predict(model_inputs).tolist()
            ranked_passage_indexes = np.array(scores).argsort()[::-1][:max_num_documents].tolist()
            
            results = []
            for idx in ranked_passage_indexes:
                docs[idx]['score'] = scores[idx]
                results.append(docs[idx])
            ranking_results.append(results)
            
        return ranking_results
    