from typing import List, Dict
import os
from dataclasses import dataclass, field
import json
import numpy as np
import warnings

from primeqa.components.base import Reranker as BaseReranker
from primeqa.ir.dense.dpr_top.dpr.config import DPRSearchArguments
from primeqa.ir.dense.dpr_top.dpr.searcher import DPRSearcher



from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.searcher import Searcher

@dataclass
class DPRReranker(BaseReranker):
    """_summary_

    Args:
        model (str, optional): Model to load.
        max_num_documents (int, optional): Maximum number of reranked document to return. Defaults to -1.
        include_title (bool, optional): Whether to concatenate text and title. Defaults to True

    Important:
    1. Each field has the metadata property which can carry additional information for other downstream usages.
    2. Two special keys (api_support and exclude_from_hash) are defined in "metadata" property.
        a. api_support (bool, optional): If set to True, that parameter is exposed via the service layer. Defaults to False.
        b. exclude_from_hash (bool,optional): If set to True, that parameter is not considered while building the hash representation for the object. Defaults to False.

    Returns:
        _type_: _description_

    """

    '''checkpoint: str = field(
        default=None,
        metadata={
            "name": "Checkpoint",
            "description": "Path to checkpoint",
        },
    )
    
    checkpoint: str = field(
        default=None,
        metadata={
            "name": "Checkpoint",
            "description": "Path to checkpoint",
        },
    )
    '''
    def __post_init__(self):
        self._config = DPRSearchArguments(
            index_location=None,
            rescore_only = True,
            qry_encoder_name_or_path=os.path.join(self.model, 'qry_encoder'),
            ctx_encoder_name_or_path=os.path.join(self.model, 'ctx_encoder')
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
        self._searcher = DPRSearcher(
            self._config,
        )

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass

    def predict(self, queries: List[str],
                    documents:  List[List[Dict]],
                    *args,
                    **kwargs):
        warnings.warn("The 'predict' method is deprecated. Please use `rerank'", FutureWarning)
        return self.rerank(queries, documents, *args, **kwargs)

    def rerank(self, queries: List[str],
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
            List[List[Dict]] A list of reranked documents in the same structure as the input documents
             with the score replaced with the reranker score for each query.
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

            scores = self._searcher.rescore(query, texts).tolist()
            ranked_passage_indexes = np.array(scores).argsort()[::-1][:max_num_documents if max_num_documents > 0 else len(scores)].tolist()

            results = []
            for idx in ranked_passage_indexes:
                docs[idx]['score'] = scores[idx]
                results.append(docs[idx])
            ranking_results.append(results)

        return ranking_results
