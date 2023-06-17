from typing import List, Dict
import os
from dataclasses import dataclass, field
import json
import numpy as np
import warnings
import torch.nn.functional as F

from primeqa.components.base import Reranker as BaseReranker
from transformers import AutoTokenizer, AutoModelForSequenceClassification


@dataclass
class SeqClassificationReranker(BaseReranker):
    """_summary_

    This implementation is based on https://github.com/IBM/kgi-slot-filling/blob/re2g/reranker/reranker_apply.py

    Args:
        model (str, optional): Model to load. 
        max_num_documents (int, optional): Maximum number of reranked document to return. Defaults to -1.
        include_title (bool, optional): Whether to concatenate text and title. Defaults to True
        max_batch_size: (int, optional): Defaults to 128
        max_seq_len: (int, optional): Maximum length of question and context inputs to the model (in word pieces/bpes). Defaults to 512.
        

    Important:
    1. Each field has the metadata property which can carry additional information for other downstream usages.
    2. Two special keys (api_support and exclude_from_hash) are defined in "metadata" property.
        a. api_support (bool, optional): If set to True, that parameter is exposed via the service layer. Defaults to False.
        b. exclude_from_hash (bool,optional): If set to True, that parameter is not considered while building the hash representation for the object. Defaults to False.

    Returns:
        _type_: _description_

    """

    model: str = field(
        default="ibm/re2g-reranker-nq",
        metadata={
            "name": "Model",
            "api_support": True,
            "description": "Path to model",
        },
    )

    max_batch_size: int = field(
        default=128,
        metadata={
            "name": "Maximum batch size",
            "range": [1, 256, 8],
            "api_support": True,
            "exclude_from_hash": True,
        },
    )
    max_seq_len: int = field(
        default=512,
        metadata={
            "name": "Maximum sequence length",
            "description": "Maximum length of question and context inputs to the model (in word pieces/bpes)",
            "range": [32, 512, 8],
        },
    )

    def __post_init__(self):
        # Placeholder variables
        self._laoded_model = None
        self._tokenizer = None

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
        self._tokenizer = AutoTokenizer.from_pretrained(self.model)
        self._loaded_model = AutoModelForSequenceClassification.from_pretrained(self.model)
    
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
        for query, docs  in zip(queries,documents):
            texts_a = [query] * len(docs)
            texts_b = []
            for p in docs:
                if include_title and 'title' in p['document'] and p['document']['title'] is not None and len(p['document']['title'].strip()) > 0:
                    texts_b.append(p['document']['title'] + '\n\n' + p['document']['text'])
                else:
                    texts_b.append(p['document']['text'])
                
            scores = []
            for start_ndx in range(0, len(texts_a), self.max_batch_size):
                inputs = self._tokenizer(
                    texts_a[start_ndx:start_ndx+self.max_batch_size],
                    texts_b[start_ndx:start_ndx+self.max_batch_size],
                    add_special_tokens=True,
                    return_tensors='pt',
                    max_length=self.max_seq_len,
                    padding='longest',
                    truncation=True)
                inputs = {n: t.to(self._loaded_model.device) for n, t in inputs.items()}
                outputs = self._loaded_model(**inputs).logits.detach().cpu()
                s = outputs.shape[1] - 1
                probs = F.softmax(outputs, dim=s)[:,s].numpy().tolist()
                scores.extend(probs)
            ranked_passage_indexes = np.array(scores).argsort()[::-1][:max_num_documents if max_num_documents > 0 else len(scores)].tolist()
            results = []
            for idx in ranked_passage_indexes:
                docs[idx]['score'] = scores[idx]
                results.append(docs[idx])
            ranking_results.append(results)
        return ranking_results
    