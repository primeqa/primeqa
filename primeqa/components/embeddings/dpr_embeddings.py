from typing import List, Dict
import os
from dataclasses import dataclass, field
import json
import numpy as np
import torch

from primeqa.components.base import Embeddings

from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizerFast,
)

@dataclass
class DPREmbeddings(Embeddings):
    """_summary_
    
    """
    
    model: str = field(
        default="PrimeQA/XOR-TyDi_monolingual_DPR_ctx_encoder",
        metadata={
            "name": "Model",
            "api_support": True,
            "description": "Path to model",
        },
    )
    
    max_doc_length: int = field(
        default=512,
        metadata={
            "name": "max_doc_length",
            "api_support": True,
            "description": "maximum document length (sub-word units)",
        },
    )
    
    
    def __post_init__(self):
        # Placeholder variables
        self._ctx_encoder = None
        self._ctx_tokenizer = None
        self._device = None
    
    
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
        self._ctx_tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(self.model)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._ctx_encoder = DPRContextEncoder.from_pretrained(self.model).to(device=self._device)
        

    def get_embeddings(self, documents: List[Dict],
                    *args,
                    **kwargs):
        """
        Takes in a list of texts .
            For each text returns a dict where the 'embeddings' element contains a vector of floats.
            
            Args:
                documents List[Dict]: For each query, a list of documents containing text and title
                each document is a dictionary with these elements:
                {
                        "text": "A man is eating food.",
                        "title": "food"
                }
            
            Returns:
                List[Dict]
                
                dict:
                  {
                      'embeddings': list[float]
                      'model': str
                  }
        """

        max_doc_length = (
            kwargs["max_doc_length"]
            if "max_num_documents" in kwargs
            else self.max_doc_length
        )

        texts = { "title": [ doc["title"] if doc["title"] is not None else "" for doc in documents], 
                  "text": [ doc["text"] for doc in documents]
                 }
        
        input_ids = self._ctx_tokenizer(
            texts["title"], texts["text"], truncation=True, padding="longest", return_tensors="pt", max_length=max_doc_length
        )["input_ids"]

        vectors = self._ctx_encoder(input_ids.to(device=self._device), return_dict=True).pooler_output
        vectors = vectors.detach().cpu().to(dtype=torch.float16).numpy()
        vectors = vectors.tolist()
        
        embeddings_list = []
        for vector in vectors:
            embeddings_list.append({
                "embeddings": vector,
                "model": self.model
            })
        return embeddings_list
        