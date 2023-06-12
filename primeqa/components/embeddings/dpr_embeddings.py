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
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizerFast,
    AutoConfig
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
    
    batch_size: int = field(
        default=128,
        metadata={
            "name": "batch_size",
            "api_support": False,
            "description": "batch size",
        },
    )
    
    
    def __post_init__(self):
        # Placeholder variables
        self._encoder = None
        self._tokenizer = None
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
        
        config = AutoConfig.from_pretrained(self.model)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if 'DPRQuestionEncoder' in config.get_config_dict(self.model)[0]['architectures']:
            self._tokenizer = DPRQuestionEncoderTokenizerFast.from_pretrained(self.model)
            self._encoder = DPRQuestionEncoder.from_pretrained(self.model).to(device=self._device)
        else:
            self._tokenizer = DPRContextEncoderTokenizerFast.from_pretrained(self.model)
            self._encoder = DPRContextEncoder.from_pretrained(self.model).to(device=self._device)
        self._encoder.eval()
        

    def get_embeddings(self, input_texts: List[Dict],
                    *args,
                    **kwargs):
        """
        Takes in a list of texts .
            For each text returns a dict where the 'embeddings' element contains a vector of floats.
            
            Args:
                input_texts List[Dict]: For each query, a list of documents containing text and title
                each document is a dictionary with these elements:
                {
                        "text": "A man is eating food.",
                        "title": "food"
                }
                max_doc_length int: 
                batch_size int: default 128
                
            
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
        
        batch_size = (
            kwargs["batch_size"]
            if "batch_size" in kwargs
            else self.batch_size
        )
        
       
        embeddings_list = []
        for i in range(0, len(input_texts), batch_size):
            
            texts = { "title": [ doc["title"] if doc["title"] is not None else "" for doc in input_texts[i:i+batch_size]], 
                  "text": [ doc["text"] for doc in input_texts[i:i+batch_size]]
                 }
        
            input_ids = self._tokenizer(
                texts["title"], texts["text"], truncation=True, padding="longest", return_tensors="pt", max_length=max_doc_length
            )["input_ids"]

            vectors = self._encoder(input_ids.to(device=self._device), return_dict=True).pooler_output
            vectors = vectors.detach().cpu().to(dtype=torch.float16).numpy()
            vectors = vectors.tolist()
        
            for vector in vectors:
                embeddings_list.append({
                    "embeddings": vector,
                    "model": self.model
                })
        return embeddings_list
    
    
    
        