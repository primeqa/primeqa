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
            "exclude_from_hash": True,
        },
    )
    
    batch_size: int = field(
        default=128,
        metadata={
            "name": "batch_size",
            "api_support": False,
            "description": "batch size",
            "exclude_from_hash": True,
        },
    )
    
    embeddings_format: str = field(
        default=None,
        metadata={
            "name": "embeddings_format",
            "api_support": False,
            "description": "embeddings_format, Choices: 'pt', 'np' - Default None returns vector as a list of floats",
            "choices": "'pt'|'np'| None",
            "exclude_from_hash": True,
        }
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
        

    def get_embeddings(self, input_texts: List[str],
                    *args,
                    **kwargs):
        """
        Takes in a list of texts .
            For each text returns a dict where the 'embeddings' element contains a vector of floats.
            
            Args:
                input_texts List[str]: list of texts to be encoded
                max_doc_length int: 
                batch_size int: default 128
                embeddings_format: 
                    Default None (list of floats), choices 'pt' (tensors), 'np' (numpy array), None
                
            
            Returns:
                Dict
                  {
                      'embeddings': List[vectors]
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
        
        embeddings_format = (
            kwargs["embeddings_format"]
            if "embeddings_format" in kwargs
            else self.embeddings_format
        )
       
        
        embeddings_list = []
        for i in range(0, len(input_texts), batch_size):
                        
            texts = input_texts[i:i+batch_size]
        
            input_ids = self._tokenizer(
                texts, truncation=True, padding="longest", return_tensors="pt", max_length=max_doc_length
            )["input_ids"]

            vectors = self._encoder(input_ids.to(device=self._device), return_dict=True).pooler_output
            
            if embeddings_format == 'pt':
                vectors = vectors.detach().cpu().to(dtype=torch.float16)
                embeddings_list.extend(vectors)
            elif embeddings_format == 'np':
                vectors = vectors.detach().cpu().to(dtype=torch.float16).numpy()
                embeddings_list.extend(vectors)
            else:
                vectors = vectors.detach().cpu().to(dtype=torch.float16).numpy().tolist()
                embeddings_list.extend(vectors)
        return {
            "embeddings": embeddings_list,
            "model": self.model
        }
    
    
    
        