from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
import torch


@dataclass
class ORQADataCollator:

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features, return_tensors=None):
        import numpy as np

        if return_tensors is None:
            return_tensors = self.return_tensors

#        print(features, flush=True)
        batch_features = {}
        for feature in features:
            for k,v in feature.items(): 
                if k not in batch_features:
                    batch_features[k] = []
                batch_features[k].append(v)
        for k,v in batch_features.items():
            try: # not converting string features such as "query and "example_id" FIXME test this
                batch_features[k] = torch.tensor(v) # convert to tensor
            except:
                continue

        return batch_features
    
