from dataclasses import dataclass,field
from typing import Optional, List, Dict
import torch


@dataclass
class TapasCollator:
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        labels = torch.stack([example['labels'] for example in batch])
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        token_type_ids = torch.stack([example['token_type_ids'] for example in batch])
        numeric_values = torch.stack([example['numeric_values'] for example in batch])
        numeric_values_scale = torch.stack([example['numeric_values_scale'] for example in batch])

        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'labels': labels, 
            'token_type_ids': token_type_ids,
            'numeric_values': numeric_values,
            'numeric_values_scale':numeric_values_scale,

        }