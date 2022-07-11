from dataclasses import dataclass,field
from typing import Optional, List, Dict
import torch


@dataclass
class T2TDataCollator:
    def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
        """
        Take a list of samples from a Dataset and collate them into a batch.
        Returns:
            A dictionary of tensors
        """
        input_ids = torch.stack([example['input_ids'] for example in batch])
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
        

        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'labels': lm_labels, 
            'decoder_attention_mask': decoder_attention_mask
        }