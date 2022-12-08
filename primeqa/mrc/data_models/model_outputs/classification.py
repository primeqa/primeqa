from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.file_utils import ModelOutput


#dataclass
class ClassificationModelOutput(ModelOutput):
    """
    Classification model output
    (loss), target_type_logits
    """
    loss: Optional[torch.FloatTensor] = None
    target_type_logits: torch.FloatTensor = None

