from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from transformers.file_utils import ModelOutput


@dataclass
class ExtractiveQAModelOutput(ModelOutput):
    """
    Extractive QA model outputs comprising:
    (loss), start_logits, end_logits, target_type_logits, (hidden_states), (attentions).
    """
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    target_type_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

@dataclass
class ExtractiveQAWithConfidenceModelOutput(ModelOutput):
    """
    Extractive QA model outputs comprising:
    (loss), start_logits, end_logits, target_type_logits,
    start_stdev, end_stdev, query_passage_similarity,
    (hidden_states), (attentions).
    """
    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    target_type_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    start_stdev: Optional[torch.FloatTensor] = None
    end_stdev: Optional[torch.FloatTensor] = None
    query_passage_similarity: Optional[torch.FloatTensor] = None
