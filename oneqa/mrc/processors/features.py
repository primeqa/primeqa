import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, Hashable, List, Optional

# TODO get rid of these


@dataclass
class InputFeatures:
    # question: str
    # context: List[str]
    example_id: Hashable  # = field(default_factory=uuid.uuid4)
    context_idx: int
    window_idx: int
    model_inputs: Dict[str, Any] = field(default_factory=dict)
    targets: Optional[list['Target']] = None


@dataclass
class Target:
    position: 'Position'
    text: str
    y_n_answer: Optional[bool] = None


@dataclass
class Position:
    start: int
    end: int
    passage: int
