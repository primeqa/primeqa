from typing import List
from dataclasses import dataclass

from primeqa.pipelines.components.base import ReaderComponent


@dataclass
class GenerativeReader(ReaderComponent):
    """_summary_

    Args:

    Returns:
        _type_: _description_
    """

    def __post_init__(self):
        # Placeholder variables
        self._preprocessor = None
        self._trainer = None

    def load(self, *args, **kwargs):
        pass

    def apply(self, input_texts: List[str], context: List[List[str]], *args, **kwargs):
        pass
