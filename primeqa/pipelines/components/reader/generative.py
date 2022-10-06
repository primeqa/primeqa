import logging
from typing import List
from dataclasses import dataclass

from primeqa.pipelines.components.base import ReaderComponent


@dataclass
class GenerativeReader(ReaderComponent):
    """_summary_

    Args:
        logger (logging.Logger, optional): logger object. Defaults to logging.getLogger(GenerativeReader).


    Returns:
        _type_: _description_
    """

    logger: logging.Logger = logging.getLogger("GenerativeReader")

    def __post_init__(self):
        self.name = "Generative Reader"
        self.type = ReaderComponent.__name__

        # Placeholder variables
        self._preprocessor = None
        self._trainer = None

    def load(self, *args, **kwargs):
        pass

    def apply(self, input_texts: List[str], context: List[List[str]], *args, **kwargs):
        pass
