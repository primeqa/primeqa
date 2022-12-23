from typing import Union, List
from dataclasses import dataclass

from primeqa.Components.base import Indexer


@dataclass
class BM25Indexer(Indexer):
    """_summary_

    Args:

    Important:
    1. Each field has metadata property which can carry additional information for other downstream usages.
    2. Two special keys (api_support and exclude_from_hash) are defined in "metadata" property.
        a. api_support (bool, optional): If set to True, that parameter is exposed via service layer. Defaults to False.
        b. exclude_from_hash (bool,optional): If set to True, that parameter is not considered while building the hash representation for the object. Defaults to False.

    """

    def load(self, *args, **kwargs):
        pass

    def index(self, collection: Union[List[dict], str], *args, **kwargs):
        pass
