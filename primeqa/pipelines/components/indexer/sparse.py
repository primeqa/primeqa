from typing import Union, List
from dataclasses import dataclass

from primeqa.pipelines.components.base import IndexerComponent


@dataclass
class BM25Indexer(IndexerComponent):
    def load(self, *args, **kwargs):
        pass

    def index(self, collection: Union[List[dict], str], *args, **kwargs):
        pass
