from typing import List, Union
from abc import ABC, abstractmethod


class Pipeline(ABC):
    @abstractmethod
    def load(self, *args, **kwargs):
        pass


class ReaderPipeline(Pipeline):
    @abstractmethod
    def apply(self, input_texts: List[str], context: List[List[str]] = None, **kwargs):
        pass


class IndexerPipeline(Pipeline):
    @abstractmethod
    def index(
        self, documents: Union[List[dict], str], index_path: str, *args, **kwargs
    ):
        pass


class RetrieverPipeline(Pipeline):
    @abstractmethod
    def retrieve(self, input_texts: List[str], *args, **kwargs):
        pass
