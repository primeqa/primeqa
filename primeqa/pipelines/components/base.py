from typing import Union, List
from abc import ABC, abstractmethod


class Component(ABC):
    @abstractmethod
    def load(self, *args, **kwargs):
        pass


class ReaderComponent(Component):
    @abstractmethod
    def apply(self, input_texts: List[str], context: List[List[str]], *args, **kwargs):
        pass


class IndexerComponent(Component):
    @abstractmethod
    def index(self, collection: Union[List[dict], str], *args, **kwargs):
        pass


class RetrieverComponent(Component):
    @abstractmethod
    def retrieve(self, input_texts: List[str], *args, **kwargs):
        pass
