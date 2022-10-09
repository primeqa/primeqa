from typing import Union, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(init=False, repr=False, eq=False)
class Component(ABC):
    @abstractmethod
    def load(self, *args, **kwargs):
        pass


@dataclass(init=False, repr=False, eq=False)
class ReaderComponent(Component):
    @abstractmethod
    def apply(self, input_texts: List[str], context: List[List[str]], *args, **kwargs):
        pass


@dataclass(init=False, repr=False, eq=False)
class IndexerComponent(Component):
    index_root: str = field(
        metadata={
            "name": "Index root",
            "description": "Path to root directory where index to be stored",
        },
    )
    index_name: str = field(
        metadata={
            "name": "Index name",
        },
    )

    @abstractmethod
    def index(self, collection: Union[List[dict], str], *args, **kwargs):
        pass


@dataclass(init=False, repr=False, eq=False)
class RetrieverComponent(Component):
    index_root: str = field(
        metadata={
            "name": "Index root",
            "description": "Path to root directory where index is stored",
        },
    )
    index_name: str = field(
        metadata={
            "name": "Index name",
        },
    )

    @abstractmethod
    def retrieve(self, input_texts: List[str], *args, **kwargs):
        pass
