from typing import Union, List,Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, field



@dataclass(init=False, repr=False, eq=False)
class Component(ABC):
    config: str = field(
        metadata={
            "name": "config path",
            "description": "Path to config json file",
        },
    )
    @abstractmethod
    def load(self, *args, **kwargs):
        pass
    @abstractmethod
    def predict(self, *args, **kwargs):
        pass
    @abstractmethod
    def train(self, *args, **kwargs):
        pass
    @abstractmethod
    def eval(self, *args, **kwargs):
        pass

@dataclass(init=False, repr=False, eq=False)
class Reader(Component):
    @abstractmethod
    def predict(self, questions: List[str], contexts: List[List[Any]], *args, **kwargs):
        pass
    


#Todo: Revisit Indexer with martin and others
@dataclass(init=False, repr=False, eq=False)
class Indexer(Component):
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
class Retriever(Component):
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
