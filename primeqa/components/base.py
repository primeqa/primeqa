from typing import Union, List, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(init=False, repr=False, eq=False)
class Component(ABC):
    @abstractmethod
    def load(self, *args, **kwargs):
        pass
    def train(self, *args, **kwargs):
        pass
    def predict(self, *args, **kwargs):
        pass
    def eval(self, *args, **kwargs):
        pass


@dataclass(init=False, repr=False, eq=False)
class Reader(Component):
    @abstractmethod
    def __hash__(self) -> int:
        """
        Custom hashing function useful to compare instances of `Reader`.

        Raises:
            NotImplementedError:

        Returns:
            int: hash value
        """
        raise NotImplementedError

    @abstractmethod
    def predict(
        self,
        questions: List[str],
        contexts: List[List[str]],
        *args,
        example_ids: List[str] = None,
        **kwargs
    ) -> Dict[str, List[Dict]]:
        pass


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
    
    @abstractmethod    
    def get_engine_type() -> str:
        """
        Return this retriever engine type. Must match with the retriever tha will be used to query the index.

        Raises:
            NotImplementedError:

        Returns:
            str: engine type
        """
        raise NotImplementedError


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
    collection: str = field(
        metadata={
            "name": "The corpus file split in paragraphs",
        },
    )

    @abstractmethod
    def __hash__(self) -> int:
        """
        Custom hashing function useful to compare instances of `Retriever`.

        Raises:
            NotImplementedError:

        Returns:
            int: hash value
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, input_texts: List[str], *args, **kwargs):
        pass
    
    @abstractmethod
    def get_engine_type() -> str:
        """
        Return this retriever engine type. Must match with the indexer used to generate the index.

        Raises:
            NotImplementedError:

        Returns:
            str: engine type
        """
        raise NotImplementedError
