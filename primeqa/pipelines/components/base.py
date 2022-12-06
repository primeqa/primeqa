from typing import Union, List, Dict
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
    def __hash__(self) -> int:
        """
        Custom hashing function useful to compare instances of `ReaderComponent`.

        Raises:
            NotImplementedError:

        Returns:
            int: hash value
        """
        raise NotImplementedError

    @abstractmethod
<<<<<<< HEAD
    def predict(self, questions: List[str], contexts: List[List[str]], example_ids: List[str] = None, *args, **kwargs) -> Dict[str,List[Dict]]:
=======
    def apply(self, questions: List[str], contexts: List[List[str]], example_ids: List[str] = None, *args, **kwargs) -> Dict[str,List[Dict]]:
>>>>>>> ed8009f8 (apply returns a dict with example id as key)
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
    collection: str = field(
        metadata={
            "name": "The corpus file split in paragraphs",
        },
    )

    @abstractmethod
    def __hash__(self) -> int:
        """
        Custom hashing function useful to compare instances of `RetrieverComponent`.

        Raises:
            NotImplementedError:

        Returns:
            int: hash value
        """
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, input_texts: List[str], *args, **kwargs):
        pass
