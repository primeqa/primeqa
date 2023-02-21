from typing import Union, List, Dict
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass(init=False, repr=False, eq=False)
class Component(ABC):
    config: str = field(
        default=None,
        init=False,
        metadata={
            "name": "config path",
            "description": "Path to config json file",
        },
    )

    @abstractmethod
    def load(self, *args, **kwargs):
        pass

    @abstractmethod
    def train(self, *args, **kwargs):
        pass

    @abstractmethod
    def eval(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, *args, **kwargs):
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

    @classmethod
    def get_engine_type(cls) -> str:
        """
        Return this retriever engine type. Must match with the indexer used to generate the index.

        Raises:
            NotImplementedError:

        Returns:
            str: engine type
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, input_texts: List[str], *args, **kwargs):
        pass


@dataclass(init=False, repr=False, eq=False)
class Indexer:
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
    def load(self, *args, **kwargs):
        pass

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
    def get_engine_type(self) -> str:
        """
        Return this retriever engine type. Must match with the retriever that will be used to query the index.

        Raises:
            NotImplementedError:

        Returns:
            str: engine type
        """
        raise NotImplementedError

    @abstractmethod
    def index(self, collection: Union[List[dict], str], *args, **kwargs):
        pass
