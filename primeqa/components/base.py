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
    
    
@dataclass(init=False, repr=False, eq=False)
class Reranker(Component):
    
    model: str = field(
        metadata={
            "name": "Model",
            "api_support": True,
            "description": "Path to model",
        },
    )

    max_num_documents: int = field(
        default=-1,
        metadata={
            "name": "Maximum number of retrieved documents",
            "range": [-1, 100, 1],
            "api_support": True,
            "exclude_from_hash": True,
        },
    )

    include_title: bool = field(
        default=True,
        metadata={
            "name": "Include Title",
            "description": "Whether to concatenate text and title",
            "choices": "True|False"
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
    def predict(self, queries: List[str], 
                    documents: List[List[Dict]],
                    *args, 
                    **kwargs):
        """
        Args:
            queries (List[str]): search queries
            texts (List[List[Dict]]): For each query, a list of documents to rerank
                where each document is a dictionary with the following structure:
                {
                    "document": {
                        "text": "A man is eating food.",
                        "document_id": "0",
                        "title": "food"
                    },
                    "score": 1.4
                }
        
        Returns:
            List[List[Dict]] For each query a list of reranked documents in the same 
            structure as the input documents with the score replace with the reranker score.
        """
        pass
    
    
@dataclass(init=False, repr=False, eq=False)
class Embeddings(Component):
    
    model: str = field(
            metadata={
                "name": "Model. This could be either a query or context DPR encoder model.",
                "api_support": True,
                "description": "Path to model",
            },
        )
        
    max_doc_length: int = field(
            default=512,
            metadata={
                "name": "max_doc_length",
                "api_support": True,
                "description": "maximum document length (sub-word units)",
            },
        )
    
    batch_size: int = field(
        default=128,
        metadata={
            "name": "batch_size",
            "api_support": False,
            "description": "batch size",
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
    def get_embeddings(self, input_texts: List[str], 
                        *args, 
                        **kwargs):
        """
            Takes in a list of texts .
            For each text returns a dict where the 'embeddings' element contains a vector of floats.
            
            Args:
                input_texts List[Dict]: For each query, a list of input texts containing text and title
                each document is a dictionary with these elements:
                {
                        "text": "A man is eating food.",
                        "title": "food"
                }
            
            Optional Args:
                max_doc_length int: Default 512 maximum document length (sub-word units)
                batch_size int: Default 128 batch size
            
            Returns:
                List[Dict]
                
                dict:
                  {
                      'embeddings': list[float]
                      'model': str
                  }
        """
        pass
        

    def train(self, *args, **kwargs):
        pass

    def eval(self, *args, **kwargs):
        pass

    def predict(self, *args, **kwargs):
        pass