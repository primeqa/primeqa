from typing import Union, List
from dataclasses import dataclass, field
import json

from primeqa.components.base import Indexer as BaseIndexer
from primeqa.ir.sparse.indexer import PyseriniIndexer


@dataclass
class BM25Indexer(BaseIndexer):
    """_summary_

    Args:

    Important:
    1. Each field has metadata property which can carry additional information for other downstream usages.
    2. Two special keys (api_support and exclude_from_hash) are defined in "metadata" property.
        a. api_support (bool, optional): If set to True, that parameter is exposed via service layer. Defaults to False.
        b. exclude_from_hash (bool,optional): If set to True, that parameter is not considered while building the hash representation for the object. Defaults to False.

    """

    num_workers: int = field(
        default=1,
        metadata={
            "name": "Number of worker threads",
        },
    )

    additional_index_args: str = field(
        default="--storePositions --storeDocvectors --storeRaw",
        metadata={
            "name": "Additional index arguments",
        },
    )

    def __post_init__(self):
        self._indexer = None

    def __hash__(self) -> int:
        return hash(
            f"{self.__class__.__name__}::{json.dumps({k: v.default for k, v in self.__class__.__dataclass_fields__.items() if not 'exclude_from_hash' in v.metadata or not v.metadata['exclude_from_hash']}, sort_keys=True)}"
        )

    def load(self, *args, **kwargs):
        self._index_path = f"{self.index_root}/{self.index_name}"
        self._indexer = PyseriniIndexer()

    def get_engine_type(self) -> str:
        return "BM25"

    def index(self, collection: Union[List[dict], str], *args, **kwargs):
        if not isinstance(collection, str):
            raise TypeError(
                "Pyserini indexer expects path to `documents.tsv` as value for `collection` argument."
            )

        self._indexer.index_collection(
            collection=collection,
            index_path=self._index_path,
            fieldnames=None,
            overwrite="overwrite" in kwargs and kwargs["overwrite"],
            threads=kwargs["num_workers"] if "num_workers" in kwargs else 1,
            additional_index_cmd_args=kwargs["additional_index_args"]
            if "additional_index_args" in kwargs
            else "--storePositions --storeDocvectors --storeRaw",
        )
