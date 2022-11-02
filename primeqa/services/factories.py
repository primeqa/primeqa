import logging
import time
import json

from dataclasses import MISSING

from primeqa.pipelines.components.base import (
    ReaderComponent,
    RetrieverComponent,
    IndexerComponent,
)
from primeqa.pipelines.components.reader.extractive import ExtractiveReader
from primeqa.pipelines.components.reader.generative import GenerativeReader

from primeqa.pipelines.components.retriever.dense import ColBERTRetriever
from primeqa.pipelines.components.retriever.sparse import BM25Retriever

from primeqa.pipelines.components.indexer.dense import ColBERTIndexer
from primeqa.pipelines.components.indexer.sparse import BM25Indexer

READERS_REGISTRY = {
    ExtractiveReader.__name__: ExtractiveReader,
    GenerativeReader.__name__: GenerativeReader,
}

RETRIEVERS_REGISTRY = {
    ColBERTRetriever.__name__: ColBERTRetriever,
    BM25Retriever.__name__: BM25Retriever,
}

INDEXERS_REGISTRY = {
    ColBERTIndexer.__name__: ColBERTIndexer,
    BM25Indexer.__name__: BM25Indexer,
}


def validate(fields: dict):
    missing_fields = [
        field_name
        for field_name, field_value in fields.items()
        if field_value == MISSING
    ]
    if missing_fields:
        raise ValueError(f"Value must be defined for {', '.join(missing_fields)}")


class ReaderFactory:
    _instances = {}
    _loading = []
    _logger = logging.getLogger("ReaderFactory")

    @classmethod
    def get(
        cls, reader: ReaderComponent, reader_kwargs: dict, *load_args, **load_kwargs
    ):
        # Step 1: Validate all required fields are specified
        validate(reader_kwargs)

        # Step 2: Create unique hash based on reader's class name and keyword arguments
        instance_id = hash(
            f"{reader.__name__}::{json.dumps(reader_kwargs, sort_keys=True)}"
        )
        if instance_id not in cls._instances:
            # Step 3.a: Check if class is currently loading
            if instance_id in cls._loading:
                raise ValueError(
                    f"{reader.__name__} is currently being loading. Please try again in a short while."
                )

            # Step 3.b: Add to loading
            cls._loading.append(instance_id)

            # Step 3.c: Start creating instance
            cls._logger.info(
                "%s - initializing with arguments: %s", reader.__name__, reader_kwargs
            )
            instance = reader(**reader_kwargs)
            start_t = time.time()
            instance.load(load_args, load_kwargs)
            cls._logger.info(
                "%s - loading took %.2f seconds",
                reader.__name__,
                time.time() - start_t,
            )

            # Step 3.d: Remove from loading and add to available instances
            cls._instances[instance_id] = instance
            cls._loading.remove(instance_id)

        return cls._instances[instance_id]


class RetrieverFactory:
    _instances = {}
    _loading = []
    _logger = logging.getLogger("RetrieverFactory")

    @classmethod
    def get(
        cls,
        retriever: RetrieverComponent,
        retriever_kwargs: dict,
        *load_args,
        **load_kwargs,
    ):
        # Step 1: Validate all required fields are specified
        validate(retriever_kwargs)

        # Step 2: Create unique hash based on retriver's class name and keyword arguments
        instance_id = hash(
            f"{retriever.__name__}::{json.dumps(retriever_kwargs, sort_keys=True)}"
        )

        if instance_id not in cls._instances:
            # Step 3.a: Check if class is currently loading
            if instance_id in cls._loading:
                raise ValueError(
                    f"{retriever.__name__} is currently being loading. Please try again in a short while."
                )

            # Step 3.b: Add to loading
            cls._loading.append(instance_id)

            # Step 3.c: Start creating instance
            cls._logger.info(
                "%s - initializing with arguments: %s",
                retriever.__name__,
                retriever_kwargs,
            )
            instance = retriever(**retriever_kwargs)
            start_t = time.time()
            instance.load(load_args, load_kwargs)
            cls._logger.info(
                "%s - loading took %.2f seconds",
                retriever.__name__,
                time.time() - start_t,
            )

            # Step 3.d: Remove from loading and add to available instances
            cls._instances[instance_id] = instance
            cls._loading.remove(instance_id)

        return cls._instances[instance_id]


class IndexerFactory:
    _instances = {}
    _loading = []
    _logger = logging.getLogger("IndexerFactory")

    @classmethod
    def get(
        cls,
        indexer: IndexerComponent,
        indexer_kwargs: dict,
        *load_args,
        **load_kwargs,
    ):
        # Step 1: Validate all required fields are specified
        validate(indexer_kwargs)

        # Step 2: Create unique hash based on indexer's class name and keyword arguments
        instance_id = hash(
            f"{indexer.__name__}::{json.dumps(indexer_kwargs, sort_keys=True)}"
        )

        if instance_id not in cls._instances:
            # Step 3.a: Check if class is currently loading
            if instance_id in cls._loading:
                raise ValueError(
                    f"{indexer.__name__} is currently being loading. Please try again in a short while."
                )

            # Step 3.b: Add to loading
            cls._loading.append(instance_id)

            # Step 3.c: Start creating instance
            cls._logger.info(
                "%s - initializing with arguments: %s", indexer.__name__, indexer_kwargs
            )
            instance = indexer(**indexer_kwargs)
            start_t = time.time()
            instance.load(load_args, load_kwargs)
            cls._logger.info(
                "%s - loading took %.2f seconds",
                indexer.__name__,
                time.time() - start_t,
            )

            # Step 3.d: Remove from loading and add to available instances
            cls._instances[instance_id] = instance
            cls._loading.remove(instance_id)

        return cls._instances[instance_id]
