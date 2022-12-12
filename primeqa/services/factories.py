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

from primeqa.pipelines.components.retriever.dense import ColBERTRetriever
from primeqa.pipelines.components.retriever.sparse import BM25Retriever

from primeqa.pipelines.components.indexer.dense import ColBERTIndexer
from primeqa.pipelines.components.indexer.sparse import BM25Indexer

READERS_REGISTRY = {
    ExtractiveReader.__name__: ExtractiveReader,
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

        # Step 2: Create instance
        try:
            instance = reader(**reader_kwargs)
        except TypeError as err:
            # Step 2.a: Log exception
            cls._logger.warning(
                "Failed to intialize %s with arguments: %s",
                reader.__name__,
                reader_kwargs,
            )

            # Step 2.b: Raise exception
            raise err

        # Step 3: Create hash based unique instance id
        instance_id = hash(instance)

        # Step 4: Load and persist instance
        if instance_id not in cls._instances:
            # Step 4.a: Check if class is currently loading
            if instance_id in cls._loading:
                raise ValueError(
                    f"{reader.__name__} is currently being loading. Please try again in a short while."
                )

            # Step 4.b: Add to loading
            cls._loading.append(instance_id)

            # Step 4.d: Start loading
            try:
                cls._logger.info(
                    "Loading '%s' reader with parameters = %s",
                    reader.__name__,
                    reader_kwargs,
                )
                start_t = time.time()
                instance.load(load_args, load_kwargs)
                cls._logger.info(
                    "'%s' reader - loading took %.2f seconds",
                    reader.__name__,
                    time.time() - start_t,
                )
            except OSError as err:
                # Step 3.d.i: Log exception
                cls._logger.warning(
                    "Failed to load %s with arguments: %s",
                    reader.__name__,
                    reader_kwargs,
                )

                # Step 3.d.ii: Remove reader from loading
                cls._loading.remove(instance_id)

                # Step 3.d.iii: Raise exception
                raise ValueError(err.args[0]) from err

            # Step 4.e: Remove from loading and add to available instances
            cls._instances[instance_id] = instance
            cls._loading.remove(instance_id)

        else:
            # Step 4.a: Instance with same instance_id already exists, clean up newly created instance
            del instance

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

        # Step 2: Create instance
        try:
            instance = retriever(**retriever_kwargs)
        except TypeError as err:
            # Step 2.a: Log exception
            cls._logger.warning(
                "Failed to intialize %s with arguments: %s",
                retriever.__name__,
                retriever_kwargs,
            )

            # Step 2.b: Raise exception
            raise err

        # Step 3: Create hash based unique instance id
        instance_id = hash(instance)

        # Step 4: Load and persist instance
        if instance_id not in cls._instances:
            # Step 4.a: Check if class is currently loading
            if instance_id in cls._loading:
                raise ValueError(
                    f"{retriever.__name__} is currently being loading. Please try again in a short while."
                )

            # Step 4.b: Add to loading
            cls._loading.append(instance_id)

            # Step 4.d: Start loading
            try:
                cls._logger.info(
                    "Loading '%s' retriever with parameters = %s",
                    retriever.__name__,
                    retriever_kwargs,
                )
                start_t = time.time()
                instance.load(load_args, load_kwargs)
                cls._logger.info(
                    "'%s' retriever - loading took %.2f seconds",
                    retriever.__name__,
                    time.time() - start_t,
                )
            except OSError as err:
                # Step 3.d.i: Log exception
                cls._logger.warning(
                    "Failed to load %s with arguments: %s",
                    retriever.__name__,
                    retriever_kwargs,
                )

                # Step 3.d.ii: Remove reader from loading
                cls._loading.remove(instance_id)

                # Step 3.d.iii: Raise exception
                raise ValueError(err.args[0]) from err

            # Step 4.e: Remove from loading and add to available instances
            cls._instances[instance_id] = instance
            cls._loading.remove(instance_id)

        else:
            # Step 4.a: Instance with same instance_id already exists, clean up newly created instance
            del instance

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

            # Step 3.c: Initialize instance
            cls._logger.info(
                "%s - initializing with arguments: %s", indexer.__name__, indexer_kwargs
            )
            try:
                instance = indexer(**indexer_kwargs)
            except TypeError as err:
                # Step 3.c.i: Log exception
                cls._logger.warning(
                    "Failed to intialize %s with arguments: %s",
                    indexer.__name__,
                    indexer_kwargs,
                )

                # Step 3.c.ii: Remove reader from loading
                cls._loading.remove(instance_id)

                # Step 3.c.iii: Raise exception
                raise err

            # Step 3.d: Load instance
            try:
                start_t = time.time()
                instance.load(load_args, load_kwargs)
                cls._logger.info(
                    "%s - loading took %.2f seconds",
                    indexer.__name__,
                    time.time() - start_t,
                )
            except OSError as err:
                # Step 3.d.i: Log exception
                cls._logger.warning(
                    "Failed to load %s with arguments: %s",
                    indexer.__name__,
                    indexer_kwargs,
                )

                # Step 3.d.ii: Remove reader from loading
                cls._loading.remove(instance_id)

                # Step 3.d.iii: Raise exception
                raise ValueError(err.args[0]) from err

            # Step 3.d: Remove from loading and add to available instances
            cls._instances[instance_id] = instance
            cls._loading.remove(instance_id)

        return cls._instances[instance_id]
