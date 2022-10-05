from typing import List
from primeqa.pipelines.base import (
    Pipeline,
    ReaderPipeline,
    RetrieverPipeline,
    IndexerPipeline,
)
from primeqa.pipelines.reader_pipeline import ExtractiveReader
from primeqa.pipelines.dense_retriever_pipeline import ColBERTRetriever
from primeqa.pipelines.dense_indexer_pipeline import ColBERTIndexer

pipeline_registry = {
    "ExtractiveReader": ExtractiveReader,
    "ColBERTRetriever": ColBERTRetriever,
    "ColBERTIndexer": ColBERTIndexer,
}

_pipelines = {k: v() for k, v in pipeline_registry.items()}
_loaded_pipelines = set()


def get_pipelines() -> List[dict]:
    return _pipelines.values()


def get_pipeline(pipeline_id: str, invoke__init__: bool = False) -> Pipeline:
    """
    Get `Pipeline` object.

    Args:
        pipeline_id (str): Unique identifier
        invoke__init__ (bool, optional): If True, creates and returns a new object. Defaults to False.

    Returns:
        Pipeline: `Pipeline` object
    """
    if invoke__init__:
        return pipeline_registry[pipeline_id]()
    return _pipelines[pipeline_id]


def load_pipeline(pipeline_id: str, *args, **kwargs):
    if pipeline_id not in _loaded_pipelines:
        # Step 1: Load pipeline
        _pipelines[pipeline_id].load(*args, **kwargs)

        # Step 2: Add it set of active pipelines
        _loaded_pipelines.add(pipeline_id)

    return _loaded_pipelines[pipeline_id]
