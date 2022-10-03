from typing import List
from primeqa.pipelines.base import Pipeline, ReaderPipeline, RetrieverPipeline
from primeqa.pipelines.reader_pipeline import ExtractiveReader
from primeqa.pipelines.dense_retriever_pipeline import ColBERTRetriever

_pipelines = {
    "ExtractiveReader": ExtractiveReader(),
    "ColBERTRetriever": ColBERTRetriever(),
}
_active_pipelines = set()


def get_pipelines() -> List[dict]:
    return _pipelines.values()


def get_pipeline(pipeline_id: str) -> Pipeline:
    return _pipelines[pipeline_id]


def activate_pipeline(pipeline_id: str):
    if pipeline_id not in _active_pipelines:
        # Step 1: Load pipeline
        _pipelines[pipeline_id].load()

        # Step 2: Add it set of active pipelines
        _active_pipelines.add(pipeline_id)
