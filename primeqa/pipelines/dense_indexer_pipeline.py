import logging
from typing import Union, List
import time

from primeqa.pipelines.base import IndexerPipeline
from primeqa.ir.dense.colbert_top.colbert.infra.config import ColBERTConfig
from primeqa.ir.dense.colbert_top.colbert.indexer import Indexer


class ColBERTIndexer(IndexerPipeline):
    """
    Indexer: ColBERT
    """

    def __init__(
        self,
        logger: Union[logging.Logger, None] = None,
    ) -> None:
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger

        # Default object variables
        self.pipeline_id = self.__class__.__name__
        self.pipeline_name = "Dense Indexer (ColBERT)"
        self.pipeline_description = ""
        self.pipeline_type = IndexerPipeline.__name__

        self.parameters = {
            "model": {
                "parameter_id": "model",
                "name": "Model",
                "type": "String",
                "value": "colbert",
                "options": ["colbert", "DrDecr"],
            },
            "similarity": {
                "parameter_id": "similarity",
                "name": "Similarity metric",
                "type": "String",
                "value": "cosine",
                "options": ["cosine", "l2"],
            },
            "dim": {
                "parameter_id": "dim",
                "name": "Dimension",
                "type": "Numeric",
                "value": 128,
                "range": [32, 512, 32],
            },
            "query_maxlen": {
                "parameter_id": "query_maxlen",
                "name": "Maximum query length",
                "type": "Numeric",
                "value": 32,
                "range": [32, 128, 8],
            },
            "doc_maxlen": {
                "parameter_id": "doc_maxlen",
                "name": "Maximum document length",
                "type": "Numeric",
                "value": 180,
                "range": [32, 256, 4],
            },
            "mask_punctuation": {
                "parameter_id": "mask_punctuation",
                "name": "Should mask punctuation",
                "type": "Boolean",
                "value": True,
                "options": [True, False],
            },
            "bsize": {
                "parameter_id": "bsize",
                "name": "Batch size",
                "type": "Numeric",
                "value": 32,
                "range": [8, 128, 8],
            },
            "amp": {
                "parameter_id": "amp",
                "name": "Use amp",
                "type": "Boolean",
                "value": False,
                "options": [True, False],
            },
            "nbits": {
                "parameter_id": "nbits",
                "name": "nbits",
                "type": "Numeric",
                "value": 1,
                "options": [1, 2, 4],
            },
            "kmeans_niters": {
                "parameter_id": "kmeans_niters",
                "name": "Number of iterations (kmeans)",
                "type": "Numeric",
                "value": 4,
                "range": [1, 8, 1],
            },
            "num_partitions_max": {
                "parameter_id": "num_partitions_max",
                "name": "Maximum number of partitions",
                "type": "Numeric",
                "value": 10000000,
            },
            "nway": {
                "parameter_id": "nway",
                "name": "N way",
                "type": "Numeric",
                "value": 2,
            },
        }

        # Placeholder object variables
        self.indexer = None

    def load(self, *args, **kwargs):
        start_t = time.time()

        # Step 1: Create ColBERT configurations
        config = ColBERTConfig(
            **{
                parameter_id: parameter["value"]
                for parameter_id, parameter in self.parameters.items()
                if parameter_id not in ["model"]
            }
        )

        self.indexer = Indexer(kwargs["checkpoint"], config=config)

        self._logger.info(
            "%s pipeline - loading took %s seconds",
            self.pipeline_name,
            time.time() - start_t,
        )

    def get_parameters(self):
        return [self.parameters.values()]

    def set_parameter(self, parameter):
        self.parameters[parameter["parameter_id"]] = parameter

    def get_parameter(self, parameter_id: str):
        return self.parameters[parameter_id]

    def get_parameter_type(self, parameter_id: str):
        return self.parameters[parameter_id]["type"]

    def set_parameter_value(self, parameter_id: str, parameter_value: int):
        self.parameters[parameter_id]["value"] = parameter_value

    def get_parameter_value(self, parameter_id: str):
        return self.parameters[parameter_id]["value"]

    def serialize(self):
        return {
            "pipeline_id": self.pipeline_id,
            "parameters": {
                parameter["parameter_id"]: parameter["value"]
                for parameter in self.parameters.values()
            },
        }

    def index(
        self, documents: Union[List[dict], str], index_path: str, *args, **kwargs
    ):
        self.indexer.configure(index_path=index_path)
        self.indexer.index(name="index", collection=documents, overwrite=True)
