import logging
from typing import Union, List
import time

from primeqa.pipelines.base import RetrieverPipeline


class ColBERTRetriever(RetrieverPipeline):
    """
    Retriever: ColBERT
    """

    def __init__(
        self,
        logger: Union[logging.Logger, None] = None,
    ) -> None:
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger

        # Default class variables
        self.pipeline_id = self.__class__.__name__
        self.pipeline_name = "Dense Retriever (ColBERT)"
        self.pipeline_description = ""
        self.pipeline_type = RetrieverPipeline.__name__
        self.parameters = {}

        # Placeholder class variables
        self.preprocessor = None
        self.trainer = None

    def load(self):
        start_t = time.time()

        self._logger.info(
            "%s pipeline - loading took %s seconds",
            self.pipeline_name,
            time.time() - start_t,
        )

    def get_parameters(self):
        return [self.parameters.values()]

    def set_parameter(self, parameter):
        self.parameters[parameter["parameter_id"]] = parameter

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

    def index(self, documents: List[dict], index_path: str, *args, **kwargs):
        return super().index(documents, index_path, *args, **kwargs)

    def retrieve(self, input_text: List[str], *args, **kwargs):
        return super().retrieve(input_text, *args, **kwargs)
