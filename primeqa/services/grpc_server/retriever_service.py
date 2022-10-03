import logging
from typing import Union

from grpc import ServicerContext

from primeqa.services.configurations import Settings
from primeqa.services.grpc_server.grpc_generated.retriever_pb2_grpc import (
    RetrieverServicer,
)
from primeqa.services.grpc_server.grpc_generated.retriever_pb2 import SearchRequest, Hit

from primeqa.pipelines import get_pipeline, activate_pipeline


class RetrieverService(RetrieverServicer):
    def __init__(self, config: Settings, logger: Union[logging.Logger, None] = None):
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger
        self._config = config
        self._logger.info("%s is successfully initialized.", self.__class__.__name__)

    def Search(self, request: SearchRequest, context: ServicerContext):
        return super().Search(request, context)
