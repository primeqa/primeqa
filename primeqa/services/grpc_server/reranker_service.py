import logging
from typing import Union

from grpc import ServicerContext, StatusCode
from google.protobuf.json_format import MessageToDict

from primeqa.services.configurations import Settings
from primeqa.services.parameters import get_parameter_type

from primeqa.services.factories import RERANKERS_REGISTRY, RerankerFactory
from primeqa.services.grpc_server.utils import (
    parse_parameter_value,
    generate_parameters,
)
from primeqa.services.store import DIR_NAME_INDEX, StoreFactory
from primeqa.services.exceptions import ErrorMessages
from primeqa.services.grpc_server.grpc_generated.reranker_pb2_grpc import (
    RerankerServiceServicer,
)
from primeqa.services.grpc_server.grpc_generated.indexer_pb2 import Document
from primeqa.services.grpc_server.grpc_generated.retriever_pb2 import (
    Hit,
    HitPerQuery,
)

from primeqa.services.grpc_server.grpc_generated.reranker_pb2 import (
    GetRerankersRequest,
    Reranker,
    GetRerankersResponse,
    RerankRequest,
    RerankResponse,
)


class RerankerService(RerankerServiceServicer):
    def __init__(self, config: Settings, logger: Union[logging.Logger, None] = None):
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger
        self._config = config
        self._store = StoreFactory.get_store()
        self._logger.info("%s is successfully initialized.", self.__class__.__name__)

    def GetRerankers(
        self, request: GetRerankersRequest, context: ServicerContext
    ) -> GetRerankersResponse:
        """_summary_

        Args:
            request (GetRerankersRequest):
            context (ServicerContext): gRPC context information for method call

        Returns:
            GetRerankersResponse: List of available rerankers
        """
        return GetRerankersResponse(
            rerankers=[
                Reranker(
                    reranker_id=reranker_id,
                    parameters=generate_parameters(
                        reranker, skip=["index_root", "index_name"]
                    ),
                    #engine_type=reranker.get_engine_type(),
                )
                for reranker_id, reranker in RERANKERS_REGISTRY.items()
            ]
        )

    def Rerank(
        self, request: RerankRequest, context: ServicerContext
    ) -> RerankResponse:
        """_summary_

        Args:
            request (RerankRequest):
            context (ServicerContext): gRPC context information for method call

        Returns:
            RerankResponse:
        """
        
        # Verify requested reranker exists
        try:
            reranker = RERANKERS_REGISTRY[request.reranker.reranker_id]
        except KeyError:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(
                ErrorMessages.INVALID_RERANKER.value.format(
                    request.reranker.reranker_id,
                    ", ".join(RERANKERS_REGISTRY.keys()),
                )
            )
            return RerankResponse()

        # Load default reranker keyword arguments
        reranker_kwargs = {
            k: v.default for k, v in reranker.__dataclass_fields__.items() if v.init
        }

        # If parameters are provided in request then update keyword arguments used to instantiate reranker instance
        if request.reranker.parameters:
            for parameter in request.reranker.parameters:
                if parameter.parameter_id not in reranker_kwargs:
                    context.set_code(StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        ErrorMessages.INVALID_PARAMETER.value.format(
                            "reranker", parameter.parameter_id
                        )
                    )
                    return RerankResponse()

                reranker_kwargs[parameter.parameter_id] = parse_parameter_value(
                    parameter,
                    get_parameter_type(
                        component=reranker, parameter_id=parameter.parameter_id
                    ),
                )

        # Create reranker instance
        try:
            instance = RerankerFactory.get(reranker, reranker_kwargs)
        except (ValueError, TypeError) as err:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(err.args[0])
            return RerankResponse()

        # Rerank
        instance_fields = [
            k
            for k, v in instance.__class__.__dataclass_fields__.items()
            if not "exclude_from_hash" in v.metadata
            or not v.metadata["exclude_from_hash"]
        ]
        self._logger.info(
            "Applying '%s' reranker with parameters = %s for queries = %s",
            instance.__class__.__name__,
            {
                k: getattr(instance, k) if k in instance_fields else v
                for k, v in reranker_kwargs.items()
            },
            request.queries,
        )
        try:
            
            request_dict = MessageToDict(request, preserving_proto_field_name=True)
            queries = request_dict["queries"]
            documentsperquery = [queryhits["hits"] for queryhits in request_dict["hitsperquery"]]

            results = instance.rerank(queries=queries, documents=documentsperquery, **reranker_kwargs)
            
            self._logger.info(
                "Applying '%s' reranker for queries = %s returns results = %s",
                instance.__class__.__name__,
                request.queries,
                results,
            )
        except TypeError:
            context.set_code(StatusCode.INTERNAL)
            context.set_details(
                ErrorMessages.FAILED_TO_INITIALIZE.value.format(
                    f"{reranker.reranker_id} reranker"
                )
            )
            return RerankResponse()

        reranked_results = [ HitPerQuery(hits=r) for r in results]

        return RerankResponse(hits=reranked_results)
