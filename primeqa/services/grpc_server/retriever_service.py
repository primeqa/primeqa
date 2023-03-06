import logging
from typing import Union

from grpc import ServicerContext, StatusCode

from primeqa.services.configurations import Settings
from primeqa.services.parameters import get_parameter_type
from primeqa.services.constants import (
    ATTR_STATUS,
    ATTR_CONFIGURATION,
    ATTR_ENGINE_TYPE,
    ATTR_CHECKPOINT,
    IndexStatus,
)
from primeqa.services.factories import RETRIEVERS_REGISTRY, RetrieverFactory
from primeqa.services.grpc_server.utils import (
    parse_parameter_value,
    generate_parameters,
)
from primeqa.services.store import DIR_NAME_INDEX, StoreFactory
from primeqa.services.exceptions import ErrorMessages
from primeqa.services.grpc_server.grpc_generated.retriever_pb2_grpc import (
    RetrievingServiceServicer,
)
from primeqa.services.grpc_server.grpc_generated.indexer_pb2 import Document
from primeqa.services.grpc_server.grpc_generated.retriever_pb2 import (
    GetRetrieversRequest,
    Retriever,
    GetRetrieversResponse,
    RetrieveRequest,
    Hit,
    HitPerQuery,
    RetrieveResponse,
)


class RetrieverService(RetrievingServiceServicer):
    def __init__(self, config: Settings, logger: Union[logging.Logger, None] = None):
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger
        self._config = config
        self._store = StoreFactory.get_store()
        self._logger.info("%s is successfully initialized.", self.__class__.__name__)

    def GetRetrievers(
        self, request: GetRetrieversRequest, context: ServicerContext
    ) -> GetRetrieversResponse:
        """_summary_

        Args:
            request (GetRetrieversRequest):
            context (ServicerContext): gRPC context information for method call

        Returns:
            GetRetrieversResponse: List of available retrievers
        """
        return GetRetrieversResponse(
            retrievers=[
                Retriever(
                    retriever_id=retriever_id,
                    parameters=generate_parameters(
                        retriever, skip=["index_root", "index_name"]
                    ),
                    engine_type=retriever.get_engine_type(),
                )
                for retriever_id, retriever in RETRIEVERS_REGISTRY.items()
            ]
        )

    def Retrieve(
        self, request: RetrieveRequest, context: ServicerContext
    ) -> RetrieveResponse:
        """_summary_

        Args:
            request (RetrieveRequest):
            context (ServicerContext): gRPC context information for method call

        Returns:
            RetrieveResponse:
        """
        # Step 1: Load index information
        if request.index_id:
            index_root = self._store.get_index_directory_path(request.index_id)
            # Step 1.a: Check if `index_root` exists
            if not self._store.exists(index_root):
                context.set_code(StatusCode.NOT_FOUND)
                context.set_details(
                    ErrorMessages.FAILED_TO_LOCATE_INDEX.value.format(request.index_id)
                )
                return RetrieveResponse()

            # Step 1.b: Load index information
            index_information = self._store.get_index_information(
                index_id=request.index_id
            )
            if index_information[ATTR_STATUS] != IndexStatus.READY.value:
                context.set_code(StatusCode.INVALID_ARGUMENT)
                context.set_details(
                    ErrorMessages.INDEX_UNAVAILABLE_FOR_QUERYING.value.format(
                        index_information[ATTR_STATUS]
                    )
                )
                return RetrieveResponse()

        else:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(ErrorMessages.INVALID_REQUEST.value.format("index_id"))
            return RetrieveResponse()

        # Step 2: Verify requested retriever exists
        try:
            retriever = RETRIEVERS_REGISTRY[request.retriever.retriever_id]
        except KeyError:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(
                ErrorMessages.INVALID_RETRIEVER.value.format(
                    request.retriever.retriever_id,
                    ", ".join(RETRIEVERS_REGISTRY.keys()),
                )
            )
            return RetrieveResponse()

        # Step 3: Match engine type of requested collection and retriever
        if (
            index_information[ATTR_CONFIGURATION][ATTR_ENGINE_TYPE]
            != retriever.get_engine_type()
        ):
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(
                ErrorMessages.MISMATCHED_ENGINE_TYPE.value.format(
                    index_information[ATTR_CONFIGURATION][ATTR_ENGINE_TYPE],
                    request.retriever.retriever_id,
                    retriever.get_engine_type(),
                )
            )
            return RetrieveResponse()

        # Step 3: Load default retriever keyword arguments
        retriever_kwargs = {
            k: v.default for k, v in retriever.__dataclass_fields__.items() if v.init
        }

        # Step 4: If parameters are provided in request then update keyword arguments used to instantiate retriever instance
        if request.retriever.parameters:
            for parameter in request.retriever.parameters:
                if parameter.parameter_id not in retriever_kwargs:
                    context.set_code(StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        ErrorMessages.INVALID_PARAMETER.value.format(
                            "retriever", parameter.parameter_id
                        )
                    )
                    return RetrieveResponse()

                retriever_kwargs[parameter.parameter_id] = parse_parameter_value(
                    parameter,
                    get_parameter_type(
                        component=retriever, parameter_id=parameter.parameter_id
                    ),
                )

        # Step 5: Update index specific arguments
        retriever_kwargs["index_root"] = self._store.get_index_directory_path(
            request.index_id
        )
        retriever_kwargs["index_name"] = DIR_NAME_INDEX
        retriever_kwargs["collection"] = self._store.get_index_documents_file_path(
            index_id=request.index_id
        )
        if ATTR_CHECKPOINT in retriever_kwargs:
            retriever_kwargs[ATTR_CHECKPOINT] = self._store.get_checkpoint_path(
                index_information[ATTR_CONFIGURATION][ATTR_CHECKPOINT]
            )

        # Step 6: Create retriever instance
        try:
            instance = RetrieverFactory.get(retriever, retriever_kwargs)
        except (ValueError, TypeError) as err:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(err.args[0])
            return RetrieveResponse()

        # Step 7: Retrieve
        instance_fields = [
            k
            for k, v in instance.__class__.__dataclass_fields__.items()
            if not "exclude_from_hash" in v.metadata
            or not v.metadata["exclude_from_hash"]
        ]
        self._logger.info(
            "Applying '%s' retriever with parameters = %s for queries = %s",
            instance.__class__.__name__,
            {
                k: getattr(instance, k) if k in instance_fields else v
                for k, v in retriever_kwargs.items()
            },
            request.queries,
        )
        try:
            results = instance.predict(input_texts=request.queries, **retriever_kwargs)
            self._logger.info(
                "Applying '%s' retriever for queries = %s returns results = %s",
                instance.__class__.__name__,
                request.queries,
                results,
            )
        except TypeError:
            context.set_code(StatusCode.INTERNAL)
            context.set_details(
                ErrorMessages.FAILED_TO_INITIALIZE.value.format(
                    f"{retriever.retriever_id} retriever"
                )
            )
            return RetrieveResponse()

        hits = []
        for result_per_query in results:
            hits_per_query = []
            for hit in result_per_query:
                try:
                    document = self._store.get_index_document(
                        index_id=request.index_id, document_idx=hit[0]
                    )
                    hits_per_query.append(
                        Hit(
                            document=Document(
                                text=document["text"],
                                document_id=document["document_id"]
                                if "document_id" in document
                                else None,
                                title=document["title"]
                                if "title" in document
                                else None,
                            ),
                            score=hit[1],
                        )
                    )
                except (FileNotFoundError, KeyError):
                    continue

            hits.append(HitPerQuery(hits=hits_per_query))

        return RetrieveResponse(hits=hits)
