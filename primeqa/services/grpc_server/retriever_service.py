import logging
from typing import Union

from grpc import ServicerContext, StatusCode

from primeqa.services.configurations import Settings
from primeqa.services.parameters import get_parameter_type
from primeqa.services.constants import ATTR_STATUS, IndexStatus
from primeqa.services.factories import RETRIEVERS_REGISTRY, RetrieverFactory
from primeqa.services.grpc_server.utils import (
    parse_parameter_value,
    generate_parameters,
)
from primeqa.services.store import DIR_NAME_INDEX, StoreFactory
from primeqa.services.grpc_server.grpc_generated.retriever_pb2_grpc import (
    RetrieverServicer,
)
from primeqa.services.grpc_server.grpc_generated.indexer_pb2 import Document
from primeqa.services.grpc_server.grpc_generated.retriever_pb2 import (
    GetRetrieversRequest,
    RetrieverComponent,
    GetRetrieversResponse,
    RetrieveRequest,
    Hit,
    HitPerQuery,
    RetrieveResponse,
)


class RetrieverService(RetrieverServicer):
    def __init__(self, config: Settings, logger: Union[logging.Logger, None] = None):
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger
        self._config = config
        self._store = StoreFactory.get_store()
        self._logger.info("%s is successfully initialized.", self.__class__.__name__)

    def getRetrievers(
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
                RetrieverComponent(
                    retriever_id=retriever_id,
                    parameters=generate_parameters(
                        retriever, skip=["index_root", "index_name"]
                    ),
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
        # Step 1: Verify requested retriever
        try:
            retriever = RETRIEVERS_REGISTRY[request.retriever.retriever_id]
        except KeyError:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"Invalid retriever: {request.retriever.retriever_id}. Please select one of the following pre-defined retrievers: {', '.join(RETRIEVERS_REGISTRY.keys())}"
            )
            return RetrieveResponse()

        # Step 2: Load default retriever keyword arguments
        retriever_kwargs = {
            k: v.default for k, v in retriever.__dataclass_fields__.items()
        }

        # Step 3: If parameters are provided in request then update keyword arguments used to instantiate retriever instance
        if request.retriever.parameters:
            for parameter in request.retriever.parameters:
                try:
                    retriever_kwargs[parameter.parameter_id] = parse_parameter_value(
                        parameter,
                        get_parameter_type(
                            component=retriever, parameter_id=parameter.parameter_id
                        ),
                    )
                except KeyError:
                    context.set_code(StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        f"Invalid retriever parameter: {parameter.parameter_id}. Only pre-defined parameters can be modified."
                    )
                    return RetrieveResponse()

        # Step 4: Load index information
        if request.index_id:
            index_root = self._store.get_index_directory_path(request.index_id)
            # Step 4.a: Check if `index_root` exists
            if not self._store.exists(index_root):
                context.set_code(StatusCode.NOT_FOUND)
                context.set_details(
                    f"Invalid request. Index with id {request.index_id} doesn't exist."
                )
                return RetrieveResponse()

            # Step 4.b: Load index information
            index_information = self._store.get_index_information(
                index_id=request.index_id
            )
            if index_information[ATTR_STATUS] != IndexStatus.READY.value:
                context.set_code(StatusCode.INVALID_ARGUMENT)
                context.set_details(
                    f'Cannot query index with "{index_information[ATTR_STATUS]}" status. Please make sure index has "READY" status before querying.'
                )
                return RetrieveResponse()

            # Step 4.c: Update index specific arguments
            retriever_kwargs["index_root"] = self._store.get_index_directory_path(
                request.index_id
            )
            retriever_kwargs["index_name"] = DIR_NAME_INDEX
        else:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details("Invalid request. `index_id` must be provided.")
            return RetrieveResponse()

        try:
            instance = RetrieverFactory.get(retriever, retriever_kwargs)
        except ValueError as err:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(err.args[0])
            return RetrieveResponse()

        # Step 4: Retrieve
        results = instance.retrieve(
            input_texts=request.queries,
        )

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
