import logging
from typing import Union

from grpc import ServicerContext, StatusCode
from google.protobuf.struct_pb2 import Struct
from google.protobuf.json_format import MessageToDict

from primeqa.services.exceptions import ErrorMessages
from primeqa.services.configurations import Settings
from primeqa.services.constants import (
    ATTR_INDEX_ID,
    ATTR_STATUS,
    ATTR_ENGINE_TYPE,
    ATTR_METADATA,
    IndexStatus,
)
from primeqa.services.store import DIR_NAME_INDEX, StoreFactory
from primeqa.services.grpc_server.utils import (
    parse_parameter_value,
    generate_parameters,
)
from primeqa.services.parameters import get_parameter_type
from primeqa.services.factories import (
    INDEXERS_REGISTRY,
    IndexerFactory,
)
from primeqa.services.grpc_server.grpc_generated.indexer_pb2_grpc import (
    IndexerServicer,
)
from primeqa.services.grpc_server.grpc_generated.indexer_pb2 import (
    GetIndexersRequest,
    GetIndexersResponse,
    IndexerComponent,
    GenerateIndexResponse,
    GetIndexStatusRequest,
    IndexStatusResponse,
    GetIndexesRequest,
    IndexInformation,
    GetIndexesResponse,
    READY,
    INDEXING,
    CORRUPT,
    DOES_NOT_EXISTS,
)


class IndexerService(IndexerServicer):
    def __init__(
        self, config: Settings, logger: Union[logging.Logger, None] = None
    ):
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger
        self._config = config
        self._store = StoreFactory.get_store()
        self._logger.info(
            "%s is successfully initialized.", self.__class__.__name__
        )

    def GetIndexers(
        self, request: GetIndexersRequest, context: ServicerContext
    ) -> GetIndexersResponse:
        """_summary_

        Args:
            request (GetIndexersRequest):
            context (ServicerContext): gRPC context information for method call

        Returns:
            GetIndexersResponse: List of available indexers
        """
        return GetIndexersResponse(
            indexers=[
                IndexerComponent(
                    indexer_id=indexer_id,
                    parameters=generate_parameters(
                        indexer, skip=["index_root", "index_name"]
                    ),
                )
                for indexer_id, indexer in INDEXERS_REGISTRY.items()
            ]
        )

    def GenerateIndex(self, request_iterator, context: ServicerContext):
        # Step 1: Assign unique index id
        index_information = {
            ATTR_INDEX_ID: self._store.generate_index_uuid(),
            ATTR_STATUS: IndexStatus.INDEXING.value,
        }

        # Step 2: Iterate over all index requests to collect documents
        documents_to_index = list()
        instance = None
        for request_idx, request in enumerate(request_iterator):
            if request_idx == 0:
                # Step 2.a: Verify requested indexer
                try:
                    indexer = INDEXERS_REGISTRY[request.indexer.indexer_id]
                except KeyError:
                    context.set_code(StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        ErrorMessages.INVALID_INDEXER.value.format(
                            request.indexer.indexer_id,
                            ", ".join(INDEXERS_REGISTRY.keys()),
                        )
                    )
                    return GenerateIndexResponse()

                # Step 2.b: Remove existing index if index_id is provide in the request
                if request.index_id:
                    self._store.delete_index(request.index_id)
                    index_information[ATTR_INDEX_ID] = request.index_id

                # Step 2.c: If additional metadata is provided, copy it over
                if request.metadata:
                    index_information[ATTR_METADATA] = MessageToDict(
                        request.metadata
                    )

                # Step 2.c: Load default retriever keyword arguments
                indexer_kwargs = {
                    k: v.default
                    for k, v in indexer.__dataclass_fields__.items()
                }

                # Step 2.d: If parameters are provided in request then update keyword arguments used to instantiate indexer instance
                if request.indexer.parameters:
                    for parameter in request.indexer.parameters:
                        if parameter.parameter_id not in indexer_kwargs:
                            context.set_code(StatusCode.INVALID_ARGUMENT)
                            context.set_details(
                                ErrorMessages.INVALID_PARAMETER.value.format(
                                    "indexer", parameter.parameter_id
                                )
                            )
                            return GenerateIndexResponse()

                        indexer_kwargs[
                            parameter.parameter_id
                        ] = parse_parameter_value(
                            parameter,
                            get_parameter_type(
                                component=indexer,
                                parameter_id=parameter.parameter_id,
                            ),
                        )
                        # Re-map checkpoint kwarg to point to checkpoint file path in the service's store
                        if parameter.parameter_id == "checkpoint":
                            indexer_kwargs[
                                "checkpoint"
                            ] = self._store.get_checkpoint_path(
                                indexer_kwargs["checkpoint"]
                            )

                # Step 2.e: Update index specific arguments
                indexer_kwargs[
                    "index_root"
                ] = self._store.get_index_directory_path(
                    index_information[ATTR_INDEX_ID]
                )
                indexer_kwargs["index_name"] = DIR_NAME_INDEX

                # Step 2.e: Create indexer instance
                try:
                    instance = IndexerFactory.get(indexer, indexer_kwargs)
                except (ValueError, TypeError) as err:
                    context.set_code(StatusCode.INVALID_ARGUMENT)
                    context.set_details(err.args[0])
                    return GenerateIndexResponse()

            # Append documents from each index request
            documents_to_index.extend(
                MessageToDict(request, preserving_proto_field_name=True)[
                    "documents"
                ]
            )

        # Step 3: Save index information
        index_information[ATTR_ENGINE_TYPE] = indexer.get_engine_type()
        self._store.save_index_information(
            index_id=index_information[ATTR_INDEX_ID],
            information=index_information,
        )

        # Step 4: Save documents used in index
        self._store.save_index_documents(
            index_id=index_information[ATTR_INDEX_ID],
            documents=documents_to_index,
        )

        # Step 5: Kick-off async index generation
        try:
            instance.index(
                self._store.get_index_documents_file_path(
                    index_id=index_information[ATTR_INDEX_ID]
                ),
            )

            # Step 5.b: Set index status to "READY" once indexing is complete
            index_information[ATTR_STATUS] = IndexStatus.READY.value
        except (TypeError, RuntimeError) as err:
            index_information[ATTR_STATUS] = IndexStatus.CORRUPT.value
            logging.exception(
                "Generation failed for index with id=%s. Resultant index may be corrupted.",
                index_information[ATTR_INDEX_ID],
            )
            logging.exception(err.args[0])

        self._store.save_index_information(
            index_information[ATTR_INDEX_ID], information=index_information
        )

        # Step 6: Return
        return GenerateIndexResponse(
            index_id=index_information[ATTR_INDEX_ID],
            status=READY
            if index_information[ATTR_STATUS] == IndexStatus.READY.value
            else (
                INDEXING
                if index_information[ATTR_STATUS] == IndexStatus.INDEXING.value
                else CORRUPT
            ),
        )

    def GetIndexStatus(
        self, request: GetIndexStatusRequest, context: ServicerContext
    ) -> IndexStatusResponse:
        try:
            index_information = self._store.get_index_information(
                index_id=request.index_id
            )
            if index_information[ATTR_STATUS] == IndexStatus.READY.value:
                return IndexStatusResponse(status=READY)
            elif index_information[ATTR_STATUS] == IndexStatus.INDEXING.value:
                return IndexStatusResponse(status=INDEXING)
            else:
                return IndexStatusResponse(status=IndexStatus.CORRUPT.value)
        except KeyError:
            return IndexStatusResponse(status=CORRUPT)
        except FileNotFoundError:
            return IndexStatusResponse(status=DOES_NOT_EXISTS)

    def GetIndexes(
        self, request: GetIndexesRequest, context: ServicerContext
    ) -> GetIndexesResponse:
        resp = GetIndexesResponse()
        # Step 1: Check if "engine_type" is provided
        engine_type = None
        if request.engine_type:
            engine_type = request.engine_type

        # Step 2: Iterate over each index individual to return matching indexes
        for index_id in self._store.get_index_ids():
            try:
                # Step 2.a: Load index information from store
                index_information_dict = self._store.get_index_information(
                    index_id=index_id
                )

                # Step 2.b: Skip index if engine type is provided in request and doesn't match with the one in current index's information
                if engine_type and (
                    ATTR_ENGINE_TYPE in index_information_dict
                    and index_information_dict[ATTR_ENGINE_TYPE] != engine_type
                ):
                    continue

                # Step 2.c: Place index information response payload object
                index_information_grpc_response_object = IndexInformation(
                    index_id=index_id
                )

                # Step 2.d: Add "metadata" information if exists
                if (
                    ATTR_METADATA in index_information_dict
                    and index_information_dict[ATTR_METADATA]
                ):
                    index_information_grpc_response_object.metadata.update(
                        index_information_dict[ATTR_METADATA]
                    )

                # Step 2.e: Add "status" information
                try:
                    status = index_information_dict[ATTR_STATUS]
                    if status == IndexStatus.READY.value:
                        index_information_grpc_response_object.status = READY
                    elif status == IndexStatus.INDEXING.value:
                        index_information_grpc_response_object.status = (
                            INDEXING
                        )
                    else:
                        index_information_grpc_response_object.status = CORRUPT
                except KeyError:
                    index_information_grpc_response_object.status = CORRUPT
            except FileNotFoundError:
                self._logger.warning(
                    ErrorMessages.FAILED_TO_LOCATE_INDEX_INFORMATION.value.format(
                        index_id
                    ).strip()
                )

            resp.indexes.append(index_information_grpc_response_object)

        return resp
