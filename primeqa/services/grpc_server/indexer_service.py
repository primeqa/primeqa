import logging
from typing import Union, List
import multiprocessing as mp

from grpc import ServicerContext, StatusCode
from google.protobuf.struct_pb2 import Struct
from google.protobuf.json_format import MessageToDict

from primeqa.services.configurations import Settings
from primeqa.services.grpc_server.grpc_generated.indexer_pb2_grpc import (
    IndexerServicer,
)
from primeqa.services.grpc_server.grpc_generated.indexer_pb2 import (
    GetIndexStatusRequest,
    GenerateIndexResponse,
    IndexStatusResponse,
    READY,
    INDEXING,
    CORRUPT,
    DOES_NOT_EXISTS,
)

from primeqa.pipelines import get_pipeline, activate_pipeline, RetrieverPipeline
from primeqa.services.store import StoreFactory, Store


def index(
    store: Store,
    index_id: str,
    pipeline: RetrieverPipeline,
    documents_to_index: List[dict],
):
    index_information = store.get_index_information(index_id)
    try:
        pipeline.index(documents_to_index, store.get_index_directory_path(index_id))
        index_information["status"] = READY
    except RuntimeError as err:
        index_information["status"] = CORRUPT
        logging.exception(
            "Generation failed for index with id=%s. Resultant index may be corrupted.",
            index_id,
        )
        logging.exception(err.args[0])

    store.save_index_information(index_id, information=index_information)


class IndexerService(IndexerServicer):
    def __init__(self, config: Settings, logger: Union[logging.Logger, None] = None):
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger
        self._config = config
        self._store = StoreFactory.get_store()
        self._logger.info("%s is successfully initialized.", self.__class__.__name__)

    def GenerateIndex(self, request_iterator, context: ServicerContext):
        # Step 1: Assign unique index id
        index_information = {
            "index_id": self._store.generate_index_uuid(),
            "status": INDEXING,
        }

        # Step 2: Iterate over all index requests to collect documents
        documents_to_index = list()
        pipeline = None
        for request_idx, request in enumerate(request_iterator):
            # First request
            if request_idx == 0:
                # Remove existing index if index_id is provide in the request
                if request.index_id:
                    self._store.delete_index(request.index_id)
                    index_information["index_id"] = request.index_id

                # Activate requested pipeline
                pipeline = get_pipeline(pipeline_id=request.pipeline.pipeline_id)
                if pipeline.pipeline_type != RetrieverPipeline.__name__:
                    context.set_code(StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        f"Invalid pipeline type: {pipeline.pipeline_type}. Only pipelines with type: {RetrieverPipeline.__name__} are applicable."
                    )
                    return GenerateIndexResponse()
                else:
                    activate_pipeline(pipeline.pipeline_id)

                index_information["metadata"] = {"pipeline": pipeline.serialize()}

            # Append documents from each index request
            documents_to_index.extend(
                MessageToDict(request, preserving_proto_field_name=True)["documents"]
            )

        # Step 3: Save index information
        self._store.save_index_information(
            index_id=index_information["index_id"],
            information=index_information,
        )

        # Step 4: Save documents used in index
        self._store.save_documents(
            index_id=index_information["index_id"], documents=documents_to_index
        )

        # Step 5: Kick-off async index generation
        process = mp.Process()
        process = mp.Process(
            target=index,
            args=(
                self._store,
                index_information["index_id"],
                pipeline,
                documents_to_index,
            ),
            daemon=True,
        )
        process.start()

        # Step 6: Return with "INDEXING" status
        metadata = Struct()
        metadata.update(index_information["metadata"])
        return GenerateIndexResponse(
            index_id=index_information["index_id"],
            status=index_information["status"],
            metadata=metadata,
        )

    def GetIndexStatus(self, request: GetIndexStatusRequest, context: ServicerContext):
        try:
            index_information = self._store.get_index_information(
                index_id=request.index_id
            )
            if index_information["status"] == READY:
                return IndexStatusResponse(status=READY)
            elif index_information["status"] == INDEXING:
                return IndexStatusResponse(status=INDEXING)
            else:
                return IndexStatusResponse(status=CORRUPT)
        except KeyError:
            return IndexStatusResponse(status=CORRUPT)
        except FileNotFoundError:
            return IndexStatusResponse(status=DOES_NOT_EXISTS)
