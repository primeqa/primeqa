import logging
from typing import Union, List
import multiprocessing as mp

from grpc import ServicerContext, StatusCode
from google.protobuf.json_format import MessageToDict

from primeqa.services.configurations import Settings
from primeqa.services.grpc_server.utils import parse_parameter_value
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

from primeqa.pipelines import get_pipeline, load_pipeline, IndexerPipeline
from primeqa.services.store import StoreFactory, Store


def index(
    store: Store,
    index_id: str,
    pipeline: IndexerPipeline,
    documents_to_index: List[dict],
):
    index_information = store.get_index_information(index_id)
    try:
        pipeline.index(documents_to_index, store.get_index_file_path(index_id))
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
            if request_idx == 0:
                # Step 2.a: Remove existing index if index_id is provide in the request
                if request.index_id:
                    self._store.delete_index(request.index_id)
                    index_information["index_id"] = request.index_id

                # Step 2.b: Verify requested pipeline's type
                pipeline = get_pipeline(pipeline_id=request.pipeline.pipeline_id)
                if pipeline.pipeline_type != IndexerPipeline.__name__:
                    context.set_code(StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        f"Invalid pipeline type: {pipeline.pipeline_type}. Only pipelines with type: {IndexerPipeline.__name__} are applicable."
                    )
                    return GenerateIndexResponse()

                # Step 2.c: If parameters are provided in request and they differ from existing pipeline, create new pipeline object
                if request.pipeline.parameters:
                    try:
                        # Step 2.c.i: Parse parameter's value from request
                        parameter_value = parse_parameter_value(
                            parameter=parameter,
                            _type=pipeline.get_parameter_type(parameter.parameter_id),
                        )

                        # Step 2.c.ii: If unable to parse value, raise gRPC error
                        if parameter_value is None:
                            context.set_code(StatusCode.INVALID_ARGUMENT)
                            context.set_details(
                                f"Invalid pipeline parameter value: {parameter.parameter_id}. Only use pre-defined parameter value types."
                            )
                            return GenerateIndexResponse()

                        # Step 2.c.iii: Compare against existing parameter value
                        if (
                            pipeline.get_parameter_value(parameter.parameter_id)
                            != parameter_value
                        ):
                            pipeline = get_pipeline(
                                pipeline_id=request.pipeline.pipeline_id,
                                invoke__init__=True,
                            )
                            break
                    except KeyError:
                        context.set_code(StatusCode.INVALID_ARGUMENT)
                        context.set_details(
                            f"Invalid pipeline parameter: {parameter.parameter_id}. Only pre-defined parameters can be modified."
                        )
                        return GenerateIndexResponse()

                    # Step 2.c.iv: Set different parameter values in new pipeline object
                    for parameter in request.pipeline.parameters:
                        # Step 2.c.iv.*: Parse parameter's value from request
                        parameter_value = parse_parameter_value(
                            parameter=parameter,
                            _type=pipeline.get_parameter_type(parameter.parameter_id),
                        )
                        # Step 2.c.iv.**: Update different parameter value
                        if (
                            pipeline.get_parameter_value(parameter.parameter_id)
                            != parameter_value
                        ):
                            pipeline.set_parameter_value(
                                parameter_id=parameter.parameter_id,
                                parameter_value=parameter_value,
                            )

                    # Step 2.c.v: Load latest checkpoint, if available else load default model
                    try:
                        checkpoint = self._store.get_latest_checkpoint_path(
                            pipeline.pipeline_id
                        )
                    except IndexError:
                        checkpoint = self._store.get_model_path(
                            pipeline.get_parameter_value("model")
                        )

                    # Step 2.c.vi: Load pipeline
                    pipeline.load(checkpoint=checkpoint)
                else:
                    # Step 2.c.i: Load latest checkpoint, if available else load default model
                    try:
                        checkpoint = self._store.get_latest_checkpoint_path(
                            pipeline.pipeline_id
                        )
                    except IndexError:
                        checkpoint = self._store.get_model_path(
                            pipeline.get_parameter_value("model")
                        )

                    # Step 2.c.ii: Activate if existing pipeline, if not active already
                    load_pipeline(pipeline.pipeline_id, checkpoint=checkpoint)

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
        self._store.save_index_documents(
            index_id=index_information["index_id"], documents=documents_to_index
        )

        # Step 5: Kick-off async index generation
        try:
            pipeline.index(
                documents_to_index,
                self._store.get_index_file_path(index_information["index_id"]),
            )
        except RuntimeError as err:
            index_information["status"] = CORRUPT
            logging.exception(
                "Generation failed for index with id=%s. Resultant index may be corrupted.",
                index_information["index_id"],
            )
            logging.exception(err.args[0])

        self._store.save_index_information(
            index_information["index_id"], information=index_information
        )

        # Step 6: Return with "INDEXING" status
        return GenerateIndexResponse(
            index_id=index_information["index_id"],
            status=index_information["status"],
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
