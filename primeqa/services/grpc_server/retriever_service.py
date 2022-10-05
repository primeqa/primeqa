import logging
from typing import Union

from grpc import ServicerContext, StatusCode
from google.protobuf.struct_pb2 import Struct

from primeqa.services.configurations import Settings
from primeqa.services.grpc_server.utils import parse_parameter_value
from primeqa.services.store import StoreFactory
from primeqa.services.grpc_server.grpc_generated.retriever_pb2_grpc import (
    RetrieverServicer,
)

from primeqa.services.grpc_server.grpc_generated.indexer_pb2 import Document
from primeqa.services.grpc_server.grpc_generated.retriever_pb2 import (
    SearchRequest,
    SearchResponse,
    Hit,
    HitPerQuery,
)

from primeqa.pipelines import get_pipeline, RetrieverPipeline


class RetrieverService(RetrieverServicer):
    def __init__(self, config: Settings, logger: Union[logging.Logger, None] = None):
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger
        self._config = config
        self._store = StoreFactory.get_store()
        self._logger.info("%s is successfully initialized.", self.__class__.__name__)

    def Search(self, request: SearchRequest, context: ServicerContext):
        # Step 1: Get requested pipeline
        pipeline = get_pipeline(pipeline_id=request.pipeline.pipeline_id)

        # Step 2: Verify requested pipeline's type
        if pipeline.pipeline_type != RetrieverPipeline.__name__:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"Invalid pipeline type: {pipeline.pipeline_type}. Only pipelines with type: {RetrieverPipeline.__name__} are applicable."
            )
            return SearchResponse()

        # Step 3: If parameters are provided in request and they differ from existing pipeline, create new pipeline object
        if request.pipeline.parameters:
            # Step 3.a: Create new pipeline object, if necessary
            for parameter in request.pipeline.parameters:
                try:
                    # Step 3.a.i: Parse parameter's value from request
                    parameter_value = parse_parameter_value(
                        parameter=parameter,
                        _type=pipeline.get_parameter_type(parameter.parameter_id),
                    )

                    # Step 3.a.ii: If unable to parse value, raise gRPC error
                    if parameter_value is None:
                        context.set_code(StatusCode.INVALID_ARGUMENT)
                        context.set_details(
                            f"Invalid pipeline parameter value: {parameter.parameter_id}. Only use pre-defined parameter value types."
                        )
                        return SearchResponse()

                    # Step 3.a.iii: Compare against existing parameter value
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
                    return SearchResponse()

            # Step 3.b: Set different parameter values in new pipeline object
            for parameter in request.pipeline.parameters:
                # Step 3.b.i: Parse parameter's value from request
                parameter_value = parse_parameter_value(
                    parameter=parameter,
                    _type=pipeline.get_parameter_type(parameter.parameter_id),
                )

                # Step 3.b.ii: Update different parameter value
                if (
                    pipeline.get_parameter_value(parameter.parameter_id)
                    != parameter_value
                ):
                    pipeline.set_parameter_value(
                        parameter_id=parameter.parameter_id,
                        parameter_value=parameter_value,
                    )

        # Step 4: Retrieve
        results = pipeline.retrieve(
            input_texts=request.query,
            index_path=self._store.get_index_file_path(request.index_id),
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
                                document_id=document["document_id"],
                                text=document["text"],
                                title=document["title"]
                                if "title" in document
                                else None,
                                metadata=Struct().update(document["metadata"])
                                if "metadata" in document
                                else None,
                            ),
                            score=hit[1],
                        )
                    )
                except (FileNotFoundError, KeyError):
                    continue

            hits.append(HitPerQuery(hits=hits_per_query))

        return SearchResponse(hits=hits)
