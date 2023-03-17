import logging
from typing import Union

from grpc import ServicerContext, StatusCode

from primeqa.services.configurations import Settings
from primeqa.services.parameters import get_parameter_type
from primeqa.services.constants import (
    ATTR_STATUS,
    ATTR_CONFIGURATION,
    ATTR_CHECKPOINT,
    IndexStatus,
)
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
        # # Step 1: Load index information
        # if request.index_id:
        #     index_root = self._store.get_index_directory_path(request.index_id)
        #     # Step 1.a: Check if `index_root` exists
        #     if not self._store.exists(index_root):
        #         context.set_code(StatusCode.NOT_FOUND)
        #         context.set_details(
        #             ErrorMessages.FAILED_TO_LOCATE_INDEX.value.format(request.index_id)
        #         )
        #         return RerankResponse()

        #     # Step 1.b: Load index information
        #     index_information = self._store.get_index_information(
        #         index_id=request.index_id
        #     )
        #     if index_information[ATTR_STATUS] != IndexStatus.READY.value:
        #         context.set_code(StatusCode.INVALID_ARGUMENT)
        #         context.set_details(
        #             ErrorMessages.INDEX_UNAVAILABLE_FOR_QUERYING.value.format(
        #                 index_information[ATTR_STATUS]
        #             )
        #         )
        #         return RerankResponse()

        # else:
        #     context.set_code(StatusCode.INVALID_ARGUMENT)
        #     context.set_details(ErrorMessages.INVALID_REQUEST.value.format("index_id"))
        #     return RerankResponse()

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

        # # Step 5: Update index specific arguments
        # if request.index_id:
        #     reranker_kwargs["index_root"] = self._store.get_index_directory_path(
        #         request.index_id
        #     )
        #     reranker_kwargs["index_name"] = DIR_NAME_INDEX
        #     reranker_kwargs["collection"] = self._store.get_index_documents_file_path(
        #         index_id=request.index_id
        #     )
            
        # if ATTR_CHECKPOINT not in reranker_kwargs:
        #     reranker_kwargs[ATTR_CHECKPOINT] = self._store.get_checkpoint_path(
        #         index_information[ATTR_CONFIGURATION][ATTR_CHECKPOINT]
        #     )

        # Step 6: Create reranker instance
        try:
            instance = RerankerFactory.get(reranker, reranker_kwargs)
        except (ValueError, TypeError) as err:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(err.args[0])
            return RerankResponse()

        # Step 7: Rerank
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
            
            document_ids = []
            texts = []
            titles = []
            docid_doc_mappings = []
            for queryhits in request.hitsperquery:
                q_doc_ids = []
                q_texts = []
                q_docid_to_doc = {}
                for hit in queryhits.hits:
                    q_texts.append(hit.document.text)
                    q_doc_ids.append(hit.document.document_id)
                    if hit.document.document_id in q_docid_to_doc:
                        context.set_code(StatusCode.INTERNAL)
                        context.set_details(
                            ErrorMessages.FAILED_TO_INITIALIZE.value.format(
                                f"{reranker.reranker_id} reranker"
                            )
                        )
                        return RerankResponse()
                    q_docid_to_doc[hit.document.document_id] =  hit.document
                document_ids.append(q_doc_ids)
                texts.append(q_texts)
                docid_doc_mappings.append(q_docid_to_doc)
            
            results = instance.predict(queries=request.queries, texts=texts, doc_ids=document_ids, **reranker_kwargs)
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

        hits = []
        for q, result_per_query in enumerate(results):
            hits_per_query = []
            docid_doc_mapping = docid_doc_mappings[q]
            for hit in result_per_query:
                docid = hit[0]
                try:
                    hits_per_query.append(
                        Hit(
                            document=Document(
                                text=docid_doc_mapping[docid].text,
                                document_id=docid,
                                title=docid_doc_mapping[docid].title
                            ),
                            score=hit[1],
                        )
                    )
                except (FileNotFoundError, KeyError):
                    continue

            hits.append(HitPerQuery(hits=hits_per_query))

        return RerankResponse(hits=hits)
