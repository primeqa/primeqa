import logging
import time
from typing import List


import uvicorn
from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from primeqa.services.configurations import Settings
from primeqa.services.constants import ATTR_STATUS, ATTR_INDEX_ID, IndexStatus, ATTR_ENGINE_TYPE
from primeqa.services.factories import (
    READERS_REGISTRY,
    INDEXERS_REGISTRY,
    RETRIEVERS_REGISTRY,
    ReaderFactory,
    IndexerFactory,
    RetrieverFactory,
)
from primeqa.services.store import DIR_NAME_INDEX, StoreFactory
from primeqa.services.exceptions import PATTERN_ERROR_MESSAGE, Error, ErrorMessages
from primeqa.services.rest_server.data_models import (
    Indexer,
    Reader,
    Retriever,
    GetAnswersRequest,
    Answer,
    GenerateIndexRequest,
    RetrieveRequest,
    IndexInformation,
    Hit,
)
from primeqa.services.rest_server.utils import generate_parameters


class RestServer:
    def __init__(self, config: Settings = None, logger: logging.Logger = None):
        try:
            if logger is None:
                self._logger = logging.getLogger(self.__class__.__name__)
            else:
                self._logger = logger

            # Initialize application config
            if config is None:
                self._config = Settings()
            else:
                self._config = config

            self._store = StoreFactory.get_store()
        except Exception as ex:
            self._logger.exception("Error configuring server: %s", ex)
            raise

    def run(self) -> None:
        start_t = time.time()

        ############################################################################################
        #                                   API SERVER
        ############################################################################################
        app = FastAPI(
            title="PrimeQA Service",
            version="0.9.2",
            contact={
                "name": "PrimeQA Team",
                "url": "https://github.com/primeqa/primeqa",
                "email": "primeqa@us.ibm.com",
            },
            license_info={
                "name": "Apache 2.0",
                "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
            },
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True if self._config.require_client_auth else False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        ############################################################################################
        #                           Reader API
        ############################################################################################
        @app.get(
            "/readers",
            status_code=status.HTTP_200_OK,
            response_model=List[Reader],
            tags=["Reader"],
        )
        def get_readers():
            try:
                return [
                    {"reader_id": reader_id, "parameters": generate_parameters(reader)}
                    for reader_id, reader in READERS_REGISTRY.items()
                ]
            except Error as err:
                error_message = err.args[0]

                # Identify error code
                mobj = PATTERN_ERROR_MESSAGE.match(error_message)
                if mobj:
                    error_code = mobj.group(1).strip()
                    error_message = mobj.group(2).strip()
                else:
                    error_code = 500

                raise HTTPException(
                    status_code=500,
                    detail={"code": error_code, "message": error_message},
                ) from None

        @app.post(
            "/answers",
            status_code=status.HTTP_201_CREATED,
            response_model=List[List[Answer]],
            tags=["Reader"],
        )
        def get_answers(request: GetAnswersRequest):
            try:
                # Step 1: If contexts are provided, number of contexts need to match number of queries
                if request.contexts and len(request.queries) != len(request.contexts):
                    raise Error(
                        ErrorMessages.MISSING_CONTEXT.value.format(
                            len(request.contexts), len(request.queries)
                        )
                    )

                # Step 2: Verify requested reader
                try:
                    reader = READERS_REGISTRY[request.reader.reader_id]
                except KeyError as err:
                    raise Error(
                        ErrorMessages.INVALID_READER.value.format(
                            request.reader.reader_id, ", ".join(READERS_REGISTRY.keys())
                        )
                    ) from err

                # Step 3: Load default reader keyword arguments
                reader_kwargs = {
                    k: v.default for k, v in reader.__dataclass_fields__.items()
                }

                # Step 4: If parameters are provided in request then update keyword arguments used to instantiate reader instance
                if request.reader.parameters:
                    for parameter in request.reader.parameters:
                        if parameter.parameter_id not in reader_kwargs:
                            raise Error(
                                ErrorMessages.INVALID_PARAMETER.value.format(
                                    "reader", parameter.parameter_id
                                )
                            )

                        reader_kwargs[parameter.parameter_id] = parameter.value

                try:
                    instance = ReaderFactory.get(reader, reader_kwargs)
                except (ValueError, TypeError) as err:
                    raise Error(err.args[0]) from err

                # Step 5: Run apply method
                instance_fields = [
                    k
                    for k, v in instance.__class__.__dataclass_fields__.items()
                    if not "exclude_from_hash" in v.metadata
                    or not v.metadata["exclude_from_hash"]
                ]
                answers_response = []
                try:
                    for idx, query in enumerate(request.queries):
                        # Step 5.a: Run "apply" per query
                        self._logger.info(
                            "Applying '%s' reader with parameters = %s for query = '%s' and contexts = %s",
                            instance.__class__.__name__,
                            {
                                k: getattr(instance, k) if k in instance_fields else v
                                for k, v in reader_kwargs.items()
                            },
                            query,
                            request.contexts[idx].texts,
                        )
                        try:
                            predictions = instance.apply(
                                input_texts=[query] * len(request.contexts[idx]),
                                context=[[text] for text in request.contexts[idx]],
                                **reader_kwargs,
                            )
                            self._logger.info(
                                "Applying '%s' reader for query = '%s' returns predictions = %s",
                                instance.__class__.__name__,
                                query,
                                predictions,
                            )

                            # Step 5.b: Add answers for current query into response object
                            answers_response.append(
                                [
                                    [
                                        {
                                            "text": prediction["span_answer_text"],
                                            "start_char_offset": prediction[
                                                "span_answer"
                                            ]["start_position"],
                                            "end_char_offset": prediction[
                                                "span_answer"
                                            ]["end_position"],
                                            "confidence_score": prediction[
                                                "confidence_score"
                                            ],
                                            "passage_index": int(
                                                prediction["example_id"]
                                            ),
                                        }
                                        for prediction in predictions_for_context
                                    ]
                                    for predictions_for_context in predictions
                                ]
                            )

                        except TypeError as err:
                            raise Error(
                                ErrorMessages.FAILED_TO_INITIALIZE.value.format(
                                    f"{request.reader.reader_id} reader"
                                )
                            ) from err

                except IndexError as err:
                    raise Error(
                        ErrorMessages.MISSING_CONTEXT.value.format(
                            len(request.contexts), len(request.queries)
                        )
                    ) from err
                # Step 5: Return
                return [
                    [
                        {
                            "text": prediction["span_answer_text"],
                            "start_char_offset": prediction["span_answer"][
                                "start_position"
                            ],
                            "end_char_offset": prediction["span_answer"][
                                "end_position"
                            ],
                            "confidence_score": prediction["confidence_score"],
                            "context_index": int(prediction["example_id"]),
                        }
                        for prediction in predictions_for_passage
                    ]
                    for predictions_for_passage in predictions
                ]

            except Error as err:
                error_message = err.args[0]

                # Identify error code
                mobj = PATTERN_ERROR_MESSAGE.match(error_message)
                if mobj:
                    error_code = mobj.group(1).strip()
                    error_message = mobj.group(2).strip()
                else:
                    error_code = 500

                raise HTTPException(
                    status_code=500,
                    detail={"code": error_code, "message": error_message},
                ) from None

        ############################################################################################
        #                           Indexer API
        ############################################################################################
        @app.get(
            "/indexers",
            status_code=status.HTTP_200_OK,
            response_model=List[Indexer],
            tags=["Indexer"],
        )
        def get_indexers():
            return [
                {"indexer_id": indexer_id, "parameters": generate_parameters(indexer)}
                for indexer_id, indexer in INDEXERS_REGISTRY.items()
            ]

        @app.post(
            "/indexes",
            status_code=status.HTTP_201_CREATED,
            response_model=IndexInformation,
            tags=["Indexer"],
        )
        def generate_index(request: GenerateIndexRequest):
            try:
                # Step 1: Assign unique index id
                index_information = {
                    "index_id": self._store.generate_index_uuid(),
                    "status": IndexStatus.INDEXING,
                }

                # Step 2: Verify requested indexer
                try:
                    indexer = INDEXERS_REGISTRY[request.indexer.indexer_id]
                except KeyError as err:
                    raise Error(
                        ErrorMessages.INVALID_INDEXER.value.format(
                            request.indexer.indexer_id,
                            ", ".join(INDEXERS_REGISTRY.keys()),
                        )
                    ) from err

                # Step 3: Remove existing index if index_id is provide in the request
                if request.index_id:
                    self._store.delete_index(request.index_id)
                    index_information["index_id"] = request.index_id

                # Step 4: Load default retriever keyword arguments
                indexer_kwargs = {
                    k: v.default for k, v in indexer.__dataclass_fields__.items()
                }

                # Step 5: If parameters are provided in request then update keyword arguments used to instantiate indexer instance
                if request.indexer.parameters:
                    for parameter in request.indexer.parameters:
                        if parameter.parameter_id not in indexer_kwargs:
                            raise Error(
                                ErrorMessages.INVALID_PARAMETER.value.format(
                                    "indexer", parameter.parameter_id
                                )
                            )

                        indexer_kwargs[parameter.parameter_id] = parameter.value

                        # Re-map checkpoint kwarg to point to checkpoint file path in the service's store
                        if parameter.parameter_id == "checkpoint":
                            indexer_kwargs[
                                "checkpoint"
                            ] = self._store.get_checkpoint_path(
                                indexer_kwargs["checkpoint"]
                            )

                # Step 6: Update index specific arguments
                indexer_kwargs["index_root"] = self._store.get_index_directory_path(
                    index_information[ATTR_INDEX_ID]
                )
                indexer_kwargs["index_name"] = DIR_NAME_INDEX

                # Step 7: Create indexer instance
                try:
                    instance = IndexerFactory.get(indexer, indexer_kwargs)
                except (ValueError, TypeError) as err:
                    raise Error(err.args[0]) from err

                # Step 8: Save index information
                index_information[ATTR_ENGINE_TYPE] = instance.get_engine_type()
                self._store.save_index_information(
                    index_id=index_information[ATTR_INDEX_ID],
                    information=index_information,
                )

                # Step 9: Save documents used in index
                self._store.save_index_documents(
                    index_id=index_information[ATTR_INDEX_ID],
                    documents=request.documents,
                )

                # Step 10: Kick-off async index generation
                try:
                    instance.index(
                        self._store.get_index_documents_file_path(
                            index_id=index_information[ATTR_INDEX_ID]
                        ),
                    )

                    # Step 10.b: Set index status to "READY" once indexing is complete
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

                # Step 11: Return
                return index_information

            except Error as err:
                error_message = err.args[0]

                # Identify error code
                mobj = PATTERN_ERROR_MESSAGE.match(error_message)
                if mobj:
                    error_code = mobj.group(1).strip()
                    error_message = mobj.group(2).strip()
                else:
                    error_code = 500

                raise HTTPException(
                    status_code=500,
                    detail={"code": error_code, "message": error_message},
                ) from None

        @app.get(
            "/index/{index_id}/status",
            status_code=status.HTTP_200_OK,
            response_model=dict,
            tags=["Indexer"],
        )
        def get_index_status(index_id: str):
            try:
                return {
                    ATTR_STATUS: self._store.get_index_information(index_id=index_id)[
                        ATTR_STATUS
                    ]
                }
            except KeyError:
                return {ATTR_STATUS: IndexStatus.CORRUPT}
            except FileNotFoundError:
                return {ATTR_STATUS: IndexStatus.DOES_NOT_EXISTS}
            except Error as err:
                error_message = err.args[0]

                # Identify error code
                mobj = PATTERN_ERROR_MESSAGE.match(error_message)
                if mobj:
                    error_code = mobj.group(1).strip()
                    error_message = mobj.group(2).strip()
                else:
                    error_code = 500

                raise HTTPException(
                    status_code=500,
                    detail={"code": error_code, "message": error_message},
                ) from None

        ############################################################################################
        #                           Retriever API
        ############################################################################################
        @app.get(
            "/retrievers",
            status_code=status.HTTP_200_OK,
            response_model=List[Retriever],
            tags=["Retriever"],
        )
        def get_retrievers():
            return [
                {
                    "retriever_id": retriever_id,
                    "parameters": generate_parameters(retriever),
                }
                for retriever_id, retriever in RETRIEVERS_REGISTRY.items()
            ]

        @app.post(
            "/documents",
            status_code=status.HTTP_201_CREATED,
            response_model=List[List[Hit]],
            tags=["Retriever"],
        )
        def get_documents(request: RetrieveRequest):
            try:
                # Step 1: Verify requested retriever
                try:
                    retriever = RETRIEVERS_REGISTRY[request.retriever.retriever_id]
                except KeyError as err:
                    raise Error(
                        ErrorMessages.INVALID_RETRIEVER.value.format(
                            request.retriever.retriever_id,
                            ", ".join(RETRIEVERS_REGISTRY.keys()),
                        )
                    ) from err

                # Step 2: Load default retriever keyword arguments
                retriever_kwargs = {
                    k: v.default for k, v in retriever.__dataclass_fields__.items()
                }

                # Step 3: If parameters are provided in request then update keyword arguments used to instantiate retriever instance
                if request.retriever.parameters:
                    for parameter in request.retriever.parameters:
                        if parameter.parameter_id not in retriever_kwargs:
                            raise Error(
                                ErrorMessages.INVALID_PARAMETER.value.format(
                                    "retriever", parameter.parameter_id
                                )
                            )

                        retriever_kwargs[parameter.parameter_id] = parameter.value

                # Step 4: Load index information
                if request.index_id:
                    index_root = self._store.get_index_directory_path(request.index_id)
                    # Step 4.a: Check if `index_root` exists
                    if not self._store.exists(index_root):
                        raise Error(
                            ErrorMessages.FAILED_TO_LOCATE_INDEX.value.format(
                                request.index_id
                            )
                        )

                    # Step 4.b: Load index information
                    index_information = self._store.get_index_information(
                        index_id=request.index_id
                    )
                    if index_information[ATTR_STATUS] != IndexStatus.READY.value:
                        raise Error(
                            ErrorMessages.INDEX_UNAVAILABLE_FOR_QUERYING.value.format(
                                index_information[ATTR_STATUS]
                            )
                        )

                    # Step 4.c: Update index specific arguments
                    retriever_kwargs[
                        "index_root"
                    ] = self._store.get_index_directory_path(request.index_id)
                    retriever_kwargs["index_name"] = DIR_NAME_INDEX
                else:
                    raise Error(ErrorMessages.INVALID_REQUEST.value.format("index_id"))

                # Step 5: Create retriever instance
                try:
                    instance = RetrieverFactory.get(retriever, retriever_kwargs)
                except (ValueError, TypeError) as err:
                    raise Error(err.args[0]) from err

                # Step 6: Retrieve
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
                    results = instance.retrieve(
                        input_texts=request.queries, **retriever_kwargs
                    )
                    self._logger.info(
                        "Applying '%s' retriever for queries = %s returns results = %s",
                        instance.__class__.__name__,
                        request.queries,
                        results,
                    )
                except TypeError as err:
                    raise Error(
                        ErrorMessages.FAILED_TO_INITIALIZE.value.format(
                            f"{retriever.retriever_id} retriever"
                        )
                    ) from err

                # Step 7: Return
                hits = []
                for result_per_query in results:
                    hits_per_query = []
                    for hit in result_per_query:
                        try:
                            document = self._store.get_index_document(
                                index_id=request.index_id, document_idx=hit[0]
                            )
                            hits_per_query.append(
                                {
                                    "document": {
                                        "text": document["text"],
                                        "document_id": document["document_id"]
                                        if "document_id" in document
                                        else None,
                                        "title": document["title"]
                                        if "title" in document
                                        else None,
                                    },
                                    "score": hit[1],
                                }
                            )
                        except (FileNotFoundError, KeyError):
                            continue

                    hits.append(hits_per_query)

                return hits

            except Error as err:
                error_message = err.args[0]

                # Identify error code
                mobj = PATTERN_ERROR_MESSAGE.match(error_message)
                if mobj:
                    error_code = mobj.group(1).strip()
                    error_message = mobj.group(2).strip()
                else:
                    error_code = 500

                raise HTTPException(
                    status_code=500,
                    detail={"code": error_code, "message": error_message},
                ) from None

        ############################################################################################
        #                                   API SERVER CONFIGURATION
        ############################################################################################
        if self._config.require_ssl:
            server_config = uvicorn.Config(
                app,
                host=self._config.rest_host,
                port=self._config.rest_port,
                workers=self._config.num_rest_server_workers,
                ssl_keyfile=self._config.tls_server_key,
                ssl_certfile=self._config.tls_server_cert,
                ssl_ca_certs=self._config.tls_ca_cert,
            )
        else:
            server_config = uvicorn.Config(
                app,
                host=self._config.rest_host,
                port=self._config.rest_port,
                workers=self._config.num_rest_server_workers,
            )

        # Create and run server
        try:
            uvicorn.Server(server_config).run()
            self._logger.info(
                "Server instance started on port %s - initialization took %s seconds",
                self._config.rest_port,
                time.time() - start_t,
            )
        except Exception as ex:
            self._logger.exception("Error starting server: %s", ex)
            raise
