import logging
import time
from typing import List, Union


import uvicorn
from fastapi import FastAPI, status, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from primeqa.services.configurations import Settings
from primeqa.services.parameters import get_parameter_type
from primeqa.services.constants import ATTR_STATUS, IndexStatus
from primeqa.services.factories import INDEXERS_REGISTRY, IndexerFactory
from primeqa.services.grpc_server.utils import (
    parse_parameter_value,
    generate_parameters,
)

from primeqa.services.rest_server.data_models import (
    Pipeline,
    ReaderQuery,
    Answer,
    IndexRequest,
    IndexInformation,
    RetrieverQuery,
    Hit,
)
from primeqa.services.exceptions import PATTERN_ERROR_MESSAGE, Error, ErrorMessages
from primeqa.services.store import StoreFactory, Store
from primeqa.services.constants import IndexStatus


# def index(
#     store: Store,
#     index_id: str,
#     pipeline: RetrieverPipeline,
#     documents_to_index: List[dict],
# ):
#     index_information = store.get_index_information(index_id)
#     try:
#         pipeline.index(documents_to_index, store.get_index_directory_path(index_id))
#         index_information["status"] = IndexStatus.READY
#     except RuntimeError as err:
#         index_information["status"] = IndexStatus.CORRUPT
#         logging.exception(
#             "Generation failed for index with id=%s. Resultant index may be corrupted.",
#             index_id,
#         )
#         logging.exception(err.args[0])

#     store.save_index_information(index_id, information=index_information)


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
            version="0.9.1",
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
        #                           Pipelines API
        ############################################################################################
        @app.get(
            "/pipelines",
            status_code=status.HTTP_200_OK,
            response_model=List[Pipeline],
            tags=["Pipelines"],
        )
        def fetch_pipelines(with_parameters: bool = True):
            try:
                return [
                    {
                        "pipeline_id": pipeline.pipeline_id,
                        "name": pipeline.pipeline_name,
                        "type": pipeline.pipeline_type,
                        "parameters": [
                            {
                                "parameter_id": parameter["parameter_id"],
                                "name": parameter["name"],
                                "type": parameter["type"],
                                "value": parameter["value"],
                                "options": parameter["options"]
                                if "options" in parameter and parameter["options"]
                                else None,
                                "range": parameter["range"]
                                if "range" in parameter and parameter["range"]
                                else None,
                            }
                            for parameter in pipeline.parameters.values()
                        ]
                        if with_parameters
                        else None,
                    }
                    for pipeline in get_pipelines()
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

        @app.get(
            "/pipelines/{pipeline_id}",
            status_code=status.HTTP_200_OK,
            response_model=Pipeline,
            tags=["Pipelines"],
        )
        def fetch_pipeline(pipeline_id: str, with_parameters: bool = True):
            try:
                pipeline = get_pipeline(pipeline_id=pipeline_id)
                return {
                    "pipeline_id": pipeline.pipeline_id,
                    "name": pipeline.pipeline_name,
                    "type": pipeline.pipeline_type,
                    "parameters": [
                        {
                            "parameter_id": parameter["parameter_id"],
                            "name": parameter["name"],
                            "type": parameter["type"],
                            "value": parameter["value"],
                            "options": parameter["options"]
                            if "options" in parameter and parameter["options"]
                            else None,
                            "range": parameter["range"]
                            if "range" in parameter and parameter["range"]
                            else None,
                        }
                        for parameter in pipeline.parameters.values()
                    ]
                    if with_parameters
                    else None,
                }

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
        #                           Answers API
        ############################################################################################
        @app.post(
            "/answers",
            status_code=status.HTTP_201_CREATED,
            response_model=List[List[Answer]],
            tags=["Reader"],
        )
        def get_answers(request: ReaderQuery):
            try:
                # Step 1: Fetch requested pipeline
                pipeline = get_pipeline(pipeline_id=request.pipeline.pipeline_id)

                # Step 2: Verify requested pipeline's type
                if pipeline.pipeline_type != ReaderPipeline.__name__:
                    raise Error(
                        ErrorMessages.INVALID_PIPELINE_TYPE.format(
                            pipeline.pipeline_type, ReaderPipeline.__name__
                        )
                    )

                # Step 3: If parameters are provided in request and they differ from existing pipeline, create new pipeline object
                if request.pipeline.parameters:
                    # Step 3.a: Create new pipeline object, if necessary
                    for parameter in request.pipeline.parameters:
                        try:
                            # Step 3.a.i: Compare against existing parameter value
                            if (
                                pipeline.get_parameter_value(parameter.parameter_id)
                                != parameter.value
                            ):
                                pipeline = get_pipeline(
                                    pipeline_id=request.pipeline.pipeline_id,
                                    invoke__init__=True,
                                )
                                break
                        except KeyError as err:
                            raise Error(
                                ErrorMessages.INVALID_PIPELINE_PARAMETER.format(
                                    parameter.parameter_id
                                )
                            ) from err

                    # Step 3.b: Set different parameter values in new pipeline object
                    for parameter in request.pipeline.parameters:
                        # Step 3.b.i: Update different parameter value
                        if (
                            pipeline.get_parameter_value(parameter.parameter_id)
                            != parameter.value
                        ):
                            pipeline.set_parameter_value(
                                parameter_id=parameter.parameter_id,
                                parameter_value=parameter.value,
                            )

                    # Step 3.c: Load pipeline
                    pipeline.load()
                else:
                    # Step 3: Load if existing pipeline, if not active already
                    load_pipeline(pipeline.pipeline_id)

                # Step 4: Run apply method
                predictions = pipeline.apply(
                    input_texts=[request.question] * len(request.passages),
                    context=[[passage] for passage in request.passages],
                )

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
                            "passage_index": int(prediction["example_id"]),
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
        #                           Index API
        ############################################################################################
        @app.post(
            "/indexs",
            status_code=status.HTTP_201_CREATED,
            response_model=IndexInformation,
            tags=["Indexer"],
        )
        def generate_index(request: IndexRequest):
            try:
                # Step 1: Assign unique index id
                index_information = {
                    "index_id": self._store.generate_index_uuid(),
                    "status": IndexStatus.INDEXING,
                }

                # Step 2: Remove existing index if index_id is provide in the request
                if request.index_id:
                    self._store.delete_index(request.index_id)
                    index_information["index_id"] = request.index_id

                # Step 3: Validate requested pipeline's type
                pipeline = get_pipeline(pipeline_id=request.pipeline.pipeline_id)
                if pipeline.pipeline_type != RetrieverPipeline.__name__:
                    raise Error(
                        ErrorMessages.INVALID_PIPELINE_TYPE.format(
                            pipeline.pipeline_type, RetrieverPipeline.__name__
                        )
                    )

                # Step 4: If parameters are provided in request and they differ from existing pipeline, create new pipeline object
                if request.pipeline.parameters:
                    # Step 4.a: Create new pipeline object, if necessary
                    for parameter in request.pipeline.parameters:
                        try:
                            # Step 4.a.i: Compare against existing parameter value
                            if (
                                pipeline.get_parameter_value(parameter.parameter_id)
                                != parameter.value
                            ):
                                pipeline = get_pipeline(
                                    pipeline_id=request.pipeline.pipeline_id,
                                    invoke__init__=True,
                                )
                                break
                        except KeyError as err:
                            raise Error(
                                ErrorMessages.INVALID_PIPELINE_PARAMETER.format(
                                    parameter.parameter_id
                                )
                            ) from err

                    # Step 4.b: Set different parameter values in new pipeline object
                    for parameter in request.pipeline.parameters:
                        # Step 4.b.i: Update different parameter value
                        if (
                            pipeline.get_parameter_value(parameter.parameter_id)
                            != parameter.value
                        ):
                            pipeline.set_parameter_value(
                                parameter_id=parameter.parameter_id,
                                parameter_value=parameter.value,
                            )

                    # Step 4.c: Load latest checkpoint, if available else load default model
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
                    # Step 4.a: Load latest checkpoint, if available else load default model
                    try:
                        checkpoint = self._store.get_latest_checkpoint_path(
                            pipeline.pipeline_id
                        )
                    except IndexError:
                        checkpoint = self._store.get_model_path(
                            pipeline.get_parameter_value("model")
                        )

                    # Step 4.b: Activate if existing pipeline, if not active already
                    load_pipeline(pipeline.pipeline_id, checkpoint=checkpoint)

                # Step 5: Update index information with pipeline information
                index_information["metadata"] = {"pipeline": pipeline.serialize()}

                # Step 6: Save index information
                self._store.save_index_information(
                    index_id=index_information["index_id"],
                    information=index_information,
                )

                # Step 7: Save documents used in index
                self._store.save_index_documents(
                    index_id=index_information["index_id"], documents=request.documents
                )

                # Step 8: Kick-off async index generation

                # Step 8: Return
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
                index_information = self._store.get_index_information(index_id=index_id)
                if index_information["status"] == IndexStatus.READY:
                    return {"status": IndexStatus.READY}
                elif index_information["status"] == IndexStatus.INDEXING:
                    return {"status": IndexStatus.INDEXING}
                else:
                    return {"status": IndexStatus.CORRUPT}
            except KeyError:
                return {"status": IndexStatus.CORRUPT}
            except FileNotFoundError:
                return {"status": IndexStatus.DOES_NOT_EXISTS}
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
        #                           Documents API
        ############################################################################################
        @app.post(
            "/documents",
            status_code=status.HTTP_201_CREATED,
            response_model=List[List[Hit]],
            tags=["Retriever"],
        )
        def get_documents(request: RetrieverQuery):
            try:
                # Step 1: Fetch requested pipeline
                pipeline = get_pipeline(pipeline_id=request.pipeline.pipeline_id)

                # Step 2: Verify requested pipeline's type
                if pipeline.pipeline_type != RetrieverPipeline.__name__:
                    raise Error(
                        ErrorMessages.INVALID_PIPELINE_TYPE.format(
                            pipeline.pipeline_type, RetrieverPipeline.__name__
                        )
                    )

                # Step 3: If parameters are provided in request and they differ from existing pipeline, create new pipeline object
                if request.pipeline.parameters:
                    # Step 3.a: Create new pipeline object, if necessary
                    for parameter in request.pipeline.parameters:
                        try:
                            # Step 3.a.i: Compare against existing parameter value
                            if (
                                pipeline.get_parameter_value(parameter.parameter_id)
                                != parameter.value
                            ):
                                pipeline = get_pipeline(
                                    pipeline_id=request.pipeline.pipeline_id,
                                    invoke__init__=True,
                                )
                                break
                        except KeyError as err:
                            raise Error(
                                ErrorMessages.INVALID_PIPELINE_PARAMETER.format(
                                    parameter.parameter_id
                                )
                            ) from err

                    # Step 3.b: Set different parameter values in new pipeline object
                    for parameter in request.pipeline.parameters:
                        # Step 3.b.i: Update different parameter value
                        if (
                            pipeline.get_parameter_value(parameter.parameter_id)
                            != parameter.value
                        ):
                            pipeline.set_parameter_value(
                                parameter_id=parameter.parameter_id,
                                parameter_value=parameter.value,
                            )

                # Step 4: Run retrieve method
                results = pipeline.retrieve(
                    input_texts=request.queries,
                    index_path=self._store.get_index_file_path(request.index_id),
                )

                # Step 5: Return
                hits = []
                for result_per_query in results:
                    hits_per_query = []
                    for hit in result_per_query:
                        try:
                            hits_per_query.append(
                                {
                                    "document": self._store.get_index_document(
                                        index_id=request.index_id, document_idx=hit[0]
                                    ),
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
                self._config.grpc_port,
                time.time() - start_t,
            )
        except Exception as ex:
            self._logger.exception("Error starting server: %s", ex)
            raise
