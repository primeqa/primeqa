import logging
from typing import Union

from primeqa.components.reader.extractive import ExtractiveReader

from grpc import ServicerContext, StatusCode

from primeqa.components.reader.extractive import ExtractiveReader
from primeqa.services.exceptions import Error, ErrorMessages
from primeqa.services.configurations import Settings
from primeqa.services.grpc_server.utils import (
    parse_parameter_value,
    generate_parameters,
)
from primeqa.services.parameters import get_parameter_type
from primeqa.services.factories import (
    READERS_REGISTRY,
    ReaderFactory,
)
from primeqa.services.grpc_server.grpc_generated.reader_pb2_grpc import (
    ReadingServiceServicer,
)
from primeqa.services.grpc_server.grpc_generated.reader_pb2 import (
    GetReadersRequest,
    GetReadersResponse,
    Reader,
    GetAnswersRequest,
    Answer,
    AnswersForContext,
    AnswersForQuery,
    GetAnswersResponse,
    Evidence,
    Offset,
)


class ReaderService(ReadingServiceServicer):
    def __init__(self, config: Settings, logger: Union[logging.Logger, None] = None):
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger
        self._config = config
        self.loaded_readers = {}
        self._logger.info("%s is successfully initialized.", self.__class__.__name__)

    def GetReaders(
        self, request: GetReadersRequest, context: ServicerContext
    ) -> GetReadersResponse:
        """_summary_

        Args:
            request (GetReadersRequest):
            context (ServicerContext): gRPC context information for method call

        Returns:
            GetReadersResponse: List of available readers
        """
        try:
            return GetReadersResponse(
                readers=[
                    Reader(reader_id=reader_id, parameters=generate_parameters(reader))
                    for reader_id, reader in READERS_REGISTRY.items()
                ]
            )
        except Error as err:
            context.set_code(StatusCode.INTERNAL)
            context.set_details(err.args[0])
            return GetAnswersResponse()

    def GetAnswers(
        self, request: GetAnswersRequest, context: ServicerContext
    ) -> GetAnswersResponse:
        """
        :param GetAnswersRequest request:
        :param ServicerContext context: gRPC context information for method call
        :return: Found answers
        :rtype: GetAnswersResponse
        """
        # Step 1: If contexts are provided, number of contexts need to match number of queries
        if request.contexts and len(request.queries) != len(request.contexts):
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(
                ErrorMessages.MISSING_CONTEXT.value.format(
                    len(request.contexts), len(request.queries)
                )
            )
            return GetAnswersResponse()

        # Step 2: Verify requested reader
        try:
            reader = READERS_REGISTRY[request.reader.reader_id]
        except KeyError:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(
                ErrorMessages.INVALID_READER.value.format(
                    request.reader.reader_id, ", ".join(READERS_REGISTRY.keys())
                )
            )
            return GetAnswersResponse()

        # Step 3: Load default reader keyword arguments
        reader_kwargs = {
            k: v.default for k, v in reader.__dataclass_fields__.items() if v.init
        }

        # Step 4: If parameters are provided in request then update keyword arguments used to instantiate reader instance
        if request.reader.parameters:
            for parameter in request.reader.parameters:
                if parameter.parameter_id not in reader_kwargs:
                    context.set_code(StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        ErrorMessages.INVALID_PARAMETER.value.format(
                            "reader", parameter.parameter_id
                        )
                    )
                    return GetAnswersResponse()

                reader_kwargs[parameter.parameter_id] = parse_parameter_value(
                    parameter,
                    get_parameter_type(
                        component=reader, parameter_id=parameter.parameter_id
                    ),
                )

        try:
            instance = ReaderFactory.get(reader, reader_kwargs)
        except (ValueError, TypeError) as err:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(err.args[0])
            return GetAnswersResponse()

        # Step 5: Run apply method
        instance_fields = [
            k
            for k, v in instance.__class__.__dataclass_fields__.items()
            if not "exclude_from_hash" in v.metadata
            or not v.metadata["exclude_from_hash"]
        ]
        answers_response = GetAnswersResponse()
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
                    if isinstance(instance, ExtractiveReader):
                        predictions = instance.predict(
                            questions=[query] * len(request.contexts[idx].texts),
                            contexts=[[text] for text in request.contexts[idx].texts],
                            example_ids=[
                                str(example_id)
                                for example_id in range(
                                    1, len(request.contexts[idx].texts) + 1
                                )
                            ],
                            **reader_kwargs,
                        )
                        self._logger.info(
                            "Applying '%s' reader for query = '%s' returns predictions = %s",
                            instance.__class__.__name__,
                            query,
                            predictions,
                        )

                        # Step 5.b: Add answers for current query into response object
                        answers_response.query_answers.append(
                            AnswersForQuery(
                                context_answers=[
                                    AnswersForContext(
                                        answers=[
                                            Answer(
                                                text=prediction["span_answer_text"],
                                                confidence_score=prediction[
                                                    "confidence_score"
                                                ],
                                                evidences=[
                                                    Evidence(
                                                        context_index=int(
                                                            prediction["example_id"]
                                                        ),
                                                        offsets=[
                                                            Offset(
                                                                start=prediction[
                                                                    "span_answer"
                                                                ]["start_position"],
                                                                end=prediction[
                                                                    "span_answer"
                                                                ]["end_position"],
                                                            )
                                                        ],
                                                    )
                                                ],
                                            )
                                            for prediction in predictions_for_context
                                        ]
                                    )
                                    for predictions_for_context in predictions.values()
                                ]
                            )
                        )
                    else:
                        # This is a generative reader
                        predictions = instance.predict(
                            questions=[query],
                            contexts=[request.contexts[idx].texts],
                            **reader_kwargs,
                        )
                        self._logger.info(
                            "Applying '%s' reader for query = '%s' returns predictions = %s",
                            instance.__class__.__name__,
                            query,
                            predictions,
                        )
                        # Step 5.b: Add answers for current query into response object
                        answers_response.query_answers.append(
                            AnswersForQuery(
                                context_answers=[
                                    AnswersForContext(
                                        answers=[
                                            Answer(
                                                text=prediction["span_answer_text"],
                                                confidence_score=prediction[
                                                    "confidence_score"
                                                ],
                                                evidences=[
                                                    Evidence(
                                                        context_index=context_index + 1
                                                    )
                                                    for context_index in range(
                                                        len(
                                                            request.contexts[
                                                                prediction["example_id"]
                                                            ].texts
                                                        )
                                                    )
                                                ],
                                            )
                                            for prediction in predictions_for_context
                                        ]
                                    )
                                    for predictions_for_context in predictions.values()
                                ]
                            )
                        )
                except AssertionError:
                    context.set_code(StatusCode.INTERNAL)
                    context.set_details(ErrorMessages.INVALID_READER_INPUT.value)
                    return GetAnswersResponse()

                except TypeError:
                    context.set_code(StatusCode.INTERNAL)
                    context.set_details(
                        ErrorMessages.FAILED_TO_INITIALIZE.value.format(
                            f"{request.reader.reader_id} reader"
                        )
                    )
                    return GetAnswersResponse()

        except IndexError:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(
                ErrorMessages.MISSING_CONTEXT.value.format(
                    len(request.contexts), len(request.queries)
                )
            )
            return GetAnswersResponse()

        # Step 6: Return
        return answers_response
