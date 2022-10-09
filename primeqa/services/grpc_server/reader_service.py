import logging
from typing import Union

from grpc import ServicerContext, StatusCode

from primeqa.services.configurations import Settings
from primeqa.services.parameters import get_parameter_type
from primeqa.services.factories import READERS_REGISTRY, ReaderFactory
from primeqa.services.grpc_server.utils import (
    parse_parameter_value,
    generate_parameters,
)
from primeqa.services.grpc_server.grpc_generated.reader_pb2_grpc import ReaderServicer
from primeqa.services.grpc_server.grpc_generated.reader_pb2 import (
    ReaderComponent,
    GetReadersRequest,
    GetReadersResponse,
    GetAnswersRequest,
    Answer,
    AnswersForContext,
    AnswersForQuery,
    GetAnswersResponse,
)


class ReaderService(ReaderServicer):
    def __init__(self, config: Settings, logger: Union[logging.Logger, None] = None):
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger
        self._config = config
        self.loaded_readers = {}
        self._logger.info("%s is successfully initialized.", self.__class__.__name__)

    def _log_question(self, procedure_name: str, question: str):
        self._logger.debug(
            "%s Prediction request of question: '%s'", procedure_name, question
        )

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
        return GetReadersResponse(
            readers=[
                ReaderComponent(
                    reader_id=reader_id, parameters=generate_parameters(reader)
                )
                for reader_id, reader in READERS_REGISTRY.items()
            ]
        )

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
                f"If contexts are provided, number of contexts({len(request.contexts)}) must match number of queries({len(request.queries)})"
            )
            return GetAnswersResponse()

        # Step 2: Verify requested reader
        try:
            reader = READERS_REGISTRY[request.reader.reader_id]
        except KeyError:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"Invalid reader: {request.reader.reader_id}. Please select one of the following pre-defined readers: {', '.join(READERS_REGISTRY.keys())}"
            )
            return GetAnswersResponse()

        # Step 3: Load default reader keyword arguments
        reader_kwargs = {k: v.default for k, v in reader.__dataclass_fields__.items()}

        # Step 4: If parameters are provided in request then update keyword arguments used to instantiate reader instance
        if request.reader.parameters:
            for parameter in request.reader.parameters:
                try:
                    reader_kwargs[parameter.parameter_id] = parse_parameter_value(
                        parameter,
                        get_parameter_type(
                            component=reader, parameter_id=parameter.parameter_id
                        ),
                    )
                except KeyError:
                    context.set_code(StatusCode.INVALID_ARGUMENT)
                    context.set_details(
                        f"Invalid reader parameter: {parameter.parameter_id}. Only pre-defined parameters can be modified."
                    )
                    return GetAnswersResponse()
        try:
            instance = ReaderFactory.get(reader, reader_kwargs)
        except ValueError as err:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(err.args[0])
            return GetAnswersResponse()

        # Step 5: Run apply method
        answers_response = GetAnswersResponse()
        try:
            for idx, query in enumerate(request.queries):
                # Step 5.a: Run "apply" per query
                predictions = instance.apply(
                    input_texts=[query] * len(request.contexts[idx].texts),
                    context=[[text] for text in request.contexts[idx].texts],
                )

                # Step 5.b: Add answers for current query into response object
                answers_response.query_answers.append(
                    AnswersForQuery(
                        context_answers=[
                            AnswersForContext(
                                answers=[
                                    Answer(
                                        text=prediction["span_answer_text"],
                                        start_char_offset=prediction["span_answer"][
                                            "start_position"
                                        ],
                                        end_char_offset=prediction["span_answer"][
                                            "end_position"
                                        ],
                                        confidence_score=prediction["confidence_score"],
                                        passage_index=int(prediction["example_id"]),
                                    )
                                    for prediction in predictions_for_context
                                ]
                            )
                            for predictions_for_context in predictions
                        ]
                    )
                )
        except IndexError:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(f"Missing contexts for query: {query}")
            return GetAnswersResponse()

        # Step 6: Return
        return answers_response
