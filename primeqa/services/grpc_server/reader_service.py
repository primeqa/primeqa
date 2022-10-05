import logging
from typing import Union

from grpc import ServicerContext, StatusCode

from primeqa.services.configurations import Settings
from primeqa.services.grpc_server.utils import parse_parameter_value
from primeqa.services.grpc_server.grpc_generated.reader_pb2_grpc import ReaderServicer
from primeqa.services.grpc_server.grpc_generated.reader_pb2 import (
    GetAnswersRequest,
    GetAnswersResponse,
    AnswersForPassage,
    Answer,
)

from primeqa.pipelines import get_pipeline, load_pipeline, ReaderPipeline


class ReaderService(ReaderServicer):
    def __init__(self, config: Settings, logger: Union[logging.Logger, None] = None):
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger
        self._config = config
        self._logger.info("%s is successfully initialized.", self.__class__.__name__)

    def _log_question(self, procedure_name: str, question: str):
        self._logger.debug(
            "%s Prediction request of question: '%s'", procedure_name, question
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
        self._log_question("GetAnswers", request.question)
        # Step 1: Get requested pipeline
        pipeline = get_pipeline(pipeline_id=request.pipeline.pipeline_id)

        # Step 2: Verify requested pipeline's type
        if pipeline.pipeline_type != ReaderPipeline.__name__:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"Invalid pipeline type: {pipeline.pipeline_type}. Only pipelines with type: {ReaderPipeline.__name__} are applicable."
            )
            return GetAnswersResponse()

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
                        return GetAnswersResponse()

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
                    return GetAnswersResponse()

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

            # Step 3.c: Load pipeline
            pipeline.load()
        else:
            # Step 3: Activate if existing pipeline, if not active already
            load_pipeline(pipeline.pipeline_id)

        predictions = pipeline.apply(
            input_texts=[request.question] * len(request.passages),
            context=[[passage] for passage in request.passages],
        )

        return GetAnswersResponse(
            answers=[
                AnswersForPassage(
                    answers=[
                        Answer(
                            text=prediction["span_answer_text"],
                            start_char_offset=prediction["span_answer"][
                                "start_position"
                            ],
                            end_char_offset=prediction["span_answer"]["end_position"],
                            confidence_score=prediction["confidence_score"],
                            passage_index=int(prediction["example_id"]),
                        )
                        for prediction in predictions_for_passage
                    ]
                )
                for predictions_for_passage in predictions
            ]
        )
