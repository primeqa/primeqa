import logging
from typing import Union

from grpc import ServicerContext, StatusCode

from primeqa.services.configurations import Settings
from primeqa.services.grpc_server.grpc_generated.reader_pb2_grpc import ReaderServicer
from primeqa.services.grpc_server.grpc_generated.reader_pb2 import (
    GetAnswersRequest,
    GetAnswersResponse,
    AnswersForPassage,
    Answer,
)

from primeqa.pipelines import get_pipeline, activate_pipeline, ReaderPipeline


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
        pipeline = get_pipeline(pipeline_id=request.pipeline.pipeline_id)

        # Activate pipeline
        if pipeline.pipeline_type != ReaderPipeline.__name__:
            context.set_code(StatusCode.INVALID_ARGUMENT)
            context.set_details(
                f"Invalid pipeline type: {pipeline.pipeline_type}. Only pipelines with type: {ReaderPipeline.__name__} are applicable."
            )
            return GetAnswersResponse()
        else:
            activate_pipeline(pipeline.pipeline_id)

        predictions = pipeline.apply(
            input_texts=[request.question] * len(request.passages),
            context=[[passage] for passage in request.passages],
        )

        filtered_predictions = []
        for predictions_for_passage in predictions:
            filtered_predictions_for_passage = []
            for prediction in predictions_for_passage:
                if prediction["confidence_score"] >= pipeline.get_parameter_value(
                    "min_score_threshold"
                ):
                    filtered_predictions_for_passage.append(prediction)

            filtered_predictions.append(filtered_predictions_for_passage)

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
                for predictions_for_passage in filtered_predictions
            ]
        )
