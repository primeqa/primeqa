import logging
from typing import Optional

from grpc import ServicerContext

from primeqa.services.configurations import Settings
from primeqa.services.grpc_server.grpc_generated.pipelines_pb2_grpc import (
    PipelinesServicer,
)
from primeqa.services.grpc_server.grpc_generated.pipelines_pb2 import (
    Pipeline,
    PipelineParameter,
    GetPipelinesRequest,
    GetPipelinesResponse,
    GetPipelineRequest,
    Value,
)

from primeqa.pipelines import get_pipelines, get_pipeline
from primeqa.services.exceptions import ErrorMessages


class PipelinesService(PipelinesServicer):
    def __init__(self, config: Settings, logger: Optional[logging.Logger] = None):
        if logger is None:
            self._logger = logging.getLogger(self.__class__.__name__)
        else:
            self._logger = logger
        self._config = config
        self._logger.info("%s is successfully initialized.", self.__class__.__name__)

    def _build_pipeline_parameter_obj(self, parameter: dict) -> PipelineParameter:
        options_arg = None

        if parameter["type"] == "Numeric":
            value = Value(number_value=parameter["value"])
        elif parameter["type"] == "String":
            value = Value(string_value=parameter["value"])
            if "options" in parameter and parameter["options"]:
                options_arg = [
                    Value(string_value=choice) for choice in parameter["options"]
                ]
        elif parameter["type"] == "Boolean":
            value = Value(bool_value=parameter["value"])
            if "options" in parameter and parameter["options"]:
                options_arg = [
                    Value(bool_value=choice) for choice in parameter["options"]
                ]
        else:
            error_msg = ErrorMessages.format({parameter["value"]})
            self._logger.exception(error_msg)
            raise ValueError(error_msg)

        return PipelineParameter(
            parameter_id=parameter["parameter_id"],
            name=parameter["name"],
            type=parameter["type"],
            value=value,
            options=options_arg if options_arg else None,
            range=parameter["range"]
            if "range" in parameter and parameter["range"]
            else None,
        )

    def GetPipelines(self, request: GetPipelinesRequest, context: ServicerContext):
        """
        :param GetPipelinesRequest request:
        :param ServicerContext context: gRPC context information for method call
        :return: Pipelines
        :rtype: GetPipelinesResponse
        """
        return GetPipelinesResponse(
            pipelines=[
                Pipeline(
                    pipeline_id=pipeline.pipeline_id,
                    name=pipeline.pipeline_name,
                    type=pipeline.pipeline_type,
                    parameters=[
                        self._build_pipeline_parameter_obj(parameter)
                        for parameter in pipeline.parameters.values()
                    ]
                    if request.with_parameters
                    else None,
                )
                for pipeline in get_pipelines()
            ]
        )

    def GetPipeline(self, request: GetPipelineRequest, context: ServicerContext):
        """
        :param GetPipelineRequest request:
        :param ServicerContext context: gRPC context information for method call
        :return: Pipeline
        """
        pipeline = get_pipeline(request.pipeline_id)
        return Pipeline(
            pipeline_id=pipeline.pipeline_id,
            name=pipeline.pipeline_name,
            type=pipeline.pipeline_type,
            parameters=[
                self._build_pipeline_parameter_obj(parameter)
                for parameter in pipeline.parameters.values()
            ]
            if request.with_parameters
            else None,
        )
