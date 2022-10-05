from typing import Union
from primeqa.services.grpc_server.grpc_generated.pipelines_pb2 import PipelineParameter


def parse_parameter_value(
    parameter: PipelineParameter, _type: str
) -> Union[int, float, bool, str, None]:
    if _type == "String":
        return parameter.value.string_value

    if _type == "Numeric":
        if parameter.value.number_value.is_integer():
            return int(parameter.value.number_value)
        else:
            return parameter.value.number_value

    if _type == "Boolean":
        return parameter.value.bool_value

    return None
