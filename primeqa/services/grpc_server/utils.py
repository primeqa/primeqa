from typing import Union, List, Dict, Any
from dataclasses import MISSING
from math import inf
from google.protobuf.struct_pb2 import Value

from primeqa.components.base import Component
from primeqa.services.grpc_server.grpc_generated.parameter_pb2 import Parameter
from primeqa.services.parameters import get_parameters


def generate_parameters(component: Component, skip: List[str] = None) -> Dict[str, Any]:
    grpc_parameters = []
    for parameter_dict in get_parameters(component):
        # Step 1: Exclude parameters provided in skip list
        if skip and parameter_dict["id"] in skip:
            continue

        grpc_parameter = Parameter(
            parameter_id=parameter_dict["id"], name=parameter_dict["name"]
        )
        try:
            grpc_parameter.description = parameter_dict["description"]
        except KeyError:
            # If "description" doesn't exist, continue processing other fields
            pass

        if parameter_dict["type"] == str:
            grpc_parameter.type = "String"
            grpc_parameter.value.CopyFrom(
                Value(
                    string_value=parameter_dict["value"]
                    if parameter_dict["value"] != MISSING
                    else "** REQUIRED **"
                )
            )
            try:
                grpc_parameter.options.extend(
                    [Value(string_value=entry) for entry in parameter_dict["options"]]
                )
            except KeyError:
                # if "options" aren't provide, continue processing other parameters
                pass
        elif parameter_dict["type"] == int or parameter_dict["type"] == float:
            grpc_parameter.type = "Numeric"
            grpc_parameter.value.CopyFrom(
                Value(
                    number_value=parameter_dict["value"]
                    if parameter_dict["value"] != MISSING
                    else -inf
                )
            )
            try:
                grpc_parameter.range.extend(parameter_dict["range"])
            except KeyError:
                # if "range" isn't provide, continue processing other parameters
                pass
            try:
                grpc_parameter.options.extend(
                    [Value(number_value=entry) for entry in parameter_dict["options"]]
                )
            except KeyError:
                # if "options" isn't provide, continue processing other parameters
                pass
        elif parameter_dict["type"] == bool:
            grpc_parameter.type = "Boolean"
            grpc_parameter.value.CopyFrom(
                Value(
                    bool_value=parameter_dict["value"]
                    if parameter_dict["value"] != MISSING
                    else True
                )
            )
            try:
                grpc_parameter.options.extend(
                    [Value(bool_value=entry) for entry in parameter_dict["options"]]
                )
            except KeyError:
                # if "options" aren't provide, continue processing other parameters
                pass
        else:
            raise ValueError(f"Unsupported parameter type: {parameter_dict['type']}.")

        grpc_parameters.append(grpc_parameter)

    return grpc_parameters


def parse_parameter_value(
    parameter: Parameter, _type
) -> Union[int, float, bool, str, None]:
    if _type == str:
        return parameter.value.string_value

    if _type == int or _type == float:
        if parameter.value.number_value.is_integer():
            return int(parameter.value.number_value)
        else:
            return parameter.value.number_value

    if _type == bool:
        return parameter.value.bool_value

    if _type == type(None):
        return parameter.value.null_value

    return None
