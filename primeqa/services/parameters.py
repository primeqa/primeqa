from primeqa.components.base import Component
from primeqa.services.exceptions import Error, ErrorMessages


def get_parameters(component: Component):
    parameters = []
    for key, field in component.__dataclass_fields__.items():
        # Step 1: Skip parameters which are not supported via APIs
        if "api_support" not in field.metadata or not field.metadata["api_support"]:
            continue

        try:
            parameter = {
                "id": key,
                "type": field.type,
                "value": field.default,
            }
        except KeyError as err:
            raise Error(
                ErrorMessages.INVALID_PARAMETER_DEFINITION.value.format(key).strip()
            ) from err
        for k, v in field.metadata.items():
            parameter[k] = v
        parameters.append(parameter)

    return parameters


def get_parameter_type(component: Component, parameter_id: str):
    return component.__dataclass_fields__[parameter_id].type
