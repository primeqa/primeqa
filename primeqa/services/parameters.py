from primeqa.pipelines.components.base import Component


def get_parameters(component: Component):
    parameters = []
    for key, field in component.__dataclass_fields__.items():
        parameter = {
            "id": key,
            "type": field.type,
            "value": field.default,
        }
        for k, v in field.metadata.items():
            parameter[k] = v
        parameters.append(parameter)

    return parameters


def get_parameter_type(component: Component, parameter_id: str):
    return component.__dataclass_fields__[parameter_id].type
