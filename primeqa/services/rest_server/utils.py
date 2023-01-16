from typing import List, Dict, Any
from dataclasses import MISSING
from math import inf


from primeqa.components.base import Component
from primeqa.services.parameters import get_parameters


def generate_parameters(component: Component, skip: List[str] = None) -> Dict[str, Any]:
    rest_parameters = []
    for parameter_dict in get_parameters(component):
        # Step 1: Exclude parameters provided in skip list
        if skip and parameter_dict["id"] in skip:
            continue

        rest_parameter = {
            "parameter_id": parameter_dict["id"],
            "name": parameter_dict["name"],
            "description": parameter_dict["description"]
            if "description" in parameter_dict
            else None,
            "options": parameter_dict["options"]
            if "options" in parameter_dict
            else None,
            "range": parameter_dict["range"] if "range" in parameter_dict else None,
        }

        if parameter_dict["type"] == str:
            rest_parameter["type"] = "String"
            rest_parameter["value"] = (
                parameter_dict["value"]
                if parameter_dict["value"] != MISSING
                else "** REQUIRED **"
            )
        elif parameter_dict["type"] == int or parameter_dict["type"] == float:
            rest_parameter["type"] = "Numeric"
            rest_parameter["value"] = (
                parameter_dict["value"] if parameter_dict["value"] != MISSING else -inf
            )
        elif parameter_dict["type"] == bool:
            rest_parameter["type"] = "Boolean"
            rest_parameter["value"] = (
                parameter_dict["value"] if parameter_dict["value"] != MISSING else True
            )

        else:
            raise ValueError(f"Unsupported parametr type: {parameter_dict['type']}.")

        rest_parameters.append(rest_parameter)

    return rest_parameters
