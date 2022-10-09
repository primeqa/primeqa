import re
from enum import Enum

PATTERN_ERROR_MESSAGE = re.compile("(E[0-9]{4,5}):(.*)")


class Error(Exception):
    pass


class ErrorMessages(str, Enum):
    # INDEX
    FAILED_TO_LOCATE_INDEX = "E3001: Failed to locate index : {}"

    # PIPELINES
    UNSUPPORTED_PARAMETER_TYPE = "E5001: Unsupported parameter type: {}"
    INVALID_PIPELINE_TYPE = (
        "E5002: Invalid pipeline type: {}. Only pipelines of type: {} are applicable."
    )
    INVALID_PIPELINE_PARAMETER = "E5003: Invalid pipeline parameter: {}. Only pre-defined parameters can be modified."
