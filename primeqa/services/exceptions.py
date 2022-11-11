import re
from enum import Enum

PATTERN_ERROR_MESSAGE = re.compile("(E[0-9]{4,5}):(.*)")


class Error(Exception):
    pass


class ErrorMessages(str, Enum):
    # REQUEST
    INVALID_REQUEST = "E1001: Missing mandatory field: {} from request"

    # PARAMETER
    INVALID_PARAMETER_DEFINITION = (
        "E3001: Invalid {} parameter definition. Please contact us."
    )
    INVALID_PARAMETER = (
        "E3002: Invalid {} parameter: {}. Only pre-defined parameters can be modified."
    )

    # READER
    INVALID_READER = "E4001: Invalid reader: {}. Please select one of the following pre-defined readers: {}"
    MISSING_CONTEXT = "E4005: If contexts are provided, number of contexts({}) must match number of queries({})"

    # RETRIEVER
    INVALID_RETRIEVER = "E5001: Invalid retriever: {}. Please select one of the following pre-defined retrievers: {}"
    INDEX_UNAVAILABLE_FOR_QUERYING = 'E5002: Cannot query index with "{}" status. Please make sure index has "READY" status before querying.'

    # INDEXER
    INVALID_INDEXER = "E6001: Invalid indexer: {}. Please select one of the following pre-defined indexers: {}"
    FAILED_TO_LOCATE_INDEX = "E6002: Index with id {} doesn't exist."

    # INITIALIZATION
    FAILED_TO_INITIALIZE = "E9001: Failed to initalize {}. Please contact us."
