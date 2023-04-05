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
    INVALID_READER_INPUT = "E4006: Mismatched number of questions and contexts."

    # RETRIEVER
    INVALID_RETRIEVER = "E5001: Invalid retriever: {}. Please select one of the following pre-defined retrievers: {}"
    INDEX_UNAVAILABLE_FOR_QUERYING = 'E5002: Cannot query index with "{}" status. Please make sure index has "READY" status before querying.'
    MISMATCHED_ENGINE_TYPE = 'E5003: Cannot query index with "{}" engine_type with {} retriever of "{}" engine type.'

    # INDEXER
    INVALID_INDEXER = "E6001: Invalid indexer: {}. Please select one of the following pre-defined indexers: {}"
    FAILED_TO_LOCATE_INDEX = "E6002: Index with id {} doesn't exist."
    FAILED_TO_LOCATE_INDEX_INFORMATION = (
        "E6003: Index information for index with id {} doesn't exist."
    )
    
    # RETRANKER
    INVALID_RETRANKER = "E5001: Invalid reranker: {}. Please select one of the following pre-defined rerankers: {}"

    # INITIALIZATION
    FAILED_TO_INITIALIZE = "E9001: Failed to initialize {}. Please contact us."
