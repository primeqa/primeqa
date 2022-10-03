from enum import Enum


class IndexStatus(str, Enum):
    READY = "READY"
    INDEXING = "INDEXING"
    DOES_NOT_EXISTS = "DOES_NOT_EXISTS"
    CORRUPT = "CORRUPT"
