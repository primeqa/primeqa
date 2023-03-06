from enum import Enum

ATTR_INDEX_ID = "index_id"
ATTR_STATUS = "status"
ATTR_METADATA = "metadata"
ATTR_CONFIGURATION = "configuration"
ATTR_ENGINE_TYPE = "engine_type"
ATTR_CHECKPOINT = "checkpoint"


class IndexStatus(str, Enum):
    READY = "READY"
    INDEXING = "INDEXING"
    DOES_NOT_EXISTS = "DOES_NOT_EXISTS"
    CORRUPT = "CORRUPT"
