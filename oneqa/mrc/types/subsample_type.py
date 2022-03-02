from enum import IntEnum, unique


@unique
class SubsampleType(IntEnum):
    POSITIVE = 0
    NEGATIVE_HAS_ANSWER = 1
    NEGATIVE_NO_ANSWER = 2