from enum import IntEnum, unique


@unique
class SubsampleType(IntEnum):
    """
    Enumeration type representing whether a feature is positive or negative (with/without an answer)
    for use in subsampling.
    """
    POSITIVE = 0
    NEGATIVE_HAS_ANSWER = 1
    NEGATIVE_NO_ANSWER = 2
