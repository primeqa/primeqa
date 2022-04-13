from enum import IntEnum, unique


@unique
class TargetType(IntEnum):
    """
    Enumeration representing different types of answers to be used as a target in training.
    """
    NO_ANSWER = 0
    SPAN_ANSWER = 1
    PASSAGE_ANSWER = 2
    YES = 3
    NO = 4

    @classmethod
    def from_bool_label(cls, label: str) -> 'TargetType':
        """
        Alternate constructor from a boolean label string.

        Args:
            label (`str`): yes|no|none

        Returns:
            ([`oneqa.mrc.data_models.target_type.TargetType`]): target type corresponding to label
        """
        label = label.upper()
        if label == 'NONE':
            label = 'NO_ANSWER'
        return cls[label]
