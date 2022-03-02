from enum import IntEnum, unique


@unique
class TargetType(IntEnum):
    NO_ANSWER = 0
    SPAN_ANSWER = 1
    PASSAGE_ANSWER = 2
    YES = 3
    NO = 4

    @classmethod
    def from_bool_label(cls, label: str) -> 'TargetType':
        label = label.upper()
        if label == 'NONE':
            label = 'NO_ANSWER'
        return cls[label]

    # def __int__(self):
    #     return self.value