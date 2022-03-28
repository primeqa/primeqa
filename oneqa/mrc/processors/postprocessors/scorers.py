"""
BEGIN_COPYRIGHT

IBM Confidential
OCO Source Materials

5727-I17
(C) Copyright IBM Corp. 2019 All Rights Reserved.
 
The source code for this program is not published or otherwise
divested of its trade secrets, irrespective of what has been
deposited with the U.S. Copyright Office.

END_COPYRIGHT
"""
import logging
from functools import partial

from typing import List, Optional, Callable, Union
from enum import Enum


from oneqa.mrc.data_models.target_type import TargetType


class SupportedSpanScorers(Enum):
    SCORE_DIFF_BASED = 'score_diff_based'
    TARGET_TYPE_WEIGHTED_SCORE_DIFF = 'target_type_weighted_score_diff'
    WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF = 'weighted_sum_target_type_and_score_diff'

    @classmethod
    def get_supported(cls):
        return [entry.name for entry in cls]


def initialize_scorer(scorer_type: Union[str, SupportedSpanScorers], target_type_weight: Optional[float]=0.5) -> Callable:
    if not isinstance(scorer_type, SupportedSpanScorers):
        scorer_type = SupportedSpanScorers(scorer_type)
    if scorer_type == SupportedSpanScorers.SCORE_DIFF_BASED:
        logging.debug("\tInitialized scorer %s" % compute_score_diff_between_span_and_cls.__name__)
        return compute_score_diff_between_span_and_cls

    elif scorer_type == SupportedSpanScorers.TARGET_TYPE_WEIGHTED_SCORE_DIFF:
        logging.debug("\tInitialized scorer %s" %
                      compute_short_answer_type_weighted_score_diff_between_span_and_cls.__name__)
        return compute_short_answer_type_weighted_score_diff_between_span_and_cls

    elif scorer_type == SupportedSpanScorers.WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF:
        logging.debug(
            "\tInitialized scorer %s with weight %s for the target type score and weight %s for "
            "the score diff score" % (
                compute_weighted_sum_short_answer_type_score_diff_between_span_and_cls.__name__,
                target_type_weight, 1 - target_type_weight))

        return partial(
            compute_weighted_sum_short_answer_type_score_diff_between_span_and_cls,
            target_type_weight=target_type_weight)
    else:
        raise ValueError('Unsupported scorer type: %s' % scorer_type)


def compute_score_diff_between_span_and_cls(
        span_score: float, null_span_score: float, *args, **kwargs):
    return span_score - null_span_score


def compute_short_answer_type_weighted_score_diff_between_span_and_cls(
        span_score: float, null_span_score: float,
        target_type_logits: List, *args, **kwargs):
    score_diff = compute_score_diff_between_span_and_cls(
        span_score=span_score, null_span_score=null_span_score)
    return score_diff * target_type_logits[int(TargetType.SPAN_ANSWER)]


def compute_weighted_sum_short_answer_type_score_diff_between_span_and_cls(
        span_score: float, null_span_score: float,
        target_type_logits: List, target_type_weight: float, *args, **kwargs):
    score_diff = compute_score_diff_between_span_and_cls(
        span_score=span_score, null_span_score=null_span_score)

    return (1 - target_type_weight) * score_diff + target_type_weight * target_type_logits[
        int(TargetType.SPAN_ANSWER)]
