import logging
from functools import partial

from typing import List, Optional, Callable, Union
from enum import Enum

from primeqa.mrc.data_models.target_type import TargetType


class SupportedSpanScorers(Enum):
    """
    Enumeration of supported scoring algorithms.
    """
    SCORE_DIFF_BASED = 'score_diff_based'
    TARGET_TYPE_WEIGHTED_SCORE_DIFF = 'target_type_weighted_score_diff'
    WEIGHTED_SUM_TARGET_TYPE_AND_SCORE_DIFF = 'weighted_sum_target_type_and_score_diff'

    @classmethod
    def get_supported(cls):
        """
        Returns the names of the supported scoring algorithms.
        """
        return [entry.value for entry in cls]


def initialize_scorer(scorer_type: Union[str, SupportedSpanScorers],
                      target_type_weight: Optional[float] = 0.5) -> Callable:
    """
    Factory method to initialize scorer.

    Args:
        scorer_type: Which scoring algorithm to use.
        target_type_weight: How much weight [0-1] to put on target type logits vs start/end logits.

    Returns:
        Initialized scorer.
    """
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
    """
    Compute score as difference between span score (e.g. `start=i`, `end=j`) and null span score (e.g. CLS).
    """
    return span_score - null_span_score


def compute_short_answer_type_weighted_score_diff_between_span_and_cls(
        span_score: float, null_span_score: float,
        target_type_logits: List, *args, **kwargs):
    """
    Compute score as product of target type logit (for span answer) and `compute_score_diff_between_span_and_cls`
    """
    score_diff = compute_score_diff_between_span_and_cls(
        span_score=span_score, null_span_score=null_span_score)
    return score_diff * target_type_logits[int(TargetType.SPAN_ANSWER)]


def compute_weighted_sum_short_answer_type_score_diff_between_span_and_cls(
        span_score: float, null_span_score: float,
        target_type_logits: List, target_type_weight: float, *args, **kwargs):
    """
    Compute score as sum of products `target_type_weight` times target type logit (for span answer) and
    `1 - target_type_weight` times `compute_score_diff_between_span_and_cls`.
    """
    score_diff = compute_score_diff_between_span_and_cls(
        span_score=span_score, null_span_score=null_span_score)

    return (1 - target_type_weight) * score_diff + target_type_weight * target_type_logits[
        int(TargetType.SPAN_ANSWER)]
