"""
Test fixtures and parameterization
"""
from functools import wraps
from typing import Callable

import pytest

_MODEL_NAMES = [
        'roberta-base', 'xlm-roberta-base', 'bert-base-uncased', 'albert-base-v2',
]


# Cannot be rewritten as constant parametrize(fixture)
# noinspection PyPep8Naming
def parameterize_fixture_with_model_name(f: Callable) -> Callable:
    @pytest.mark.flaky(reruns=10, reruns_delay=30)  # Account for intermittent S3 errors downloading HF data
    @pytest.fixture(scope='session', params=_MODEL_NAMES)
    @wraps(f)
    def inner(*args, **kwargs):
        return f(*args, **kwargs)
    return inner


_INVALID_PROBABILITIES = [-0.01, 1.01]
PARAMETERIZE_INVALID_SUBSAMPLING_PROBABILITIES = pytest.mark.parametrize(
    ["negative_sampling_prob_when_has_answer", "negative_sampling_prob_when_no_answer"],
    [(p1, p2) for p1 in _INVALID_PROBABILITIES for p2 in _INVALID_PROBABILITIES])
