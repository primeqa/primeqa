"""
Test fixtures and parameterization
"""
from functools import wraps
from typing import Callable

import pytest

# Parameterize model names in test fixtures
_MODEL_NAMES = ("model_name", [
        'roberta-base', 'xlm-roberta-base', 'bert-base-uncased', 'albert-base-v2',
        # 'facebook/bart-base'  # TODO: add bart support
])
# _PARAMETERIZE_TEST_WITH_MODEL_NAME = pytest.mark.parametrize(*_MODEL_NAMES)  # TODO: remove
# _PARAMETERIZE_FIXTURE_WITH_MODEL_NAME = pytest.fixture(scope='package', params=_MODEL_NAMES[1])  # TODO: remove


# noinspection PyPep8Naming
def PARAMETERIZE_TEST_WITH_MODEL_NAME(f: Callable) -> Callable:  # Cannot be rewritten as constant parametrize(fixture)
    @pytest.mark.flaky(reruns=5, reruns_delay=2)  # Account for intermittent S3 errors downloading HF models
    @pytest.mark.parametrize(*_MODEL_NAMES)
    @wraps(f)
    def inner(*args, **kwargs):
        return f(*args, **kwargs)

    return inner


# noinspection PyPep8Naming
def PARAMETERIZE_FIXTURE_WITH_MODEL_NAME(f: Callable) -> Callable:
    @pytest.mark.flaky(reruns=5, reruns_delay=2)
    @pytest.fixture(scope='package', params=_MODEL_NAMES[1])
    @wraps(f)
    def inner(*args, **kwargs):
        return f(*args, **kwargs)
    return inner


_INVALID_PROBABILITIES = [-0.01, 1.01]
PARAMETERIZE_INVALID_SUBSAMPLING_PROBABILITIES = pytest.mark.parametrize(
    ["negative_sampling_prob_when_has_answer", "negative_sampling_prob_when_no_answer"],
    [(p1, p2) for p1 in _INVALID_PROBABILITIES for p2 in _INVALID_PROBABILITIES])
