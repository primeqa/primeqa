import pytest

# Parameterize model names in test fixtures
_MODEL_NAMES = ("model_name", [
        'roberta-base', 'xlm-roberta-base', 'bert-base-uncased', 'albert-base-v2',
        # 'facebook/bart-base'  # TODO: add bart support
])
PARAMETERIZE_TEST_WITH_MODEL_NAME = pytest.mark.parametrize(*_MODEL_NAMES)
PARAMETERIZE_FIXTURE_WITH_MODEL_NAME = pytest.fixture(scope='package', params=_MODEL_NAMES[1])

_INVALID_PROBABILITIES = [-0.01, 1.01]
PARAMETERIZE_INVALID_SUBSAMPLING_PROBABILITIES = pytest.mark.parametrize(
    ["negative_sampling_prob_when_has_answer", "negative_sampling_prob_when_no_answer"],
    [(p1, p2) for p1 in _INVALID_PROBABILITIES for p2 in _INVALID_PROBABILITIES])
