import pytest
from pytest import raises
from transformers import PretrainedConfig

from oneqa.mrc.models.heads.abstract import AbstractTaskHead
from tests.oneqa.mrc.common.base import UnitTest


class TestAbstractTaskHead(UnitTest):
    @pytest.fixture(scope='function')
    def mocked_config(self, mocker):
        return mocker.Mock(PretrainedConfig)

    def test_cannot_instantiate_abstract_class(self, mocked_config):
        with raises(TypeError):
            _ = AbstractTaskHead(mocked_config)
