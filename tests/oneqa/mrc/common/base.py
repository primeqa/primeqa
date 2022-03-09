import torch
from transformers import AutoConfig

from tests.oneqa.mrc.common.parameterization import PARAMETERIZE_FIXTURE_WITH_MODEL_NAME


class UnitTest:
    """
    Base class for all unit test classes
    """

    @PARAMETERIZE_FIXTURE_WITH_MODEL_NAME
    def model_name_and_config(self, request):
        model_name = request.param
        return model_name, AutoConfig.from_pretrained(model_name)

    @staticmethod
    def _assert_is_floating_point_tensor(tensor):
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype.is_floating_point
