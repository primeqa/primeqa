import torch
from transformers import AutoConfig, AutoModel

from tests.oneqa.mrc.common.parameterization import PARAMETERIZE_FIXTURE_WITH_MODEL_NAME


class UnitTest:
    """
    Base class for all unit test classes
    """

    @PARAMETERIZE_FIXTURE_WITH_MODEL_NAME
    def model_name_and_config(self, request):
        model_name = request.param
        _ = AutoModel.from_pretrained(model_name)  # Pre-download LM inside flaky fixture so other tests have it
        return model_name, AutoConfig.from_pretrained(model_name)

    @staticmethod
    def _assert_is_floating_point_tensor(tensor):
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype.is_floating_point
