import torch


class UnitTest:
    """
    Base class for all unit test classes
    """

    @staticmethod
    def _assert_is_floating_point_tensor(tensor):
        assert isinstance(tensor, torch.Tensor)
        assert tensor.dtype.is_floating_point
