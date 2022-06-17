import logging
import torch
from primeqa.util.transformers_utils.hypers_base import HypersBase

logger = logging.getLogger(__name__)


def to_tensor(hypers: HypersBase, tensor):
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, device=hypers.device)
    else:
        tensor = tensor.to(hypers.device).detach()
    return tensor


def all_gather(tensor):
    """
    all gather the tensor with dimensions [d0 x d1 x...], returning a tensor with dimensions [d0*world_size x d1 x...]
    :param tensor: this process's tensor
    :return:
    """
    tensor = tensor.detach()
    gather_list = [torch.zeros_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gather_list, tensor)
    return torch.cat(gather_list, 0).detach()


def reduce(hypers: HypersBase, tensor, *, op=torch.distributed.ReduceOp.SUM):
    """
    all reduce the tensor, modifying the tensor
    :param tensor: the tensor that will be all-reduced
    :param op: operation to reduce (example: torch.distributed.ReduceOp.SUM)
    :param check_id: identifier for this call to all_reduce (to check that there is no cross talk)
    :return:
    """
    tensor = to_tensor(hypers, tensor)
    if hypers.world_size == 1:
        return tensor
    torch.distributed.all_reduce(tensor, op)
    return tensor

