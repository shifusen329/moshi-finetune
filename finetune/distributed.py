import logging
import os
from functools import lru_cache
from typing import List, Union

import torch
import torch.distributed as dist

logger = logging.getLogger("distributed")

BACKEND = "nccl"


@lru_cache()
def get_rank() -> int:
    return dist.get_rank()


@lru_cache()
def get_world_size() -> int:
    return dist.get_world_size()


def visible_devices() -> List[int]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_visible is None or cuda_visible == "":
        # Not set means all GPUs are visible
        return list(range(torch.cuda.device_count()))
    return [int(d) for d in cuda_visible.split(",")]


def set_device():
    logger.info(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', '<not set>')}")
    logger.info(f"local rank: {int(os.environ['LOCAL_RANK'])}")

    assert torch.cuda.is_available()

    # Only check device count match if CUDA_VISIBLE_DEVICES is explicitly set
    if "CUDA_VISIBLE_DEVICES" in os.environ:
        assert len(visible_devices()) == torch.cuda.device_count()

    if torch.cuda.device_count() == 1:
        # gpus-per-task set to 1
        torch.cuda.set_device(0)
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    logger.info(f"Set cuda device to {local_rank}")

    assert 0 <= local_rank < torch.cuda.device_count(), (
        local_rank,
        torch.cuda.device_count(),
    )
    torch.cuda.set_device(local_rank)


def avg_aggregate(metric: Union[float, int]) -> Union[float, int]:
    buffer = torch.tensor([metric], dtype=torch.float32, device="cuda")
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    return buffer[0].item() / get_world_size()


def is_torchrun() -> bool:
    return "TORCHELASTIC_RESTART_COUNT" in os.environ
