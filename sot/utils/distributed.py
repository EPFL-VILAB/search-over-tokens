"""Minimal distributed utilities for dataset-parallel execution.

For dataset parallelization (each GPU processes different captions),
no synchronization is needed - GPUs work completely independently!
"""

import os
import torch
import torch.distributed as dist
from typing import Tuple

__all__ = ["is_distributed", "get_rank_info"]


def is_distributed() -> bool:
    """Check if running in distributed mode (torchrun).
    
    Returns:
        True if running with torchrun, False otherwise
    """
    # Two ways we can detect distributed execution:
    #   1) A torch.distributed process group has been initialized
    #   2) torchrun-style environment variables are present (RANK/WORLD_SIZE)
    #
    # For this project we only need *dataset* parallelism and never call any
    # collectives, so it's safe (and convenient) to treat the torchrun env vars
    # as "distributed" even if dist.init_process_group() was never called.
    if dist.is_available() and dist.is_initialized():
        return True
    
    # Fallback: detect torchrun via environment variables
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return world_size > 1


def get_rank_info() -> Tuple[int, int, int]:
    """Get rank, local_rank, world_size.
    
    Works for both distributed (torchrun) and single GPU modes.
    
    Returns:
        (rank, local_rank, world_size): Tuple of ints
            - rank: Global rank across all nodes
            - local_rank: Rank on this node (for GPU selection)
            - world_size: Total number of processes
    
    Example:
        >>> rank, local_rank, world_size = get_rank_info()
        >>> device = f"cuda:{local_rank}"
        >>> # Partition dataset
        >>> my_captions = all_captions[rank::world_size]
    """
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size()
    else:
        # Either single-GPU mode or "lightweight" distributed mode where we are
        # launched with torchrun but never initialize a process group.
        #
        # torchrun always sets the following env vars:
        #   - RANK:       global rank
        #   - LOCAL_RANK: rank on this node (for GPU selection)
        #   - WORLD_SIZE: total number of processes
        #
        # When WORLD_SIZE>1 we still want different (rank, local_rank)
        # tuples on each process so that:
        #   - each process picks a different CUDA device
        #   - datasets can be sharded by rank
        rank = int(os.environ.get("RANK", "0"))
        # Default LOCAL_RANK to RANK if not set so that single-process
        # execution (without torchrun) continues to use GPU 0.
        local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
    
    return rank, local_rank, world_size

