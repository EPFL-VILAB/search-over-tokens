"""Common utilities for SoT."""

import logging
import random
import json
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.nn.init import trunc_normal_

__all__ = [
    "set_seed", "save_checkpoint", "load_checkpoint", "is_caption_complete", "setup_logging",
    "trunc_normal_",  # Re-export for DreamSim/DINO compatibility (their code does "from utils import trunc_normal_")
]


def set_seed(seed: int):
    """Set random seeds for reproducibility.
    
    Different ranks get different seeds to ensure diversity while
    maintaining reproducibility.
    
    Args:
        seed: Base seed
        rank: Process rank (for different seeds per rank)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(
    output_dir: Path,
    step: int,
    selected_tokens: List[List[int]],
    scores: List[float]
):
    """Save checkpoint for resume capability.
    
    Allows interrupting and resuming beam search from the last completed step.
    
    Args:
        output_dir: Directory to save checkpoint
        step: Current step number (0-indexed)
        selected_tokens: List of token sequences (beam candidates)
        scores: Scores for each sequence
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "step": step,
        "tokens": selected_tokens,
        "scores": scores
    }
    
    ckpt_file = output_dir / "checkpoint.json"
    with open(ckpt_file, 'w') as f:
        json.dump(checkpoint, f, indent=2)


def load_checkpoint(output_dir: Path) -> Tuple[int, List[List[int]], Optional[List[float]]]:
    """Load checkpoint if exists.
    
    Args:
        output_dir: Directory containing checkpoint
        
    Returns:
        (step, tokens, scores): Checkpoint data or (0, [[]], None) if not found
    """
    ckpt_file = Path(output_dir) / "checkpoint.json"
    
    if ckpt_file.exists():
        with open(ckpt_file, 'r') as f:
            ckpt = json.load(f)
        return ckpt["step"], ckpt["tokens"], ckpt.get("scores")
    
    # No checkpoint found - start from beginning
    return 0, [[]], None


def is_caption_complete(output_dir: Path) -> bool:
    """Check if a caption has been fully processed.
    
    A caption is considered complete if it has result images saved.
    
    Args:
        output_dir: Caption output directory
        
    Returns:
        True if complete (has result images), False otherwise
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return False
    
    # Check for result images
    result_images = list(output_dir.glob("result_*.png"))
    return len(result_images) > 0


def setup_logging(rank: int = 0):
    """Setup Python logging to write to both console and Hydra log file.
    
    Only sets up logging on rank 0 to avoid conflicts in multi-GPU runs.
    Other ranks will have logging disabled (WARNING level).
    
    Args:
        rank: Process rank (only rank 0 sets up logging)
    
    Returns:
        Logger instance configured for the current rank
    """
    from hydra.core.hydra_config import HydraConfig

    logger = logging.getLogger("sot")
    
    def _add_handler(h: logging.Handler) -> None:
        h.setLevel(logging.INFO)
        h.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(h)

    # Only configure on rank 0
    if rank == 0:
        logger.setLevel(logging.INFO)
        logger.handlers.clear()  # avoid duplicate handlers on re-init

        _add_handler(logging.StreamHandler(sys.stdout))

        hydra = HydraConfig.get()
        log_file = Path(hydra.runtime.output_dir) / f"{hydra.job.name}.log"
        _add_handler(logging.FileHandler(log_file, mode="a"))
    else:
        logger.setLevel(logging.WARNING)  # silence non-rank-0 workers
    
    return logger

