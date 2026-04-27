"""ImageReward verifier for text-image alignment."""

import io
import contextlib
import torch
import numpy as np
import logging
from typing import List, Union, Dict
from PIL import Image

from soto.verifiers.base import BaseVerifier, VerifierFactory

logger = logging.getLogger("soto.imagereward")

# Suppress FutureWarning from timm >=1.0 about deprecated import paths used by image-reward
import warnings
warnings.filterwarnings("ignore", message=".*Importing from timm.models.*deprecated.*", category=FutureWarning)

try:
    import ImageReward as RM
    IMAGEREWARD_AVAILABLE = True
    IMAGEREWARD_ERROR = None
except ImportError as e:
    IMAGEREWARD_AVAILABLE = False
    IMAGEREWARD_ERROR = str(e)

__all__ = ["ImageRewardVerifier"]


@VerifierFactory.register("image_reward")
class ImageRewardVerifier(BaseVerifier):
    """ImageReward-based image-text alignment scorer.
    
    Uses the ImageReward model to score how well images match text prompts.
    Higher scores indicate better alignment.
    
    Config parameters:
        - batch_size: Batch size for scoring (default: 32)
    """

    def __init__(self, config: Dict, device: str = "cuda"):
        if not IMAGEREWARD_AVAILABLE:
            error_msg = (
                "ImageReward could not be imported.\n"
                "Install with:\n"
                "  pip install '.[imagereward]' && pip install image-reward --no-deps\n"
                "  (--no-deps avoids image-reward's pinned timm==0.6.13 which conflicts with l3m)\n"
            )
            if IMAGEREWARD_ERROR:
                error_msg += f"\nOriginal error: {IMAGEREWARD_ERROR}"
            raise ImportError(error_msg)

        super().__init__(config, device)

        with contextlib.redirect_stdout(io.StringIO()):   # suppress "load checkpoint from..." prints
            self.model = RM.load("ImageReward-v1.0", device=device)
        logger.info("Loaded ImageReward model (ImageReward-v1.0)")
    
    def _score(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: List[str],
        **kwargs
    ) -> torch.Tensor:
        """Score images using ImageReward.

        Args:
            images: List of PIL Images
            prompts: List of text prompts (one per image)
            **kwargs: Additional parameters

        Returns:
            Scores tensor of shape [B]
        """
        batch_size = self.config.get("batch_size", 32)
        all_rewards = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            # ImageReward expects a single prompt string for a batch
            # All images in this batch should have the same prompt
            batch_prompt = prompts[i] if i < len(prompts) else prompts[0]
            
            # Score the batch
            rewards = self.model.score(batch_prompt, batch_images)
            
            # Handle single vs batch return
            if isinstance(rewards, (list, np.ndarray)):
                all_rewards.extend(rewards)
            else:
                all_rewards.append(rewards)
        
        return torch.tensor(all_rewards, dtype=torch.float32)

