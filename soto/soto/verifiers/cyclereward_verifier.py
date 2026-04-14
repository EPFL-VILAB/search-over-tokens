"""CycleReward verifier for text-image alignment.

CycleReward uses a cycle-consistent reward model to score how well images
match text prompts.

Model: https://huggingface.co/NagaSaiAbhinay/CycleReward-Combo
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Union
from PIL import Image

from soto.verifiers.base import BaseVerifier, VerifierFactory

logger = logging.getLogger("soto.cyclereward")

try:
    from imscore.cyclereward.model import CycleReward
    from einops import rearrange
    CYCLEREWARD_AVAILABLE = True
except ImportError:
    CYCLEREWARD_AVAILABLE = False

__all__ = ["CycleRewardVerifier"]


@VerifierFactory.register("cyclereward")
class CycleRewardVerifier(BaseVerifier):
    """CycleReward-based image-text alignment scorer.

    Uses CycleReward-Combo to score how well images match text prompts.
    Higher scores indicate better alignment.

    Config parameters:
        - model_name: HuggingFace model name
          (default: ``"NagaSaiAbhinay/CycleReward-Combo"``)
        - batch_size: Batch size for scoring (default: 32)
    """

    def __init__(self, config: Dict, device: str = "cuda"):
        if not CYCLEREWARD_AVAILABLE:
            raise ImportError(
                "CycleReward not installed. Install imscore and einops:\n"
                "  pip install imscore einops"
            )

        super().__init__(config, device)

        model_name = config.get("model_name", "NagaSaiAbhinay/CycleReward-Combo")
        self.model = CycleReward.from_pretrained(model_name).to(device)
        logger.info(f"Loaded CycleReward model ({model_name})")

    def _score(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: List[str],
        **kwargs,
    ) -> torch.Tensor:
        """Score images using CycleReward.

        Args:
            images: List of PIL Images.
            prompts: List of text prompts.
            **kwargs: Additional parameters.

        Returns:
            Scores tensor of shape ``[B]``.
        """
        batch_size = self.config.get("batch_size", 32)
        text_prompt = prompts[0] if isinstance(prompts, list) else prompts

        all_rewards: List[float] = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]

            pixels_np = np.stack([np.array(img) for img in batch_images])
            pixels_tensor = torch.tensor(pixels_np, dtype=torch.float32) / 255.0
            pixels_tensor = pixels_tensor.to(self.device)
            pixels_tensor = rearrange(pixels_tensor, "b h w c -> b c h w")

            rewards = self.model.score(pixels_tensor, text_prompt).squeeze(-1)
            all_rewards.extend(rewards.cpu().detach().tolist())

        return torch.tensor(all_rewards, dtype=torch.float32)


