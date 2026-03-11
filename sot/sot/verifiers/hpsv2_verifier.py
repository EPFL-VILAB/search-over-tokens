"""HPSv2 verifier for human preference scoring.

HPSv2 (Human Preference Score v2) predicts how well an image matches
human aesthetic and alignment preferences.

Model: https://huggingface.co/RE-N-Y/hpsv21
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Union
from PIL import Image

from sot.verifiers.base import BaseVerifier, VerifierFactory

logger = logging.getLogger("sot.hpsv2")

try:
    from imscore.hps.model import HPSv2
    from einops import rearrange
    HPSV2_AVAILABLE = True
except ImportError:
    HPSV2_AVAILABLE = False

__all__ = ["HPSv2Verifier"]


@VerifierFactory.register("hpsv2")
class HPSv2Verifier(BaseVerifier):
    """HPSv2-based human preference scorer.

    Uses HPSv2 to score how well images match human preferences for
    text-to-image alignment and aesthetic quality.

    Config parameters:
        - model_name: HuggingFace model name (default: ``"RE-N-Y/hpsv21"``)
    """

    def __init__(self, config: Dict, device: str = "cuda"):
        if not HPSV2_AVAILABLE:
            raise ImportError(
                "HPSv2 not installed. Install imscore and einops:\n"
                "  pip install imscore einops"
            )

        super().__init__(config, device)

        model_name = config.get("model_name", "RE-N-Y/hpsv21")
        self.model = HPSv2.from_pretrained(model_name).to(device)
        logger.info(f"Loaded HPSv2 model ({model_name})")

    def _score(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: List[str],
        **kwargs,
    ) -> torch.Tensor:
        """Score images using HPSv2.

        Args:
            images: List of PIL Images.
            prompts: List of text prompts.
            **kwargs: Additional parameters.

        Returns:
            Scores tensor of shape ``[B]``.
        """
        text_prompt = prompts[0] if isinstance(prompts, list) else prompts
        all_rewards: List[float] = []

        for img in images:
            pixels_np = np.array(img)
            pixels_tensor = torch.tensor(pixels_np, dtype=torch.float32) / 255.0
            pixels_tensor = pixels_tensor.to(self.device)
            pixels_tensor = rearrange(pixels_tensor, "h w c -> 1 c h w")

            reward = self.model.score(pixels_tensor, text_prompt).squeeze(-1)
            all_rewards.append(reward.detach().cpu().item())

        return torch.tensor(all_rewards, dtype=torch.float32)


