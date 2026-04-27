"""PickScore verifier for text-image alignment.

PickScore is a scoring function trained on the Pick-a-Pic dataset of
human preferences for text-to-image generation.

Paper: https://arxiv.org/abs/2305.01569
Model: https://huggingface.co/yuvalkirstain/PickScore_v1
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Union
from PIL import Image

from soto.verifiers.base import BaseVerifier, VerifierFactory

try:
    from transformers import AutoProcessor, AutoModel
    PICKSCORE_AVAILABLE = True
except (ImportError, Exception):
    PICKSCORE_AVAILABLE = False

logger = logging.getLogger("soto.pickscore")

__all__ = ["PickScoreVerifier"]


@VerifierFactory.register("pickscore")
class PickScoreVerifier(BaseVerifier):
    """PickScore-based image-text alignment scorer.

    Uses the PickScore model (fine-tuned CLIP ViT-H/14) to score how well
    images match text prompts based on human preferences.

    Config parameters:
        - batch_size: Batch size for scoring (default: 32)
    """

    def __init__(self, config: Dict, device: str = "cuda"):
        if not PICKSCORE_AVAILABLE:
            raise ImportError(
                "PickScore dependencies not available. Either transformers is "
                "not installed or is incompatible with your torch version.\n"
                "  pip install transformers"
            )

        super().__init__(config, device)

        processor_name = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
        model_name = "yuvalkirstain/PickScore_v1"

        self.processor = AutoProcessor.from_pretrained(processor_name)
        self.model = AutoModel.from_pretrained(model_name).eval().to(device)

        logger.info("Loaded PickScore model (PickScore_v1)")

    def _score(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: List[str],
        **kwargs,
    ) -> torch.Tensor:
        """Score images using PickScore.

        Args:
            images: List of PIL Images.
            prompts: List of text prompts (one per image, or a single prompt
                broadcast to all images).
            **kwargs: Additional parameters.

        Returns:
            Scores tensor of shape ``[B]``.
        """
        batch_size = self.config.get("batch_size", 32)

        text_prompt = prompts[0] if isinstance(prompts, list) else prompts

        all_scores: List[float] = []

        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]

            image_inputs = self.processor(
                images=batch_images,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            text_inputs = self.processor(
                text=text_prompt,
                padding=True,
                truncation=True,
                max_length=77,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                image_embs = self.model.get_image_features(**image_inputs)
                image_embs = image_embs / image_embs.norm(dim=-1, keepdim=True)

                text_embs = self.model.get_text_features(**text_inputs)
                text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

                # Cosine similarity scaled by learned temperature
                scores = self.model.logit_scale.exp() * (text_embs @ image_embs.T)[0]

            all_scores.extend(scores.cpu().tolist())

        return torch.tensor(all_scores, dtype=torch.float32)


