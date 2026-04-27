"""DreamSim verifier for image-to-image perceptual similarity.

DreamSim measures perceptual similarity between images, aligning well with
human perception. Unlike text-based verifiers (CLIP, ImageReward), DreamSim
compares a generated image against a **reference image**.

This is useful for subject-driven generation benchmarks like DreamBench++
where the goal is to preserve the identity/appearance of a reference subject.

Paper: https://arxiv.org/abs/2306.09344
Repo: https://github.com/ssundaram21/dreamsim
"""

import torch
import logging
from typing import Dict, List, Union
from PIL import Image

from soto.verifiers.base import BaseVerifier, VerifierFactory

logger = logging.getLogger("soto.dreamsim")

try:
    from dreamsim import dreamsim
    DREAMSIM_AVAILABLE = True
except ImportError:
    DREAMSIM_AVAILABLE = False

__all__ = ["DreamSimVerifier"]


@VerifierFactory.register("dreamsim")
class DreamSimVerifier(BaseVerifier):
    """DreamSim-based perceptual similarity scorer.

    Computes perceptual similarity between generated images and reference
    images using the DreamSim model.  Higher scores indicate greater
    similarity (score = 1 − dreamsim_distance).

    Unlike text-based verifiers, DreamSim requires a **reference image** for
    each generated image.  The reference image path is passed via the
    ``metadata`` keyword argument in :meth:`score`.

    Config parameters:
        - batch_size: Batch size for scoring (default: 32)

    Keyword arguments accepted by :meth:`score`:
        - metadata: Reference image path (``str``) *or* a list of paths,
          one per generated image.  When a single path is provided it is
          broadcast to all images.
        - ref_image_path: Alternative way to pass the reference image
          path (takes precedence over ``metadata``).
    """

    def __init__(self, config: Dict, device: str = "cuda"):
        if not DREAMSIM_AVAILABLE:
            raise ImportError(
                "dreamsim not installed. Install it with:\n"
                "  pip install dreamsim\n"
                "Or install soto with the dreamsim extra:\n"
                "  pip install -e '.[dreamsim]'"
            )

        super().__init__(config, device)

        import os
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "dreamsim")
        self.model, self.preprocess = dreamsim(pretrained=True, device=device, cache_dir=cache_dir)
        self._ref_cache: dict = {}  # cache preprocessed reference tensors across score() calls
        logger.info("Loaded DreamSim model (pretrained)")

    def _score(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: List[str],
        **kwargs,
    ) -> torch.Tensor:
        """Score images by perceptual similarity to a reference image.

        Args:
            images: List of PIL Images (generated images to score).
            prompts: List of text prompts (unused by DreamSim but kept for
                interface compatibility).
            **kwargs:
                metadata (str | List[str]): Path(s) to reference image(s).
                    A single path is broadcast to all images.
                ref_image_path (str): Alternative to ``metadata``.

        Returns:
            Similarity scores tensor of shape ``[B]`` where higher is better.
        """
        # Resolve reference image path(s)
        ref_path = kwargs.get("ref_image_path", None) or kwargs.get("metadata", None)
        if ref_path is None:
            raise ValueError(
                "DreamSimVerifier requires a reference image. Pass it as "
                "'metadata' (e.g. the image_path from DreamBench) or "
                "'ref_image_path' in kwargs."
            )

        all_scores: List[float] = []

        with torch.no_grad():
            for idx, img in enumerate(images):
                # Determine reference path for this image
                if isinstance(ref_path, list):
                    rp = ref_path[idx]
                else:
                    rp = ref_path

                if rp not in self._ref_cache:
                    self._ref_cache[rp] = self.preprocess(
                        Image.open(rp).convert("RGB")
                    ).to(self.device)
                ref_tensor = self._ref_cache[rp]

                gen_tensor = self.preprocess(
                    img.convert("RGB") if isinstance(img, Image.Image) else img
                ).to(self.device)

                distance = self.model(ref_tensor, gen_tensor)
                # Convert distance to similarity (higher is better)
                similarity = 1.0 - distance.item()
                all_scores.append(similarity)

        return torch.tensor(all_scores, dtype=torch.float32)

