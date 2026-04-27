"""Aesthetic score verifier using CLIP + MLP aesthetic predictor.

Predicts how aesthetically pleasing an image is, regardless of text prompt.
Uses a CLIP ViT-L/14 backbone with a trained MLP head.

The MLP weights are **automatically downloaded from the internet** when
``checkpoint_path`` is not specified (or points to a missing file).

Source: https://github.com/christophschuhmann/improved-aesthetic-predictor
"""

import os
import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Union
from PIL import Image

import clip

from soto.verifiers.base import BaseVerifier, VerifierFactory

logger = logging.getLogger("soto.aesthetic")

__all__ = ["AestheticVerifier"]

# ── URL for auto-downloading the aesthetic predictor weights ──────────
# The original weights from improved-aesthetic-predictor (sac+logos+ava1-l14-linearMSE).
# Hosted on HuggingFace for reliable access.
_AESTHETIC_WEIGHTS_URL = (
    "https://github.com/christophschuhmann/improved-aesthetic-predictor"
    "/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth"
)
_DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "soto", "aesthetic"
)


class _AestheticMLP(nn.Module):
    """MLP head for aesthetic score prediction from CLIP embeddings."""

    def __init__(self, input_size: int = 768):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


def _ensure_aesthetic_weights(checkpoint_path: str = None) -> str:
    """Return a path to the aesthetic MLP weights, downloading if needed.

    Resolution order:
      1. If ``checkpoint_path`` is given and the file exists, use it.
      2. Otherwise, download from ``_AESTHETIC_WEIGHTS_URL`` into cache.

    Returns:
        Local file path to the ``.pth`` weights.
    """
    # 1. Explicit path provided and file exists
    if checkpoint_path and os.path.isfile(checkpoint_path):
        return checkpoint_path

    # 2. Check if already cached
    cache_dir = _DEFAULT_CACHE_DIR
    cached_path = os.path.join(cache_dir, "sac+logos+ava1-l14-linearMSE.pth")

    if os.path.isfile(cached_path):
        logger.info(f"Using cached aesthetic weights: {cached_path}")
        return cached_path

    # 3. Try huggingface_hub first (most reliable)
    try:
        from huggingface_hub import hf_hub_download
        logger.info("Downloading aesthetic weights via huggingface_hub...")
        path = hf_hub_download(
            repo_id="camenduru/improved-aesthetic-predictor",
            filename="sac+logos+ava1-l14-linearMSE.pth",
        )
        logger.info(f"Downloaded aesthetic weights to: {path}")
        return path
    except Exception as e:
        logger.debug(f"huggingface_hub download failed: {e}")

    # 4. Fall back to torch.hub download from GitHub
    logger.info(f"Downloading aesthetic weights from GitHub...")
    os.makedirs(cache_dir, exist_ok=True)
    try:
        torch.hub.download_url_to_file(_AESTHETIC_WEIGHTS_URL, cached_path)
        logger.info(f"Downloaded aesthetic weights to: {cached_path}")
        return cached_path
    except Exception as e:
        raise RuntimeError(
            f"Failed to download aesthetic predictor weights.\n"
            f"  URL: {_AESTHETIC_WEIGHTS_URL}\n"
            f"  Error: {e}\n"
            f"You can manually download the file and pass its path via:\n"
            f"  checkpoint_path: /path/to/sac+logos+ava1-l14-linearMSE.pth"
        ) from e


@VerifierFactory.register("aesthetic")
class AestheticVerifier(BaseVerifier):
    """Aesthetic score verifier using CLIP ViT-L/14 + MLP predictor.

    Scores how aesthetically pleasing images are (prompt-independent).
    Higher scores indicate more aesthetically pleasing images.

    The MLP weights are automatically downloaded from the internet if
    ``checkpoint_path`` is not specified or the file does not exist.

    Config parameters:
        - checkpoint_path: Optional path to the aesthetic predictor MLP
          weights.  When omitted or not found, weights are auto-downloaded.
        - batch_size: Batch size for scoring (default: 32)
    """

    def __init__(self, config: Dict, device: str = "cuda"):
        super().__init__(config, device)

        # Load CLIP ViT-L/14 backbone
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=device)
        self.clip_model.eval()

        # Load aesthetic MLP head (auto-download if needed)
        ckpt_path = config.get("checkpoint_path", None)
        resolved_path = _ensure_aesthetic_weights(ckpt_path)

        self.aesthetic_mlp = _AestheticMLP(768)
        state = torch.load(resolved_path, map_location="cpu", weights_only=True)
        self.aesthetic_mlp.load_state_dict(state)
        self.aesthetic_mlp.to(device).eval()

        logger.info(f"Loaded AestheticVerifier (CLIP ViT-L/14 + MLP from {resolved_path})")

    def _score(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: List[str],
        **kwargs,
    ) -> torch.Tensor:
        """Score images by aesthetic quality.

        Args:
            images: List of PIL Images or tensor ``[B, C, H, W]``.
            prompts: List of text prompts (unused – aesthetic score is
                prompt-independent, but kept for interface compatibility).
            **kwargs: Additional parameters.

        Returns:
            Aesthetic scores tensor of shape ``[B]``.
        """
        batch_size = self.config.get("batch_size", 32)
        all_scores: List[float] = []

        with torch.no_grad():
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                batch_tensors = torch.stack(
                    [self.preprocess(img) for img in batch_images]
                ).to(self.device)

                features = self.clip_model.encode_image(batch_tensors)
                features = features / features.norm(dim=-1, keepdim=True)

                preds = self.aesthetic_mlp(features.float()).squeeze(-1)
                all_scores.extend(preds.cpu().tolist())

        return torch.tensor(all_scores, dtype=torch.float32)
