"""CLIP-based image-text alignment verifier."""

import torch
import logging
from PIL import Image
from typing import Dict, List, Union

try:
    import clip
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False

from soto.verifiers.base import BaseVerifier, VerifierFactory

logger = logging.getLogger("soto.clip")

__all__ = ["CLIPVerifier"]


@VerifierFactory.register("clip")
class CLIPVerifier(BaseVerifier):
    """CLIP-based image-text alignment scorer.
    
    Uses OpenAI's CLIP model to compute similarity between images and text prompts.
    
    Config parameters:
        - model_name: CLIP model name (default: "ViT-B/16")
          Options: "RN50", "RN101", "RN50x4", "RN50x16", "RN50x64",
                   "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"
    """
    
    def __init__(self, config: Dict, device: str = "cuda"):
        if not CLIP_AVAILABLE:
            raise ImportError(
                "CLIP package not available. Install it with:\n"
                "  pip install git+https://github.com/openai/CLIP.git --no-build-isolation\n"
                "Or install the optional dependency:\n"
                "  pip install -e 'soto[clip]' --no-build-isolation"
            )
        
        super().__init__(config, device)
        
        model_name = config.get("model_name", "ViT-B/16")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        
        logger.info(f"Loaded CLIP model: {model_name}")
    
    def _score(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: List[str],
        **kwargs
    ) -> torch.Tensor:
        """Score image-text alignment using CLIP.
        
        Args:
            images: List of PIL Images or tensor [B, C, H, W]
            prompts: List of text prompts (one per image)
            **kwargs: Additional parameters
        
        Returns:
            CLIP similarity scores, shape [B]
        """
        # Preprocess images
        if isinstance(images[0], Image.Image):
            image_tensors = torch.stack([
                self.preprocess(img) for img in images
            ]).to(self.device)
        else:
            # Assume already preprocessed tensor
            image_tensors = images
            if image_tensors.device != self.device:
                image_tensors = image_tensors.to(self.device)
        
        # Tokenize text
        text_tokens = clip.tokenize(prompts, truncate=True).to(self.device)
        
        # Compute similarity
        with torch.no_grad():
            image_features = self.model.encode_image(image_tensors)
            text_features = self.model.encode_text(text_tokens)
            
            # Normalize features
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Compute cosine similarity scaled to match ImageReward range
            scores = 2.5 * (image_features * text_features).sum(dim=-1).clamp(min=0)
        
        return scores

