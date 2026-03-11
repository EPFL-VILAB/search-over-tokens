"""Uniform prior - baseline that samples tokens uniformly at random.

Unlike FlexTokARPrior, no AR model is used — each token is sampled
independently from the vocabulary.  This is the simplest baseline for
search-based generation.
"""

import torch
import logging
from typing import Dict, List, Union, Tuple, Optional

from sot.ar_priors.base import BaseARPrior, ARPriorFactory
from sot.ar_priors.flextok_wrapper import FlexTokARPrior
from flextok.flextok_wrapper import FlexTokFromHub

logger = logging.getLogger("sot.uniform")

__all__ = ["UniformPrior"]


@ARPriorFactory.register("uniform")
class UniformPrior(FlexTokARPrior):
    """Uniform prior - sample tokens uniformly at random.

    Inherits decode_tokens and get_noise_per_prompt from FlexTokARPrior.
    Does NOT load an AR model — only the FlexTok decoder is initialised.

    Config parameters:
        - vocab_size: Size of token vocabulary (default: 64000)
        - max_tokens: Maximum sequence length (default: 256)
        - tokenizer_model: HuggingFace model ID for FlexTok decoder
        - decode_kwargs: Decoding parameters (timesteps, guidance_scale, etc.)
    """

    def __init__(self, config: Dict, device: str = "cuda"):
        # Skip FlexTokARPrior.__init__ — it loads an AR model we don't need.
        BaseARPrior.__init__(self, config, device)

        self.vocab_size = config.get("vocab_size", 64000)
        self.max_tokens = config.get("max_tokens", 256)

        # Load only the FlexTok decoder (no AR model)
        tokenizer_model = config.get("tokenizer_model", "EPFL-VILAB/flextok_d18_d28_dfn")
        self._flextok_model = FlexTokFromHub.from_pretrained(tokenizer_model).to(device).eval()
        self._flextok_model.pipeline.noise_read_key = "fixed_noise"

        # Attributes used by decode_tokens / get_noise_per_prompt
        self.decode_kwargs = config.get("decode_kwargs", {})
        self.vae_image_sizes = config.get("vae_image_sizes", 32)
        self._is_2d_grid = False
        self._image_tokenizer = None
        self.generation_kwargs = {}  # only used for seed fallback in get_noise_per_prompt
        self.noise_cache = None
        self.noise_caption_idx = -1

        logger.info(
            f"UniformPrior initialised: vocab_size={self.vocab_size}, "
            f"max_tokens={self.max_tokens}"
        )

    def generate_next_tokens(
        self,
        prompt: Union[str, List[str]],
        current_tokens: torch.Tensor,
        num_new_tokens: int = 1,
        num_samples: int = 1,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Sample next tokens uniformly at random and append to current sequences.

        Args:
            prompt: Ignored — uniform prior does not condition on text.
            current_tokens: Current token sequences [batch, seq_len].
            num_new_tokens: Number of new tokens to append per candidate.
            num_samples: Number of candidates per beam (branching factor).

        Returns:
            (full_tokens, None):
                full_tokens: [batch * num_samples, seq_len + num_new_tokens]
                probs: always None (uniform has no meaningful probabilities)
        """
        batch_size = current_tokens.size(0)
        total_samples = batch_size * num_samples

        # Expand current tokens: each beam is repeated num_samples times
        # [batch, seq_len] -> [batch * num_samples, seq_len]
        expanded = current_tokens.repeat_interleave(num_samples, dim=0)

        # Draw random new tokens — use an explicit Generator tied to self.device
        # so the seed works correctly regardless of which GPU is the current device
        # (torch.cuda.manual_seed only seeds cuda:0 by default, not e.g. cuda:2)
        seed = kwargs.get("seed", 0)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        new_tokens = torch.randint(
            0, self.vocab_size,
            (total_samples, num_new_tokens),
            device=self.device,
            generator=generator
        )

        # Concatenate to form full sequences
        full_tokens = torch.cat([expanded, new_tokens], dim=1)

        return full_tokens, None

    def get_vocab_size(self) -> int:
        return self.vocab_size

    def get_max_tokens(self) -> int:
        return self.max_tokens
