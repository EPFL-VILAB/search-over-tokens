"""Likelihood (log-probability) verifier.

Uses the AR model's sequence log-probabilities as a score.  The search
algorithm passes ``log_probs`` as a keyword argument to ``score()``.

This is a lightweight verifier that does **not** load any additional
model — it simply forwards the probabilities that the AR prior already
computed during token generation.
"""

import torch
import logging
from typing import Dict, List, Union
from PIL import Image

from sot.verifiers.base import BaseVerifier, VerifierFactory

logger = logging.getLogger("sot.likelihood")

__all__ = ["LikelihoodVerifier"]


@VerifierFactory.register("likelihood")
class LikelihoodVerifier(BaseVerifier):
    """Score candidates by their AR-model log-probability.

    The search algorithm is expected to pass the ``log_probs`` tensor
    (shape ``[N]`` or ``[N, 1]``) via ``kwargs``.  If ``log_probs`` is
    not provided, the verifier returns zeros (so that ensemble fallback
    still works).

    Config parameters:
        - (none — this verifier has no tuneable settings)
    """

    requires_images = False

    def __init__(self, config: Dict, device: str = "cuda"):
        super().__init__(config, device)
        if config.get("requires_images", False):
            self.requires_images = True
        logger.info("LikelihoodVerifier ready (uses AR-model log-probs)")

    def _score(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: List[str],
        **kwargs,
    ) -> torch.Tensor:
        """Return the log-probability of each candidate.

        Args:
            images: List of PIL Images (ignored by this verifier).
            prompts: Text prompts (ignored).
            **kwargs: Must contain ``log_probs`` — a tensor of shape
                ``[N]`` or ``[N, 1]``.

        Returns:
            Log-probability scores, shape ``[N]``.
        """
        log_probs = kwargs.get("log_probs", None)

        n = len(images)

        if log_probs is None:
            logger.warning(
                "LikelihoodVerifier: no log_probs in kwargs — returning zeros. "
                "Make sure the search algorithm passes log_probs."
            )
            return torch.zeros(n, dtype=torch.float32)

        if not isinstance(log_probs, torch.Tensor):
            log_probs = torch.tensor(log_probs, dtype=torch.float32)

        # Flatten [N, 1] → [N]
        log_probs = log_probs.squeeze()

        if log_probs.dim() == 0:
            log_probs = log_probs.unsqueeze(0)

        # Ensure same length as images
        if len(log_probs) != n:
            logger.warning(
                f"LikelihoodVerifier: log_probs length ({len(log_probs)}) != "
                f"images length ({n}). Padding/truncating."
            )
            if len(log_probs) > n:
                log_probs = log_probs[:n]
            else:
                pad = torch.zeros(n - len(log_probs), dtype=log_probs.dtype)
                log_probs = torch.cat([log_probs, pad])

        num_tokens = kwargs.get("num_tokens", None)
        if num_tokens is not None and num_tokens > 0:
            log_probs = log_probs / num_tokens

        return log_probs.float().cpu()

