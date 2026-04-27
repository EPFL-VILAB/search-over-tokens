"""Ensemble verifier that combines multiple verifiers.

Supports two aggregation strategies:
  - **weighted**: Normalise each verifier's scores to [0, 1] (min-max) and
    compute a weighted sum.
  - **rank**: Rank-based aggregation — each verifier ranks the candidates and
    the final score is the (negative) sum of ranks, so that lower combined
    rank means better image.  This is the default aggregation strategy.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Union
from PIL import Image

from soto.verifiers.base import BaseVerifier, VerifierFactory

logger = logging.getLogger("soto.ensemble")

__all__ = ["EnsembleVerifier"]


@VerifierFactory.register("ensemble")
class EnsembleVerifier(BaseVerifier):
    """Ensemble verifier that combines scores from multiple verifiers.

    At construction the ensemble instantiates every sub-verifier listed in
    ``config["verifiers"]`` (a list of dicts, each with a ``name`` key and
    optional per-verifier config overrides) and optionally assigns a
    ``weight`` to each.

    Config parameters (top-level):
        - verifiers: List of sub-verifier specifications, e.g.::

              verifiers:
                - name: clip
                  weight: 1.0
                - name: image_reward
                  weight: 1.0
                - name: aesthetic
                  weight: 1.0
                  checkpoint_path: /path/to/mlp.pth

        - aggregation: ``"rank"`` (default) or ``"weighted"``.
            - ``rank``: Sum of per-verifier ranks; lower combined rank wins.
              The returned scores are *negative* combined ranks so that
              ``torch.topk`` still selects the best candidates.
            - ``weighted``: Weighted sum of min-max normalised scores.
        - batch_size: Forwarded to sub-verifiers as default (default: 32).
    """

    def __init__(self, config: Dict, device: str = "cuda"):
        super().__init__(config, device)

        sub_verifier_specs = config.get("verifiers", [])
        if not sub_verifier_specs:
            raise ValueError(
                "EnsembleVerifier requires a non-empty 'verifiers' list in config."
            )

        self.aggregation = config.get("aggregation", "rank")
        assert self.aggregation in ("rank", "weighted"), (
            f"Unknown aggregation: {self.aggregation}. Choose 'rank' or 'weighted'."
        )

        self.sub_verifiers: List[BaseVerifier] = []
        self.weights: List[float] = []

        for spec in sub_verifier_specs:
            name = spec["name"]
            weight = spec.get("weight", 1.0)
            # Build per-verifier config: merge top-level defaults with
            # verifier-specific overrides.
            sub_cfg = {k: v for k, v in spec.items() if k not in ("name", "weight")}
            sub_cfg.setdefault("batch_size", config.get("batch_size", 32))

            logger.info(f"  Loading sub-verifier '{name}' (weight={weight})")
            verifier = VerifierFactory.create(name, sub_cfg, device=device)
            self.sub_verifiers.append(verifier)
            self.weights.append(weight)

        # Requires images if any active sub-verifier does
        self.requires_images = any(
            v.requires_images for v, w in zip(self.sub_verifiers, self.weights) if w != 0
        )

        logger.info(
            f"EnsembleVerifier ready: {len(self.sub_verifiers)} verifiers, "
            f"aggregation={self.aggregation}"
        )

    # ------------------------------------------------------------------
    # Main scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: List[str],
        **kwargs,
    ) -> torch.Tensor:
        """Score images with the ensemble of verifiers.

        Args:
            images: List of PIL Images or tensor ``[B, C, H, W]``.
            prompts: List of text prompts.
            **kwargs: Forwarded to every sub-verifier.

        Returns:
            Aggregated scores tensor of shape ``[B]``.
        """
        n = len(images)
        all_scores: Dict[str, torch.Tensor] = {}

        for verifier, weight in zip(self.sub_verifiers, self.weights):
            if weight == 0:
                continue
            v_name = type(verifier).__name__
            try:
                scores = verifier.score(images, prompts, **kwargs)
                if not isinstance(scores, torch.Tensor):
                    scores = torch.tensor(scores, dtype=torch.float32)
                # Ensure float32 to avoid dtype mismatches in aggregation
                scores = scores.float()
                all_scores[v_name] = scores
            except Exception as e:
                logger.warning(f"  {v_name} failed: {e} — skipping")
                continue

        if not all_scores:
            logger.warning("No sub-verifiers produced scores; returning zeros.")
            return torch.zeros(n, dtype=torch.float32, device=self.device)

        if self.aggregation == "rank":
            return self._rank_aggregate(all_scores, n)
        else:
            return self._weighted_aggregate(all_scores, n)

    # ------------------------------------------------------------------
    # Aggregation strategies
    # ------------------------------------------------------------------

    def _rank_aggregate(
        self, all_scores: Dict[str, torch.Tensor], n: int
    ) -> torch.Tensor:
        """Rank-based aggregation.

        Each verifier produces a ranking of candidates.  The combined score
        for candidate *i* is the *negative* sum of its ranks across
        verifiers, so that ``torch.topk`` picks the candidates with the
        lowest (best) aggregate rank.
        """
        combined_rank = torch.zeros(n, dtype=torch.float32, device=self.device)

        for (v_name, scores), weight in zip(all_scores.items(), self.weights):
            if weight == 0:
                continue
            scores = scores.to(device=self.device, dtype=torch.float32)
            # argsort descending: rank 0 = best
            ranking = scores.argsort(descending=True)
            ranks = torch.zeros(len(scores), dtype=torch.float32, device=self.device)
            ranks[ranking] = torch.arange(len(scores), dtype=torch.float32, device=self.device)
            combined_rank += weight * ranks

        # Normalise to [0, 1]: 1 = best possible (rank 0 from all verifiers),
        # 0 = worst possible (rank N-1 from all verifiers).
        total_weight = sum(
            w for (_, _), w in zip(all_scores.items(), self.weights) if w != 0
        )
        worst_possible = total_weight * (n - 1)
        if worst_possible > 0:
            return 1.0 - combined_rank / worst_possible
        return -combined_rank

    def _weighted_aggregate(
        self, all_scores: Dict[str, torch.Tensor], n: int
    ) -> torch.Tensor:
        """Weighted sum of min-max normalised scores."""
        combined = torch.zeros(n, dtype=torch.float32, device=self.device)

        for (v_name, scores), weight in zip(all_scores.items(), self.weights):
            if weight == 0:
                continue
            scores = scores.to(device=self.device, dtype=torch.float32)
            s_min, s_max = scores.min(), scores.max()
            if s_max - s_min > 1e-8:
                normed = (scores - s_min) / (s_max - s_min)
            else:
                normed = torch.zeros_like(scores)
            combined += weight * normed

        return combined


