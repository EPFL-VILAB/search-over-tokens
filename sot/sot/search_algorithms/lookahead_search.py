"""Lookahead beam search: beam search with AR-completed partial sequences.

Identical to beam search except that, before decoding/scoring, partial
token sequences are extended by extra greedy AR tokens.  This gives the
decoder more context and produces better intermediate images for the
verifier.

Only the *original* (non-extended) tokens are kept in the beam.

Config parameters (on top of beam search):
    - lookahead_number: Extra AR tokens for intermediate decoding.
          -1 = extend to max_tokens (complete to end).  Default: -1.
    - max_lookahead_step: Maximum search step (0-based) for which
          lookahead is applied.  At steps >= this value, behaves like
          plain beam search (no extension).
          -1 = always apply lookahead.  Default: -1.

Examples:
    Janus:    lookahead_number: -1, max_lookahead_step: -1   → always extend to end
    FlexTok:  lookahead_number: 8,  max_lookahead_step: -1   → always extend by 8 tokens
    Infinity: lookahead_number: -1, max_lookahead_step: 8    → extend to end for steps 0-7,
                                                                pure beam search for steps 8+
"""

import torch
import logging
from typing import Union, List

from sot.search_algorithms.base import SearchAlgorithmFactory, SearchResult
from sot.search_algorithms.beam_search import BeamSearch

logger = logging.getLogger("sot.lookahead_search")

__all__ = ["LookaheadSearch"]


@SearchAlgorithmFactory.register("lookahead")
class LookaheadSearch(BeamSearch):
    """Beam search + configurable lookahead AR completion.

    Overrides ``_prepare_candidates_for_decode`` to extend partial
    sequences by greedy AR tokens before decoding and scoring.

    The extension is controlled by two parameters:
        - ``lookahead_number``: How many extra tokens to add (-1 = all remaining).
        - ``max_lookahead_step``: Stop doing lookahead beyond this step (-1 = never stop).
    """

    def _prepare_candidates_for_decode(
        self, candidates: torch.Tensor, prompt, step: int = 0
    ) -> torch.Tensor:
        """Extend candidates by greedy AR tokens, respecting step limits."""
        max_lookahead_step = self.config.get("max_lookahead_step", -1)

        # Beyond max_lookahead_step → pure beam search (no extension)
        if max_lookahead_step >= 0 and step >= max_lookahead_step:
            return candidates

        lookahead_number = self.config.get("lookahead_number", -1)
        max_tokens = self.ar_prior.get_max_tokens()
        current_len = candidates.size(1)

        if current_len >= max_tokens:
            return candidates

        # Determine how many extra tokens to add
        if lookahead_number == -1:
            n_extra = max_tokens - current_len  # extend to end
        else:
            n_extra = min(lookahead_number, max_tokens - current_len)

        if n_extra <= 0:
            return candidates

        logger.info(
            f"  Lookahead: extending {len(candidates)} candidates "
            f"by {n_extra} tokens (greedy, step {step})..."
        )
        # Save and restore the main generator state so the lookahead rollout
        # does not consume RNG state from the beam search generator.  This
        # ensures beam decisions are identical to plain beam search.
        saved_state = self.ar_prior.rng.get_state()
        extended, _ = self.ar_prior.generate_next_tokens(
            prompt,
            candidates,
            num_new_tokens=n_extra,
            num_samples=1,
            sample=False,  # greedy rollout: always take the most probable token
        )
        self.ar_prior.rng.set_state(saved_state)
        return extended
