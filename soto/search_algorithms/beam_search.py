"""Beam search over token sequences."""

import math
import torch
import logging
from typing import Union, List, Optional, Tuple
from pathlib import Path

from tqdm.auto import tqdm

from soto.search_algorithms.base import BaseSearchAlgorithm, SearchAlgorithmFactory, SearchResult

logger = logging.getLogger("soto.beam_search")

__all__ = ["BeamSearch"]


@SearchAlgorithmFactory.register("beam")
class BeamSearch(BaseSearchAlgorithm):
    """Beam search over token sequences.

    Incrementally generates tokens while maintaining top-K candidates
    based on verifier scores at each step.

    Config parameters:
        - beam_width: Number of beams to maintain (default: 5)
        - max_steps: Maximum search steps (default: 9); overridden by list length when
          token_schedule is a list
        - candidates_per_beam: Number of candidates per beam to sample (default: 10)
        - token_schedule: Token generation schedule. Options:
            - "geometric": cumulative counts 1, base, base², … (default)
            - "linear": cumulative counts tokens_per_step, 2×, 3×, …
            - "fixed": one token per step
            - list of ints: on models with get_scale_boundaries (e.g. Infinity),
              interpreted as 1-indexed scale numbers — e.g. [5, 6, 7, 8, 9] searches
              scales 5-9 and auto-completes scales 1-4 as a prefix first.
              On other models, treated as explicit cumulative token counts.
        - geometric_base: Base for geometric schedule (default: 2)
        - tokens_per_step: Tokens per step for linear schedule (default: 32)
        - decode_batch_size: Batch size for decoding to avoid OOM (default: 50)
        - complete_after_search: AR-complete remaining tokens after beam search before
          final decode. Useful when max_steps < total scales (e.g. Infinity). (default: False)
    """

    def search(
        self,
        prompt: Union[str, List[str]],
        num_results: int = 1,
        seed: int = None,
        output_dir: Optional[Path] = None,
        resume: bool = True,
        initial_tokens: Optional[torch.Tensor] = None,
        **kwargs
    ) -> SearchResult:
        """Run beam search with optional resume.

        Args:
            prompt: Text prompt or class label
            num_results: Number of final results to return
            output_dir: Directory for checkpoints (enables resume)
            resume: Whether to resume from checkpoint
            initial_tokens: Optional tensor of shape [num_beams, seq_len] to seed the
                search from a previously found token prefix (e.g. from a prior step).
                Overrides checkpoint resumption when provided.
            **kwargs: Additional parameters (metadata, caption_idx, etc.)

        Returns:
            SearchResult with top-scored sequences
        """
        from soto.utils import save_checkpoint, load_checkpoint

        if seed is not None:
            self.ar_prior.set_seed(seed)

        beam_width = self.config.get("beam_width", 5)
        candidates_per_beam = self.config.get("candidates_per_beam", 10)
        token_schedule = self.config.get("token_schedule", "geometric")
        max_steps = len(token_schedule) if isinstance(token_schedule, list) else self.config.get("max_steps", 9)
        decode_batch_size = self.config.get("decode_batch_size", 50)
        scale_first_step = self.config.get("scale_first_step", False)
        requires_images = self.verifier.requires_images

        current_tokens, start_step, beam_scores = self._init_beams(
            initial_tokens, resume, output_dir, load_checkpoint
        )

        search_step_schedule, scale_steps, scale_indexed = self._build_schedule(
            token_schedule, max_steps, current_tokens.size(1)
        )
        max_steps = len(search_step_schedule)
        logger.debug(
            f"Beam search: beam_width={beam_width}, max_steps={max_steps}, "
            f"candidates_per_beam={candidates_per_beam}, schedule={search_step_schedule}"
        )

        if scale_indexed and scale_steps[0] > 1 and initial_tokens is None and start_step == 0:
            current_tokens = self._complete_prefix(prompt, scale_steps, **kwargs)

        # decode_timesteps is NOT set here — the AR prior's decode_kwargs.timesteps
        # is the single source of truth.  Only pass search-level concerns.
        decode_kw = dict(
            decode_batch_size=decode_batch_size,
            caption_idx=kwargs.get("caption_idx", 0),
        )
        step_images, step_scores = [], []
        last_candidates = current_tokens
        last_scores = torch.zeros(1, device=self.ar_prior.device)

        bar = tqdm(range(start_step, max_steps), desc="Beam search", unit="step", leave=True)
        for step in bar:
            bar.set_description(f"Step {step + 1}/{max_steps}")
            new_token_len = search_step_schedule[step] - current_tokens.size(1)
            n_beams = current_tokens.size(0)
            effective_candidates = (
                candidates_per_beam * (beam_width // n_beams)
                if scale_first_step and n_beams < beam_width
                else candidates_per_beam
            )

            candidates, probs = self.ar_prior.generate_next_tokens(
                prompt, current_tokens,
                num_new_tokens=new_token_len, num_samples=effective_candidates,
                **{**kwargs,
                   "sample": self.config.get("sample", False),
                   "replacement": self.config.get("replacement", False)},
            )

            if requires_images:
                images = self._batched_decode(
                    self._prepare_candidates_for_decode(candidates, prompt, step=step),
                    **decode_kw,
                )
            else:
                images = [None] * len(candidates)

            prompts_list = ([prompt] if isinstance(prompt, str) else prompt) * len(images)
            score_kwargs = dict(kwargs)
            if probs is not None:
                score_kwargs["log_probs"] = probs
                score_kwargs["num_tokens"] = candidates.shape[1]
            scores = self.verifier.score(images, prompts_list, **score_kwargs)

            last_candidates, last_scores = candidates, scores
            top_k_idx = scores.topk(min(beam_width, len(candidates))).indices
            current_tokens = candidates[top_k_idx]
            beam_scores = scores[top_k_idx]

            step_images.append(images[top_k_idx[0]])
            step_scores.append(beam_scores[0].item())
            bar.set_postfix(score=f"{beam_scores[0]:.3f}", tokens=current_tokens[0].size(0))

            if output_dir:
                # Persist beam state after each step so long-running searches
                # can be resumed if interrupted (pass output_dir + resume=True).
                save_checkpoint(output_dir, step + 1,
                                current_tokens.cpu().tolist(), beam_scores.cpu().tolist())

        # Final selection: pick top num_results from the full last-step candidate pool.
        n_show = min(num_results, len(last_scores))
        show_idx = last_scores.topk(n_show).indices
        display_tokens = last_candidates[show_idx]
        final_scores = last_scores[show_idx]
        final_tokens = current_tokens  # pruned beams — used as seed for carry-forward

        if self.config.get("complete_after_search", False):
            remaining = self.ar_prior.get_max_tokens() - display_tokens.size(1)
            if remaining > 0:
                display_tokens, _ = self.ar_prior.generate_next_tokens(
                    prompt, display_tokens, num_new_tokens=remaining,
                    sample=False,  # greedy completion for reproducibility
                )
            final_tokens = display_tokens

        final_images = self._batched_decode(
            self._prepare_candidates_for_decode(display_tokens, prompt, step=max_steps),
            **decode_kw,
        )

        # Re-score after AR completion so result.scores reflects the final decoded images
        if self.config.get("complete_after_search", False):
            prompts_list = ([prompt] if isinstance(prompt, str) else prompt) * len(final_images)
            final_scores = self.verifier.score(final_images, prompts_list)

        logger.debug(f"Final best score: {final_scores[0]:.4f}")
        return SearchResult(
            tokens=final_tokens,
            images=final_images,
            scores=final_scores,
            display_tokens=display_tokens,
            metadata={
                "algorithm": "beam_search",
                "beam_width": beam_width,
                "num_steps": max_steps,
                "token_schedule": token_schedule,
                "step_schedule": search_step_schedule,
                "candidates_per_beam": candidates_per_beam,
            },
            step_images=step_images,
            step_scores=step_scores,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_beams(
        self,
        initial_tokens: Optional[torch.Tensor],
        resume: bool,
        output_dir: Optional[Path],
        load_checkpoint,
    ) -> Tuple[torch.Tensor, int, Optional[torch.Tensor]]:
        """Initialise beam tokens from initial_tokens, checkpoint, or empty.

        Checkpointing is useful for large-scale searches (many prompts / many steps)
        where the job may be interrupted and resumed later without re-running from
        scratch — especially when the AR model is slow or unavailable for re-inference.

        Returns:
            (current_tokens, start_step, beam_scores)
        """
        if initial_tokens is not None:
            tokens = initial_tokens.to(self.ar_prior.device)
            logger.info(f"Starting from {tokens.size(0)} provided beam(s) of length {tokens.size(1)}")
            return tokens, 0, None

        current_tokens = torch.zeros((1, 0), dtype=torch.long, device=self.ar_prior.device)
        start_step, beam_scores = 0, None

        if resume and output_dir:
            start_step, saved_tokens, saved_scores = load_checkpoint(output_dir)
            if saved_tokens and saved_tokens[0]:
                current_tokens = torch.tensor(saved_tokens, device=self.ar_prior.device)
                if saved_scores:
                    beam_scores = torch.tensor(saved_scores, device=self.ar_prior.device)
                logger.info(f"Resuming from step {start_step}")

        return current_tokens, start_step, beam_scores

    def _build_schedule(
        self,
        token_schedule,
        max_steps: int,
        initial_len: int,
    ) -> Tuple[List[int], Optional[List[int]], bool]:
        """Build the cumulative-token step schedule.

        When token_schedule is a list and the AR prior exposes get_scale_boundaries
        (e.g. Infinity), values are treated as 1-indexed scale numbers. Otherwise,
        they are explicit cumulative token counts.

        Returns:
            (schedule, scale_steps, scale_indexed)
        """
        max_tokens = self.ar_prior.get_max_tokens()
        tokens_per_step = self.config.get("tokens_per_step", 32)
        geometric_base = self.config.get("geometric_base", 2)

        scale_indexed = isinstance(token_schedule, list) and hasattr(self.ar_prior, "get_scale_boundaries")
        if scale_indexed:
            scale_steps = token_schedule
            all_bounds = self.ar_prior.get_scale_boundaries(max(scale_steps))
            schedule = [all_bounds[i - 1] for i in scale_steps]  # 1-indexed
        else:
            scale_steps = None
            schedule = self.get_search_step_schedule(
                token_schedule, max_steps, max_tokens, tokens_per_step, geometric_base
            )
            if initial_len > 0:
                schedule = [t + initial_len for t in schedule]

        return schedule, scale_steps, scale_indexed

    def _complete_prefix(self, prompt, scale_steps: List[int], **kwargs) -> torch.Tensor:
        """AR-complete scales 1..(scale_steps[0]-1) as a fixed prefix before beam search."""
        prefix_count = scale_steps[0] - 1
        prefix_len = self.ar_prior.get_scale_boundaries(prefix_count)[-1]
        logger.info(
            f"AR-completing prefix: scales 1-{prefix_count} ({prefix_len} tokens) "
            f"before beam search at scale {scale_steps[0]}"
        )
        tokens, _ = self.ar_prior.generate_next_tokens(
            prompt,
            torch.zeros((1, 0), dtype=torch.long, device=self.ar_prior.device),
            num_new_tokens=prefix_len, num_samples=1,
            **kwargs
        )
        return tokens

    def _batched_decode(self, tokens: torch.Tensor, decode_batch_size: int = 50, **decode_kwargs) -> List:
        """Decode token sequences in batches to avoid GPU OOM."""
        all_images: List = []
        for i in range(math.ceil(len(tokens) / decode_batch_size)):
            start = i * decode_batch_size
            all_images.extend(
                self.ar_prior.decode_tokens(tokens[start:start + decode_batch_size], **decode_kwargs)
            )
            torch.cuda.empty_cache()
        return all_images

    def _prepare_candidates_for_decode(  # noqa: ARG002
        self, candidates: torch.Tensor, prompt, step: int = 0
    ) -> torch.Tensor:
        """Hook: transform candidates before decoding/scoring.

        Subclasses (e.g. LookaheadSearch) can override this to extend
        partial sequences with extra AR tokens for better intermediate
        images. The default implementation returns candidates unchanged.
        ``prompt`` and ``step`` are available to subclasses.
        """
        return candidates

    def get_search_step_schedule(
        self,
        token_schedule,
        max_steps: int,
        max_tokens: int,
        tokens_per_step: int,
        geometric_base: float = 2,
    ) -> List[int]:
        """Return a list of cumulative token counts, one per search step.

        Args:
            token_schedule: "geometric", "linear", "fixed", or a list of
                explicit cumulative token counts.
            max_steps: Number of steps (ignored when token_schedule is a list).
            max_tokens: Upper bound on cumulative token count.
            tokens_per_step: Step size for the "linear" schedule.
            geometric_base: Multiplicative base for the "geometric" schedule.
        """
        if isinstance(token_schedule, list):
            return [min(t, max_tokens) for t in token_schedule]
        elif token_schedule == "geometric":
            return [min(int(geometric_base ** i), max_tokens) for i in range(max_steps)]
        elif token_schedule == "linear":
            return [min(tokens_per_step * (i + 1), max_tokens) for i in range(max_steps)]
        else:  # "fixed"
            return [min(i + 1, max_tokens) for i in range(max_steps)]
