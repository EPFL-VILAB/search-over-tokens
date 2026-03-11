"""Best-of-N: Sample N complete sequences and return best according to verifier.

  1. Generate N complete token sequences from the AR prior in one batched call
  2. Decode all sequences to images
  3. Score all images with the verifier
  4. Return the top-K results

Diversity comes from multinomial sampling (no explicit seed changes).
"""

import torch
import logging
from typing import Union, List, Optional
from pathlib import Path

from sot.search_algorithms.base import BaseSearchAlgorithm, SearchAlgorithmFactory, SearchResult

logger = logging.getLogger("sot.best_of_n")

__all__ = ["BestOfN"]


@SearchAlgorithmFactory.register("best_of_n")
class BestOfN(BaseSearchAlgorithm):
    """Sample N complete sequences and return best according to verifier.

    This is the simplest search strategy — no tree, no beam, just
    generate-and-rank.

    Config parameters:
        - n_samples: Number of samples to generate (default: 50)
        - batch_size: Max samples per batch (default: n_samples). Set this
          to limit GPU memory when n_samples is large.
        - decode_timesteps: Diffusion timesteps for decoding (default: 20)
    """

    def search(
        self,
        prompt: Union[str, List[str]],
        num_results: int = 1,
        seed: int = None,
        output_dir: Optional[Path] = None,
        resume: bool = True,
        **kwargs
    ) -> SearchResult:
        """Run best-of-N sampling.

        Args:
            prompt: Text prompt or class label
            num_results: Number of results to return
            output_dir: Directory for saving results (optional)
            resume: Whether to resume from checkpoint
            **kwargs: Additional parameters (metadata, caption_idx, etc.)

        Returns:
            SearchResult with top-scored samples
        """
        if seed is not None:
            self.ar_prior.set_seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        n_samples = self.config.get("n_samples", 50)
        batch_size = self.config.get("batch_size", n_samples)
        max_tokens = self.ar_prior.get_max_tokens()

        logger.info(
            f"Best-of-{n_samples}: Generating {n_samples} complete sequences "
            f"of {max_tokens} tokens..."
        )

        all_tokens_list = []
        all_images = []
        all_scores_list = []

        for start in range(0, n_samples, batch_size):
            bs = min(batch_size, n_samples - start)

            # Generate
            empty_tokens = torch.zeros(
                (bs, 0), dtype=torch.long, device=self.ar_prior.device
            )
            tokens, probs = self.ar_prior.generate_next_tokens(
                prompt, empty_tokens, num_new_tokens=max_tokens,
                sample=self.config.get("sample", True),
            )
            all_tokens_list.append(tokens)

            # Decode
            images = self.ar_prior.decode_tokens(
                tokens,
                caption_idx=kwargs.get("caption_idx", 0),
                use_same_noise_per_prompt=False,
            )
            all_images.extend(images)

            # Score
            prompts_list = ([prompt] if isinstance(prompt, str) else prompt) * len(images)
            score_kwargs = dict(kwargs)
            if probs is not None:
                score_kwargs["log_probs"] = probs
            scores_batch = self.verifier.score(images, prompts_list, **score_kwargs)
            all_scores_list.append(scores_batch)

        all_tokens = torch.cat(all_tokens_list, dim=0)
        scores = torch.cat(all_scores_list, dim=0)
        images = all_images

        # Select top results
        top_k = min(num_results, len(all_tokens))
        top_idx = scores.topk(top_k).indices
        final_tokens = all_tokens[top_idx]
        final_scores = scores[top_idx]
        final_images = [images[i] for i in top_idx.cpu().numpy()]

        logger.info(f"Best score: {final_scores[0]:.4f}")

        return SearchResult(
            tokens=final_tokens,
            images=final_images,
            scores=final_scores,
            metadata={
                "algorithm": "best_of_n",
                "n_samples": n_samples,
            },
        )
