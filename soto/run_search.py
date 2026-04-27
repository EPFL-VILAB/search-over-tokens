#!/usr/bin/env python3
"""Main script for running search over tokens.

Supports both single GPU and distributed multi-GPU execution:
  Single GPU:  python run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_beam_ir
  Multi-GPU:   torchrun --nproc_per_node=4 run_search.py --config-name eval/flextok/geneval_flextok_ar_3b_beam_ir

For distributed execution, each GPU processes different captions independently
(dataset parallelization) - no synchronization needed!
"""

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from pathlib import Path
import json
from datetime import datetime

from soto.ar_priors.base import ARPriorFactory
from soto.search_algorithms.base import SearchAlgorithmFactory
from soto.verifiers.base import VerifierFactory

from soto.data.geneval import load_geneval_captions
from soto.data.dreambench import load_dreambench_captions
from soto.data.coco import load_coco_captions
from soto.utils import is_distributed, get_rank_info, set_seed, is_caption_complete, setup_logging


@hydra.main(version_base=None, config_path="configs", config_name="default")
def main(cfg: DictConfig):
    """Run search over tokens with optional distributed support."""

    # Get rank info (works for both single and multi-GPU)
    rank, local_rank, world_size = get_rank_info()

    # Setup logging (only rank 0 logs)
    logger = setup_logging(rank)

    # Assert CUDA is available
    assert torch.cuda.is_available(), "CUDA is not available"

    # Set device
    torch.cuda.set_device(local_rank)
    device = f"cuda:{local_rank}"

    # Set seed (different per rank for diversity)
    set_seed(cfg.get("seed", 0))

    # Log header
    logger.info("=" * 80)
    logger.info("Search over Tokens (SoTo)")
    logger.info("=" * 80)
    logger.info(f"\nMode: {'Distributed' if is_distributed() else 'Single GPU'}")
    if is_distributed():
        logger.info(f"World size: {world_size} GPUs")
    logger.info(f"Device: {device}")
    logger.info("=" * 80)
    logger.info("\nConfiguration:")
    logger.info(OmegaConf.to_yaml(cfg))
    logger.info("=" * 80)

    # Load dataset
    dataset_name = cfg.dataset.get("name") if cfg.get("dataset") else None

    if dataset_name == "geneval":
        # Load GenEval dataset (auto-downloads if needed)
        all_captions, all_metadata = load_geneval_captions(
            dataset_dir=cfg.dataset.get("dataset_dir", "datasets/geneval"),
            num_samples=cfg.dataset.get("num_samples", None)  # None = all samples
        )
    elif dataset_name == "dreambench":
        # Load DreamBench++ dataset (auto-downloads if needed)
        # metadata entries are reference image paths (str) used by DreamSim verifier
        all_captions, all_metadata = load_dreambench_captions(
            dataset_dir=cfg.dataset.get("dataset_dir", "datasets/dreambench"),
            num_samples=cfg.dataset.get("num_samples", None),  # None = all 1350
            images_dir=cfg.dataset.get("images_dir", None),
        )
    elif dataset_name == "coco":
        # Load COCO captions
        all_captions = load_coco_captions(
            dataset_dir=cfg.dataset.get("dataset_dir", "datasets/coco"),
            num_samples=cfg.dataset.get("num_samples", 300)
        )
        all_metadata = [None] * len(all_captions)
    elif dataset_name is not None:
        raise ValueError(f"Unknown dataset: '{dataset_name}'. Available: geneval, dreambench, coco")
    else:
        all_captions = None
        all_metadata = None

    if all_captions is not None:
        # Partition by rank if distributed (dataset parallelization)
        if is_distributed():
            captions = all_captions[rank::world_size]
            metadata = all_metadata[rank::world_size] if all_metadata else None
            logger.info(f"[Rank {rank}] Processing {len(captions)}/{len(all_captions)} captions")
        else:
            captions = all_captions
            metadata = all_metadata
    else:
        # Use prompts from config
        captions = cfg.get("prompts", ["A beautiful landscape"])
        if isinstance(captions, str):
            captions = [captions]
        metadata = [None] * len(captions)
        all_captions = captions

    # Create components (each rank creates independently - no sync needed!)
    logger.info("\n" + "=" * 80)
    logger.info("Creating AR Prior...")
    logger.info("=" * 80)

    ar_prior_cfg = cfg.components.ar_priors.flextok
    ar_prior_cfg = OmegaConf.to_container(ar_prior_cfg, resolve=True)
    ar_prior_cfg = ar_prior_cfg.get("ar_prior", ar_prior_cfg)
    ar_prior = ARPriorFactory.create(
        name=ar_prior_cfg.get("name"),
        config=ar_prior_cfg,
        device=device
    )
    ar_prior.set_seed(cfg.get("seed", 0))

    logger.info("\n" + "=" * 80)
    logger.info("Creating Verifier...")
    logger.info("=" * 80)

    verifier_cfg = cfg.components.verifiers.verifier
    verifier_cfg = OmegaConf.to_container(verifier_cfg, resolve=True)
    verifier_cfg = verifier_cfg.get("verifier", verifier_cfg)
    if cfg.get("verifier") is not None:
        verifier_cfg = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(verifier_cfg), cfg.verifier),
            resolve=True
        )
    logger.info(f"Effective verifier config: {verifier_cfg}")
    verifier = VerifierFactory.create(
        name=verifier_cfg.get("name"),
        config=verifier_cfg,
        device=device
    )

    logger.info("\n" + "=" * 80)
    logger.info("Creating Search Algorithm...")
    logger.info("=" * 80)

    search_cfg = cfg.components.search_algorithms.search
    search_cfg = OmegaConf.to_container(search_cfg, resolve=True)
    search_cfg = search_cfg.get("search", search_cfg)
    if cfg.get("search") is not None:
        search_cfg = OmegaConf.to_container(
            OmegaConf.merge(OmegaConf.create(search_cfg), cfg.search),
            resolve=True
        )
    logger.info(f"Effective search config: {search_cfg}")
    search_algo = SearchAlgorithmFactory.create(
        name=search_cfg.get("name"),
        ar_prior=ar_prior,
        verifier=verifier,
        config=search_cfg
    )

    # Output directory - include config name
    base_output_dir = cfg.get("output_dir", "results")
    config_name = HydraConfig.get().job.config_name
    # Remove any path separators from config name for safety
    config_name_safe = config_name.replace("/", "_").replace("\\", "_")
    timestamp = datetime.now().strftime("%m%d_%H%M%S")
    output_dir = Path(base_output_dir) / f"{config_name_safe}_{timestamp}"
    resume = cfg.get("resume", True)
    num_results = cfg.get("num_results", 1)

    # Process captions independently (no sync needed!)
    for idx, (caption, meta) in enumerate(zip(captions, metadata)):
        # Compute global index
        if is_distributed():
            global_idx = rank + idx * world_size
        else:
            global_idx = idx

        caption_output_dir = output_dir / f"caption_{global_idx}"

        # Skip if complete and resuming
        if resume and is_caption_complete(caption_output_dir):
            logger.info(f"\n[Rank {rank}] Skipping completed caption {global_idx}: {caption}")
            continue

        logger.info(f"\n[Rank {rank}] " + "=" * 70)
        logger.info(f"Caption {global_idx + 1}/{len(all_captions)}: {caption}")
        logger.info("=" * 70)

        caption_output_dir.mkdir(parents=True, exist_ok=True)

        # Save caption
        with open(caption_output_dir / "caption.txt", "w") as f:
            f.write(caption)

        # Run search
        result = search_algo.search(
            caption,
            num_results=num_results,
            output_dir=caption_output_dir,
            resume=resume,
            caption_idx=global_idx,
            metadata=meta
        )

        # Save results (each rank saves independently)
        for i, (img, score) in enumerate(zip(result.images, result.scores)):
            filename = f"result_{i}_score{score:.4f}.png"
            img.save(caption_output_dir / filename)
            logger.info(f"  Saved: {filename}")

        # Save metadata
        metadata_file = caption_output_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump({
                "caption": caption,
                "global_idx": global_idx,
                "scores": [float(s) for s in result.scores],
                "algorithm_metadata": result.metadata,
                "dataset_metadata": meta
            }, f, indent=2)

        logger.info(f"  Best score: {result.scores[0]:.4f}")

    logger.info(f"\n[Rank {rank}] " + "=" * 70)
    logger.info(f"[Rank {rank}] Done!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
