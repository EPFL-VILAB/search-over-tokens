#!/usr/bin/env python3
"""CLI for generating images with FlexTok AR (Text-to-Image and Class-to-Image).

Usage (from flextok_ar/ directory with `pip install -e ..`):

  # Text-to-Image: generate from text prompt
  python generate.py --mode t2i --model-id ZhitongGao/FlexAR-3B-T2I --prompt "A serene lake at sunset" --output lake.png

  # Class-to-Image: generate from ImageNet class label (0-999)
  python generate.py --mode c2i --model-id ZhitongGao/FlexAR-1B-C2I --class-label 285 --output cat.png

  # Multiple samples
  python generate.py --mode t2i --model-id ZhitongGao/FlexAR-3B-T2I --prompt "A cat" --num-samples 4 --output cat_{i}.png
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

from flextok_ar.utils.helpers import load_model, generate_t2i, generate_c2i, tensor_to_pil


def main():
    parser = argparse.ArgumentParser(
        description="Generate images with FlexTok AR (T2I and C2I)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["t2i", "c2i"], required=True,
                        help="t2i = text-to-image, c2i = class-to-image")
    parser.add_argument("--model-id", type=str, required=True,
                        help="HuggingFace model ID (e.g., ZhitongGao/FlexAR-3B-T2I or ZhitongGao/FlexAR-1B-C2I)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path (use {i} for multiple samples, e.g., out_{i}.png)")

    # Mode-specific
    parser.add_argument("--prompt", type=str, help="Text prompt (for t2i mode)")
    parser.add_argument("--class-label", type=int, default=285,
                        help="ImageNet class label 0-999 (for c2i mode, default: 285=cat)")

    # Optional
    parser.add_argument("--num-samples", type=int, default=1,
                        help="Number of images to generate (default: 1)")
    parser.add_argument("--cfg-factor", type=float,
                        help="Classifier-free guidance scale (default: 3.0 for t2i, 1.5 for c2i)")
    parser.add_argument("--temperature", type=float,
                        help="Sampling temperature (default: 1.0)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device (default: cuda:0)")
    parser.add_argument("--debug-tokens", action="store_true", help="Print first 10 token IDs")

    args = parser.parse_args()

    if args.mode == "t2i" and not args.prompt:
        parser.error("--prompt is required for t2i mode")
    if args.mode == "c2i" and (args.class_label < 0 or args.class_label >= 1000):
        print(f"Warning: Class label {args.class_label} outside ImageNet range (0-999)")

    model, tokenizer, cfg = load_model(model_id=args.model_id, device=args.device)

    if args.debug_tokens:
        _run_debug_tokens(args, model, cfg)

    images = []
    for idx in range(args.num_samples):
        # Increment seed per iteration so each run is independent
        seed_i = args.seed + idx if args.seed is not None else None
        if args.mode == "t2i":
            result = generate_t2i(
                model=model,
                prompt=args.prompt,
                cfg=cfg,
                num_samples=1,
                cfg_factor=args.cfg_factor,
                temperature=args.temperature,
                seed=seed_i,
                device=args.device,
                verbose=(idx == 0),
            )
        else:
            result = generate_c2i(
                model=model,
                class_label=args.class_label,
                cfg=cfg,
                num_samples=1,
                cfg_factor=args.cfg_factor,
                temperature=args.temperature,
                seed=seed_i,
                device=args.device,
                verbose=(idx == 0),
            )
        images.extend(result)

    print(f"Saving {len(images)} image(s)...")
    for i, img_tensor in enumerate(images):
        out_path = args.output.replace("{i}", str(i)) if args.num_samples > 1 else args.output
        pil_img = tensor_to_pil(img_tensor)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pil_img.save(out_path)
        print(f"  Saved: {out_path}")
    print("Done!")


def _run_debug_tokens(args, model, cfg):
    gen_cfg = cfg.generation
    gen_kwargs = {
        "sample": gen_cfg.get("sample", True),
        "temperature": args.temperature if args.temperature is not None else gen_cfg.get("temperature", 1.0),
        "top_k": gen_cfg.get("top_k", 0),
        "top_p": gen_cfg.get("top_p", 0.0),
        "cfg_factor": args.cfg_factor or gen_cfg.get("cfg_factor", 3.0),
        "num_keep_tokens": gen_cfg.get("num_keep_tokens", 256),
        "num_samples": 1,
    }
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        gen_kwargs["generator"] = torch.Generator(device=args.device).manual_seed(args.seed)

    print("\n" + "=" * 80)
    print("DEBUG: First 10 generated token IDs")
    print("=" * 80)
    if args.mode == "t2i":
        data = {"text": [args.prompt]}
    else:
        data = {"target": torch.tensor([args.class_label], dtype=torch.long, device=args.device)}
    token_ids = model.generate_ids(data, **gen_kwargs)
    print(f"Shape: {token_ids.shape}")
    print(f"First 10: {token_ids[0, :10].tolist()}")
    print("=" * 80 + "\n")
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)


if __name__ == "__main__":
    main()
