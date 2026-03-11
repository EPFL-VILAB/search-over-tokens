"""Helper utilities for model handling and image generation."""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Callable, List, Optional, Tuple
from omegaconf import OmegaConf
from hydra.utils import instantiate
from pathlib import Path
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from PIL import Image

logger = logging.getLogger("flextok_ar")

__all__ = [
    "load_model",
    "unwrap_model",
    "set_generation_mode",
    "tensor_to_pil",
    "generate_t2i",
    "generate_c2i",
    "progressive_decode",
]


def progressive_decode(
    model,
    tokenizer,
    cfg: OmegaConf,
    *,
    prompt: Optional[str] = None,
    class_label: Optional[int] = None,
    token_counts: Optional[List[int]] = None,
    seed: Optional[int] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[torch.Tensor, List]:
    """Generate tokens and decode at increasing prefix lengths for progressive visualization.

    Args:
        model: ImageGenerationWrapper (T2I or L2I).
        tokenizer: FlexTok tokenizer (ImageResamplerTokenizer).
        cfg: Model config.
        prompt: Text prompt (for T2I). Mutually exclusive with class_label.
        class_label: ImageNet class index 0-999 (for L2I). Mutually exclusive with prompt.
        token_counts: Prefix lengths to decode, default [1, 4, 8, 16, 32, 64, 128, 256].
        seed: Random seed.
        device: Device string.
        verbose: Print progress (e.g. "N tokens decoded").

    Returns:
        (token_ids, list_of_PIL_images) — token_ids shape [1, 256], images at each prefix length.
    """
    if (prompt is None) == (class_label is None):
        raise ValueError("Provide either prompt (T2I) or class_label (L2I), not both or neither.")
    if token_counts is None:
        token_counts = [1, 4, 8, 16, 32, 64, 128, 256]
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    gen_cfg = cfg.generation
    if prompt is not None:
        data_dict = {"text": [prompt]}
    else:
        data_dict = {"target": torch.tensor([class_label], dtype=torch.long, device=device)}

    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
        token_ids = model.generate_ids(
            data_dict,
            sample=gen_cfg.get("sample", True),
            temperature=gen_cfg.get("temperature", 1.0),
            top_k=gen_cfg.get("top_k", 0),
            top_p=gen_cfg.get("top_p", 0.0),
            cfg_factor=gen_cfg.get("cfg_factor", 3.0),
            num_keep_tokens=256,
            num_samples=1,
            generator=torch.Generator(device=device).manual_seed(seed or 0),
        )

    vae_sizes = None
    if hasattr(cfg, "decode") and cfg.decode is not None:
        vae_sizes = cfg.decode.get("vae_image_sizes")
    decode_kw = dict(
        timesteps=gen_cfg.get("timesteps", 25),
        guidance_scale=gen_cfg.get("tokenizer_cfg_factor", 7.5),
        perform_norm_guidance=gen_cfg.get("tokenizer_perform_norm_guidance", True),
    )
    if vae_sizes is not None:
        decode_kw["vae_image_sizes"] = vae_sizes

    partial_images = []
    for n_tok in token_counts:
        partial_ids = token_ids[:, :n_tok]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            result = tokenizer.decode(
                data_dict={"image_token_ids": partial_ids},
                verbose=False,
                **decode_kw,
            )
        img = tensor_to_pil(result["decoded_images"][0])
        partial_images.append(img)
        if verbose:
            print(f"  {n_tok:>3} tokens decoded")

    return token_ids, partial_images


def unwrap_model(model: nn.Module) -> nn.Module:
    """Unwrap DDP/FSDP model to get base model.
    
    Args:
        model: Potentially wrapped model.
    
    Returns:
        Unwrapped base model.
    """
    if isinstance(model, nn.parallel.DistributedDataParallel) or hasattr(model, "module"):
        return model.module
    return model


def set_generation_mode(model: Callable, mode: bool = True) -> Callable:
    """Set model to generation mode.
    
    This recursively sets the generation_mode attribute on all ReadWriteBlock
    modules in the model.
    
    Args:
        model: Model to set to generation mode.
        mode: Whether to enable generation mode.
    
    Returns:
        Model (potentially unwrapped).
    """
    unwrapped_model = unwrap_model(model)
    # Call generation() method on root model, which recursively sets generation_mode
    # on all ReadWriteBlock children (matching l3m behavior)
    if hasattr(unwrapped_model, 'generation'):
        unwrapped_model.generation(mode)
    return unwrapped_model


def load_model(
    model_id: str,
    device: str = "cuda",
) -> Tuple[nn.Module, Optional[nn.Module], OmegaConf]:
    """Load FlexTok AR model from HuggingFace Hub or local directory.

    Args:
        model_id: HuggingFace model ID (e.g., "EPFL-VILAB/FlexAR-3B-T2I") or
                 local directory path containing model files. If a local path is provided,
                 it should contain `model.safetensors` and `config.yaml`.
        device: Device to load model on.

    Returns:
        Tuple of (wrapped_model, tokenizer, config):
            - wrapped_model: ImageGenerationWrapper around the model
            - tokenizer: FlexTok image tokenizer (ImageResamplerTokenizer)
            - config: Full configuration

    Note:
        All HuggingFace checkpoints have muP scaling baked into the weights,
        so no runtime muP instantiation is needed.
    """
    # Import here to avoid circular dependencies
    from flextok_ar.model.generation import ImageGenerationWrapper
    
    # Check if model_id is a local path or HuggingFace model ID
    model_path = Path(model_id)
    if model_path.exists() and model_path.is_dir():
        # Local directory - use it directly
        logger.info(f"Loading model from local directory: {model_id}...")
        repo_path = model_id
    else:
        # HuggingFace model ID - download or use cache
        logger.info(f"Loading model from HuggingFace Hub: {model_id}...")
        logger.info("Downloading model repository (or using cache)...")
        repo_path = snapshot_download(
            repo_id=model_id,
            repo_type="model",
            local_files_only=False,  # Allow downloading if not cached
        )
    
    # Load config.yaml from HuggingFace repository
    config_yaml_path = Path(repo_path) / "config.yaml"
    if not config_yaml_path.exists():
        raise FileNotFoundError(
            f"Config file not found in HuggingFace repository: {config_yaml_path}\n"
            f"Expected file: config.yaml in {model_id}\n"
            f"Please ensure config.yaml is uploaded to the repository."
        )
    logger.info("Loading config from HuggingFace repository...")
    cfg = OmegaConf.load(str(config_yaml_path))
    
    # Load model weights from safetensors file
    model_safetensors_path = Path(repo_path) / "model.safetensors"
    if not model_safetensors_path.exists():
        raise FileNotFoundError(
            f"Model safetensors file not found in repository: {model_safetensors_path}\n"
            f"Expected file: model.safetensors in {model_id}"
        )
    
    logger.info(f"Loading model weights from {model_safetensors_path}...")

    # Instantiate model (muP scaling already baked into checkpoint weights)
    logger.info("Building model architecture...")
    model = instantiate(cfg.model.meta_model)

    # Load state dict from safetensors
    state_dict = load_file(str(model_safetensors_path))
    
    # For 2D grid models, the AR checkpoint may not contain image_tokenizer weights.
    # Filter them out before loading to avoid mismatches.
    remove_image_tokenizer = cfg.model.get("remove_image_tokenizer", False)
    if remove_image_tokenizer:
        state_dict = {k: v for k, v in state_dict.items() if "image_tokenizer" not in k}
        logger.info("Filtered out image_tokenizer keys from checkpoint (2D grid mode)")
    
    # Load state dict
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    
    # Filter out expected mismatches:
    # - image_tokenizer loads its own weights from HuggingFace
    # - text_encoder.t5_embedder (the pretrained T5) loads its own weights from HuggingFace
    # IMPORTANT: we do NOT filter out text_encoder.mlp.* here, since those are learned
    # projection weights that should come from the AR checkpoint. If they are missing,
    # we want to see that warning.
    missing_keys = [
        k
        for k in missing_keys
        if "image_tokenizer" not in k and "t5_embedder" not in k
    ]
    unexpected_keys = [
        k
        for k in unexpected_keys
        if "image_tokenizer" not in k and "t5_embedder" not in k
    ]
    
    if missing_keys:
        logger.warning("Missing keys in checkpoint: %d keys", len(missing_keys))
    if unexpected_keys:
        logger.warning("Unexpected keys in checkpoint: %d keys", len(unexpected_keys))
    
    # Move to device and set eval mode
    model = model.to(device)
    model.eval()
    
    # Set generation mode
    set_generation_mode(model, mode=True)
    
    # Get tokenizer (image tokenizer) - FlexTok ImageResamplerTokenizer
    tokenizer = None
    if hasattr(model, 'models') and 'image_tokenizer' in model.models:
        tokenizer = model.models['image_tokenizer']
    
    # Wrap in ImageGenerationWrapper
    model_type = cfg.generation.get("model_type", "ar_text_to_image_model")
    wrapped_model = ImageGenerationWrapper(
        model=model,
        eos_token_id=0,
        pad_token_id=0,
        modality="image",
        model_type=model_type,
    )
    wrapped_model = wrapped_model.to(device)
    wrapped_model.eval()
    
    logger.info("Model loaded successfully on %s", device)

    return wrapped_model, tokenizer, cfg


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert an image tensor to a PIL Image.

    Args:
        tensor: Image tensor, shape [C, H, W] or [1, C, H, W], values in [-1, 1].

    Returns:
        PIL Image in RGB format.
    """
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = (tensor + 1.0) / 2.0
    tensor = torch.clamp(tensor, 0, 1)
    array = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return Image.fromarray(array)


def generate_t2i(
    model,
    prompt: str,
    cfg: OmegaConf,
    num_samples: int = 1,
    cfg_factor: Optional[float] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> List[torch.Tensor]:
    """Generate images from a text prompt.

    Args:
        model: ImageGenerationWrapper returned by :func:`load_model`.
        prompt: Text prompt describing the desired image.
        cfg: Model configuration (OmegaConf) returned by :func:`load_model`.
        num_samples: Number of images to generate.
        cfg_factor: Classifier-free guidance scale.  Overrides the value in
            ``cfg`` when provided.
        temperature: Sampling temperature.  Overrides the value in ``cfg``
            when provided.
        seed: Random seed for reproducibility.
        device: Device string (e.g. ``"cuda"`` or ``"cuda:0"``).
        verbose: Whether to print generation progress.

    Returns:
        List of image tensors, each of shape [C, H, W] in [-1, 1].
        Use :func:`tensor_to_pil` to convert to PIL images.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    data_dict = {"text": [prompt]}
    gen_cfg = cfg.generation
    kwargs = {
        "sample": gen_cfg.get("sample", True),
        "temperature": temperature if temperature is not None else gen_cfg.get("temperature", 1.0),
        "top_k": gen_cfg.get("top_k", 0),
        "top_p": gen_cfg.get("top_p", 0.0),
        "cfg_factor": cfg_factor if cfg_factor is not None else gen_cfg.get("cfg_factor", 3.0),
        "num_keep_tokens": gen_cfg.get("num_keep_tokens", 256),
        "num_samples": num_samples,
        "timesteps": gen_cfg.get("timesteps", 25),
        "tokenizer_cfg_factor": gen_cfg.get("tokenizer_cfg_factor", 7.5),
        "tokenizer_perform_norm_guidance": gen_cfg.get("tokenizer_perform_norm_guidance", True),
    }
    if hasattr(cfg, "decode") and cfg.decode is not None:
        vae_image_sizes = cfg.decode.get("vae_image_sizes", None)
        if vae_image_sizes is not None:
            kwargs["vae_image_sizes"] = vae_image_sizes

    if verbose:
        print(f"Generating {num_samples} image(s) for prompt: '{prompt}'...")
    images = model.generate(data_dict, **kwargs)
    return images


def generate_c2i(
    model,
    class_label: int,
    cfg: OmegaConf,
    num_samples: int = 1,
    cfg_factor: Optional[float] = None,
    temperature: Optional[float] = None,
    seed: Optional[int] = None,
    device: str = "cuda",
    verbose: bool = True,
) -> List[torch.Tensor]:
    """Generate images from an ImageNet class label (class-to-image).

    Args:
        model: ImageGenerationWrapper returned by :func:`load_model`.
        class_label: ImageNet class index in [0, 999].
        cfg: Model configuration (OmegaConf) returned by :func:`load_model`.
        num_samples: Number of images to generate.
        cfg_factor: Classifier-free guidance scale.  Overrides the value in
            ``cfg`` when provided.
        temperature: Sampling temperature.  Overrides the value in ``cfg``
            when provided.
        seed: Random seed for reproducibility.
        device: Device string (e.g. ``"cuda"`` or ``"cuda:0"``).
        verbose: Whether to print generation progress.

    Returns:
        List of image tensors, each of shape [C, H, W] in [-1, 1].
        Use :func:`tensor_to_pil` to convert to PIL images.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    data_dict = {"target": torch.tensor([class_label], dtype=torch.long, device=device)}
    gen_cfg = cfg.generation
    kwargs = {
        "sample": gen_cfg.get("sample", True),
        "temperature": temperature if temperature is not None else gen_cfg.get("temperature", 1.0),
        "top_k": gen_cfg.get("top_k", 0),
        "top_p": gen_cfg.get("top_p", 0.0),
        "cfg_factor": cfg_factor if cfg_factor is not None else gen_cfg.get("cfg_factor", 3.0),
        "num_keep_tokens": gen_cfg.get("num_keep_tokens", 256),
        "num_samples": num_samples,
        "timesteps": gen_cfg.get("timesteps", 25),
        "tokenizer_cfg_factor": gen_cfg.get("tokenizer_cfg_factor", 7.5),
        "tokenizer_perform_norm_guidance": gen_cfg.get("tokenizer_perform_norm_guidance", True),
    }
    if hasattr(cfg, "decode") and cfg.decode is not None:
        vae_image_sizes = cfg.decode.get("vae_image_sizes", None)
        if vae_image_sizes is not None:
            kwargs["vae_image_sizes"] = vae_image_sizes

    if verbose:
        print(f"Generating {num_samples} image(s) for class label: {class_label}...")
    images = model.generate(data_dict, **kwargs)
    return images
