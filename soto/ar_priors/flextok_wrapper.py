"""FlexTok AR prior wrapper - integrates flextok-ar package.

Supports multiple FlexTok model variants:
  - flextok:       1D FlexTok d36 (default)
  - flextok_d12:   1D FlexTok d12
  - flextok_d18:   1D FlexTok d18
  - flextok_d26:   1D FlexTok d26
  - gridtok:       2D Grid FlexTok d36

Each variant uses the same FlexTokARPrior class with different configs.
"""

import torch
import logging
from typing import Dict, List, Union, Tuple, Optional
from pathlib import Path

from soto.ar_priors.base import BaseARPrior, ARPriorFactory

# Default config directory: soto/configs/components/ar_priors/
_DEFAULT_CONFIG_DIR = Path(__file__).parent.parent / "configs" / "components" / "ar_priors"

logger = logging.getLogger("soto.flextok")

# Import from flextok-ar package (needs to be installed)
try:
    from flextok_ar.utils.helpers import load_model
    FLEXTOK_AR_AVAILABLE = True
except ImportError:
    FLEXTOK_AR_AVAILABLE = False

from flextok.utils.demo import batch_to_pil

__all__ = ["FlexTokARPrior", "GridTokARPrior"]


@ARPriorFactory.register("flextok")
class FlexTokARPrior(BaseARPrior):
    """Wrapper for FlexTok AR model.
    
    Integrates the flextok-ar package into the SoTo framework.
    
    Config has TWO separate parameter sets:
    
    1. generation_kwargs: AR model params (passed to generate_ids)
       - sample, temperature, cfg_factor, top_k, top_p
       - num_keep_tokens, replacement, return_probs
    
    2. decode_kwargs: FlexTok decoder params (passed to FlexTok decode)
       - timesteps: Flow matching denoising steps (default: 20)
       - guidance_scale: FlexTok decoder CFG scale (default: 7.5)
       - perform_norm_guidance: APG for decoder (default: true)
       - use_same_noise_per_prompt: Same noise across beam candidates (default: true)
    
    3. use_uncond_gen: Unconditional generation mode (default: false)
       When true, replaces the text prompt with the model's unconditional
       token (text encoder singleton value) and disables classifier-free
       guidance during AR generation.  The verifier still sees the original
       text prompt — only the AR prior is unaware of it.
    """
    
    def __init__(self, config: Dict, device: str = "cuda"):
        if not FLEXTOK_AR_AVAILABLE:
            raise ImportError("flextok-ar package not found. Install the default SoTo dependencies with `pip install -e .`.")
        
        super().__init__(config, device)
        
        # Determine model_id for loading.
        # Priority: 1) model_id from config dict, 2) checkpoint_path (local path),
        #           3) config_path YAML  — supports two formats:
        #              a) SoTo ar_prior YAML  →  ar_prior.model_id  (current format)
        #              b) legacy flextok cfg →  model.checkpoint
        #           When a SoTo YAML is provided, its ar_prior.* values are merged
        #           into config so generation_kwargs / decode_kwargs etc. are also
        #           picked up automatically. Explicit config dict values take priority.
        #           4) auto-discover: if none of the above, look for
        #              {_DEFAULT_CONFIG_DIR}/{name}.yaml using the registered name.
        model_id = config.get("model_id", None)

        if model_id is None:
            checkpoint_path = config.get("checkpoint_path", None)
            if checkpoint_path:
                model_id = checkpoint_path
            else:
                # Auto-discover config path from registered name if not given explicitly
                config_path = config.get("config_path", None)
                if config_path is None:
                    name = config.get("name", "")
                    # Search flat and one-level subdirs (flextok/, gridtok/, etc.)
                    candidates = [_DEFAULT_CONFIG_DIR / f"{name}.yaml"] + [
                        d / f"{name}.yaml"
                        for d in sorted(_DEFAULT_CONFIG_DIR.iterdir())
                        if d.is_dir()
                    ]
                    auto_path = next((p for p in candidates if p.exists()), None)
                    if auto_path:
                        config_path = str(auto_path)
                        logger.info(f"Auto-discovered config: {config_path}")
                    else:
                        raise ValueError(
                            f"No model source found for '{name}'. Provide model_id, "
                            f"checkpoint_path, or config_path, or add a YAML under {_DEFAULT_CONFIG_DIR}"
                        )
                from omegaconf import OmegaConf
                cfg_temp = OmegaConf.load(config_path)
                if OmegaConf.select(cfg_temp, "ar_prior") is not None:
                    # SoTo ar_prior YAML format: merge ar_prior.* then overlay explicit config
                    yaml_config = OmegaConf.to_container(cfg_temp.ar_prior, resolve=True)
                    config = {**yaml_config, **config}
                    model_id = config.get("model_id")
                else:
                    # Legacy format: model.checkpoint holds the HF model ID
                    model_id = OmegaConf.select(cfg_temp, "model.checkpoint")
                if not model_id:
                    raise ValueError(
                        f"Could not determine model_id. Provide model_id / checkpoint_path "
                        f"in the config dict, or ar_prior.model_id in {config_path}"
                    )
        
        logger.info(f"[Device {device}] Loading FlexTok AR model: {model_id}")
        
        # Use the load_model helper function
        # load_model returns (wrapped_model, tokenizer, cfg)
        self.generator, tokenizer, self.cfg = load_model(
            model_id=model_id,
            device=device
        )
        
        # Extract AR generation kwargs (for generate_ids)
        # Base params come from the model config (cfg.generation), just like the old code:
        #   gen_kwargs = dict(getattr(self.ar_cfg.constants, "gen_kwargs", {}))
        # Search-specific overrides (replacement, return_probs, etc.) come from SoTo config.
        base_gen_kwargs = dict(self.cfg.generation)
        base_gen_kwargs.pop("model_type", None)  # not a generate_ids param
        
        # Override with search-specific params from config
        search_overrides = config.get("generation_kwargs", {})
        base_gen_kwargs.update(search_overrides)
        self.generation_kwargs = base_gen_kwargs
        
        self.max_tokens_cfg = self.generation_kwargs.get("num_keep_tokens", 256)
        
        # Unconditional generation mode
        self.use_uncond_gen = config.get("use_uncond_gen", False)
        if self.use_uncond_gen:
            # Cache the unconditional prompt (text encoder singleton value)
            # This is used in place of the real prompt during AR generation.
            self._uncond_prompt = self.generator.get_text_encoder().singleton_value
            logger.info("Unconditional generation mode enabled (no text conditioning for AR)")
        
        # Extract FlexTok decode kwargs (for decode / flow matching pipeline)
        self.decode_kwargs = config.get("decode_kwargs", {})
        
        # VAE image sizes for decoding (needed for 2D grid tokenizer)
        # Read from flextok_ar config's decode section, or override from SOT config
        decode_cfg = self.cfg.get("decode", {})
        self.vae_image_sizes = config.get("vae_image_sizes", 
                                           decode_cfg.get("vae_image_sizes", 32))
        
        # Get the underlying FlexTok model for direct detokenization
        # This lets us inject noises into data_dict without modifying the flextok package
        # Use the tokenizer returned from load_model (FlexTok ImageResamplerTokenizer)
        if tokenizer is None:
            # Fallback: get from wrapped model if not returned
            tokenizer = self.generator.get_image_tokenizer()
        self._flextok_model = tokenizer.flextok_model
        self._image_tokenizer = tokenizer  # Keep reference for 2D grid decode
        
        # Detect if this is a 2D grid model (no enc_register_module)
        self._is_2d_grid = getattr(tokenizer, '_is_2d_grid', False)
        if self._is_2d_grid:
            logger.info("Detected 2D grid FlexTok model for SOT decode")
        
        # Set the noise read key for consistent noise injection
        self._flextok_model.pipeline.noise_read_key = "fixed_noise"
        
        # Noise control (internal)
        self.noise_cache = None
        self.noise_caption_idx = -1
    
    def generate_next_tokens(
        self,
        prompt: Union[str, List[str]],
        current_tokens: torch.Tensor,
        num_new_tokens: int = 1,
        num_samples: int = 1,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generate next tokens with AR model top-k sampling.
        
        Only AR-related params (from self.generation_kwargs) are sent to generate_ids.
        FlexTok decoder params (self.decode_kwargs) are NOT mixed in here.
        
        Args:
            prompt: Text prompt(s)
            current_tokens: Current tokens, shape [batch, seq_len]
            num_new_tokens: Number of new tokens to generate
            num_samples: Branching factor (candidates per beam)
            **kwargs: Additional AR params to override (seed, replacement, etc.)
        
        Returns:
            (new_tokens, token_probs): 
                - new_tokens: [batch*num_samples, num_new_tokens]
                - token_probs: [batch*num_samples] log probabilities or None
        """
        # Prepare input — unconditional mode replaces the real prompt
        if self.use_uncond_gen:
            ar_prompt = self._uncond_prompt  # text-encoder singleton
        else:
            if isinstance(prompt, str):
                ar_prompt = [prompt]
            else:
                ar_prompt = prompt
        
        # Prepare data dict with current tokens
        batch_size = current_tokens.size(0)
        data_dict = {
            "text": ar_prompt * batch_size if isinstance(ar_prompt, list) else [ar_prompt] * batch_size,
            "pred_image_token_ids": current_tokens
        }
        
        # Merge AR generation kwargs only (no decode kwargs mixed in)
        gen_params = {**self.generation_kwargs, **kwargs}
        gen_params["num_keep_tokens"] = current_tokens.size(1) + num_new_tokens
        gen_params["num_samples"] = num_samples
        
        # Unconditional mode: disable classifier-free guidance for AR
        if self.use_uncond_gen:
            gen_params["cfg_factor"] = None
        
        gen_params.pop("seed", None)  # seed is set on the prior via set_seed(), not per-call
        gen_params["generator"] = self.rng
        
        # Generate using l3m AR model
        # The old code wrapped generate_ids in torch.cuda.amp.autocast(enabled=True, dtype=torch.float32)
        # which forces float32 computation and affects SDPA backend selection.
        # Without it, different attention backends may be chosen, producing different
        # floating-point results that accumulate through the autoregressive loop.
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            result = self.generator.generate_ids(data_dict, **gen_params)
            
            # Handle return value (with or without probs)
            if isinstance(result, tuple):
                all_tokens, token_probs = result
            else:
                all_tokens = result
                token_probs = None
        
        return all_tokens, token_probs
    
    def decode_tokens(self, tokens: torch.Tensor, **kwargs) -> List:
        """Decode tokens to images using FlexTok flow-matching pipeline.
        
        Only FlexTok decoder params (from self.decode_kwargs) are used here.
        AR generation params (self.generation_kwargs) are NOT mixed in.
        
        Handles noise control internally based on decode_kwargs:
        - use_same_noise_per_prompt: Use same noise across batch for consistency
        - Different noise per caption (seeded by caption index)
        
        Note: The public FlexTok's detokenize() does NOT accept a noises parameter,
        so we replicate its two-step logic manually and inject noise into data_dict
        between _prepare_data_dict_for_detokenization() and decode().
        The pipeline's noise_read_key mechanism IS in the public code — we just
        need to set the key and put our noise in the data_dict at that key.
        
        Args:
            tokens: Token sequences, shape [batch, seq_len]
            **kwargs: Override decoding parameters:
                - timesteps: Flow matching denoising steps (default: from decode_kwargs)
                - guidance_scale: FlexTok decoder CFG scale (default: from decode_kwargs)
                - caption_idx: Caption index for noise caching (optional)
        
        Returns:
            List of PIL Images
        """
        
        # Allow per-call override via kwargs, falling back to the config default
        use_same_noise_per_prompt = kwargs.get(
            "use_same_noise_per_prompt",
            self.decode_kwargs.get("use_same_noise_per_prompt", True)
        )
        
        # Convert tokens to list of tensors (FlexTok expects list of [1, seq_len] tensors)
        token_ids_list = [tokens[i:i+1] for i in range(len(tokens))]
        
        # Step 1: Build data_dict with quantized embeddings
        if self._is_2d_grid:
            # For 2D grid models, use the integration layer's preparation method
            # which bypasses the missing enc_register_module / dec_nested_dropout
            data_dict = self._image_tokenizer._prepare_data_dict_for_2d_grid(token_ids_list)
        else:
            # For 1D models, use the standard FlexTok API
            data_dict = self._flextok_model._prepare_data_dict_for_detokenization(
                token_ids_list=token_ids_list
            )
        
        # Step 2: Inject fixed noise into data_dict so the pipeline reads it.
        # NOTE: The public FlexTok pipeline checks `self.noise_read_key is not None`
        # (not the data_dict value), so we must set/unset the key dynamically.
        batch_size = tokens.size(0)
        if use_same_noise_per_prompt:
            caption_idx = kwargs.get("caption_idx", 0)
            noise = self.get_noise_per_prompt(caption_idx)  # (1, 16, 32, 32)
            data_dict["fixed_noise"] = [noise.clone() for _ in range(batch_size)]
        else:
            # Generate independent random noise for each sample so diverse token
            # sequences also produce visually diverse images.  Relying on
            # noise_read_key=None (pipeline-internal noise) risks broadcasting a
            # single noise tensor across the whole batch.
            data_dict["fixed_noise"] = [
                torch.randn((1, 16, 32, 32), device=self.device)
                for _ in range(batch_size)
            ]
        self._flextok_model.pipeline.noise_read_key = "fixed_noise"
        
        # Step 3: Run decode (pipeline → VAE) with the noise-augmented data_dict
        # vae_image_sizes: 32 for 256x256 images with f8 VAE (same as detokenize)
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            data_dict = self._flextok_model.decode(
                data_dict=data_dict,
                vae_image_sizes=self.vae_image_sizes,
                timesteps=kwargs.get("timesteps", self.decode_kwargs.get("timesteps", 20)),
                guidance_scale=kwargs.get("guidance_scale", self.decode_kwargs.get("guidance_scale", 7.5)),
                perform_norm_guidance=self.decode_kwargs.get("perform_norm_guidance", True),
                verbose=False,
            )
        
        # Step 4: Extract decoded images and convert to PIL
        decoded_images_list = data_dict[self._flextok_model.image_write_key]
        decoded_images = torch.cat(decoded_images_list, dim=0)
        pil_images = [batch_to_pil(image.unsqueeze(0)) for image in decoded_images]
        
        return pil_images

    def get_noise_per_prompt(self, caption_idx: int) -> torch.Tensor:
        """Get noise per prompt for a given caption index.
        Args:
            caption_idx: Caption index
        
        Returns:
            Flow decoder noise
        """

        if caption_idx != self.noise_caption_idx or self.noise_cache is None:
            generator = torch.Generator(device=self.device).manual_seed(caption_idx)
            self.noise_cache = torch.randn(
                (1, 16, 32, 32),  # FlexTok latent shape
                generator=generator,
                device=self.device
            )
            self.noise_caption_idx = caption_idx

        return self.noise_cache
    
    def get_vocab_size(self) -> int:
        return self.generator.get_vocab_size()
    
    def get_max_tokens(self) -> int:
        return self.max_tokens_cfg


# Register additional FlexTok model size variants.
# Each uses the same FlexTokARPrior class — the model_id in the SOT YAML
# controls which model size/checkpoint is loaded.

@ARPriorFactory.register("flextok_ar_3b")
class FlexTokARPrior3B(FlexTokARPrior):
    """FlexTok AR 3B (1D, 36 blocks, 2304 embed_dim, 3.06B params)."""
    pass


@ARPriorFactory.register("flextok_ar_1b")
class FlexTokARPrior1B(FlexTokARPrior):
    """FlexTok AR 1B (1D, 26 blocks, 1664 embed_dim, 1.15B params)."""
    pass


@ARPriorFactory.register("flextok_ar_382m")
class FlexTokARPrior382M(FlexTokARPrior):
    """FlexTok AR 382M (1D, 18 blocks, 1152 embed_dim, 382M params)."""
    pass


@ARPriorFactory.register("flextok_ar_113m")
class FlexTokARPrior113M(FlexTokARPrior):
    """FlexTok AR 113M (1D, 12 blocks, 768 embed_dim, 113M params)."""
    pass


@ARPriorFactory.register("gridtok_ar_3b")
class GridTokARPrior3B(FlexTokARPrior):
    """GridTok AR 3B (2D grid, 16x16 tokens, 36 blocks, 2304 embed_dim, 3.06B params).

    Uses a 2D grid FlexTok tokenizer instead of the 1D tokenizer.
    The AR model architecture is identical to FlexTok AR 3B; only the tokenizer
    and token grid size differ.
    """
    pass

