"""Integration layer between FlexTok tokenizer and L3M framework.

This module provides ReadWriteBlock wrappers around public FlexTok models
to enable seamless integration with L3M's data_dict interface.
"""

import logging
from functools import partial
from typing import Any, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn

# Import from public packages
from l3m.model.meta_models import ReadWriteBlock
from flextok.flextok_wrapper import FlexTokFromHub
from flextok.model.utils.packed_ops import packed_call
from flextok.utils.checkpoint import ALLOWED_TARGETS

# Extend FlexTok's allowlist for targets used by 2D grid models
_EXTRA_TARGETS = ["flextok.model.utils.dict_ops.sum_tensors"]
for _t in _EXTRA_TARGETS:
    if _t not in ALLOWED_TARGETS:
        ALLOWED_TARGETS.append(_t)

logger = logging.getLogger("flextok_ar")

__all__ = ["ImageResamplerTokenizer"]


class ImageResamplerTokenizer(ReadWriteBlock):
    """Image resampler tokenizer using public FlexTok.
    
    Wraps FlexTok's tokenizer in L3M's ReadWriteBlock interface for
    seamless integration with L3M models.
    
    Args:
        checkpoint_path: Local path to FlexTok checkpoint (safetensors).
        model_id: HuggingFace model ID (e.g., "EPFL-VILAB/flextok_d18_d28_dfn").
        force_vae_encode: If recomputation of VAE latents is forced.
        sample_posterior: If VAE latent distribution should be sampled.
        image_dims: Image dimensions (H, W).
        read_key: Key to read images from data_dict.
        write_key: Key to write token IDs to data_dict.
        vae_latent_read_key: Optional key for VAE latents.
        del_decoders: Whether to delete decoders (for memory efficiency).
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        model_id: str = "EPFL-VILAB/flextok_d18_d28_dfn",
        force_vae_encode: bool = True,
        sample_posterior: bool = True,
        image_dims: Sequence[int] = (256, 256),
        read_key: str = "image",
        write_key: str = "image_token_ids",
        vae_latent_read_key: Optional[str] = None,
        del_decoders: bool = False,
        token_grid_size: Optional[Sequence[int]] = None,
        **kwargs,
    ):
        super().__init__(read_key=read_key, write_key=write_key, **kwargs)
        
        # Store configurable token grid size (default: 1D [256, 1])
        self._token_grid_size = list(token_grid_size) if token_grid_size is not None else [256, 1]
        
        # Load FlexTok model from public repository
        if checkpoint_path:
            # Load from local checkpoint
            from flextok.utils.checkpoint import load_safetensors
            from hydra.utils import instantiate
            ckpt, config = load_safetensors(checkpoint_path)
            self.flextok_model = instantiate(config).eval()
            self.flextok_model.load_state_dict(ckpt)
            logger.info(f"Loaded FlexTok from local checkpoint: {checkpoint_path}")
        else:
            # Load from HuggingFace Hub or local directory
            self.flextok_model = FlexTokFromHub.from_pretrained(model_id).eval()
            logger.info(f"Loaded FlexTok from: {model_id}")
        
        # Optionally delete decoders for memory efficiency
        if del_decoders and hasattr(self.flextok_model, 'decoder'):
            del self.flextok_model.decoder
            if hasattr(self.flextok_model, 'vae') and hasattr(self.flextok_model.vae, 'decoder'):
                del self.flextok_model.vae.decoder
        
        self.force_vae_encode = force_vae_encode
        self.sample_posterior = sample_posterior
        self.image_dims = image_dims
        self.vae_latent_read_key = vae_latent_read_key
        
        # Get codebook size from FlexTok model
        if hasattr(self.flextok_model, 'regularizer'):
            self.codebook_size = self.flextok_model.regularizer.codebook_size
        else:
            self.codebook_size = 64000  # Default FSQ codebook size
        
        # Detect if this is a 2D grid model (no enc_register_module)
        self._is_2d_grid = not hasattr(self.flextok_model.encoder, 'module_dict') or \
            "enc_register_module" not in self.flextok_model.encoder.module_dict
        if self._is_2d_grid:
            logger.info("Detected 2D grid FlexTok model (no enc_register_module)")
        
        # Freeze the tokenizer
        self.freeze()
    
    def freeze(self) -> "ImageResamplerTokenizer":
        """Freeze all parameters."""
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self
    
    @property
    def token_grid_size(self) -> Sequence[int]:
        """Token grid size. For 1D tokens: [256, 1]. For 2D grid: [16, 16]."""
        return self._token_grid_size
    
    @property
    def num_tokens(self) -> int:
        """Number of tokens."""
        return np.prod(self.token_grid_size)
    
    @torch.no_grad()
    def forward(self, data_dict: dict[str, Any]) -> dict[str, Any]:
        """Tokenize images using FlexTok.
        
        Args:
            data_dict: Input containing images at self.read_key.
                Images should be shape [B, C, H, W], normalized to [-1, 1].
        
        Returns:
            data_dict: Output with token IDs at self.write_key, shape [B, L].
        """
        images = data_dict[self.read_key]
        
        # Use FlexTok's tokenize method
        token_ids_list = self.flextok_model.tokenize(images)
        
        # Convert list of tensors to single tensor
        token_ids = torch.cat(token_ids_list, dim=0)
        
        # Flatten if needed (for 2D tokens)
        if token_ids.ndim > 2:
            token_ids = token_ids.flatten(1)
        
        data_dict[self.write_key] = token_ids
        return data_dict
    
    def _prepare_data_dict_for_2d_grid(
        self,
        token_ids_list: list[torch.Tensor],
    ) -> dict[str, Any]:
        """Prepare data_dict for 2D grid FlexTok models that lack enc_register_module.
        
        Handles both full-length (256) and partial token sequences (e.g., during
        beam search where tokens grow incrementally). Partial sequences are
        zero-padded to the full grid size before reshaping to 2D.
        
        Important: The decoder expects embeddings in 2D spatial format [B, H, W, D],
        not a flat [B, N, D] sequence, because `dec_add_noise_and_latents` sums them
        element-wise with the 2D-patched VAE noise latents.
        """
        grid_h, grid_w = self._token_grid_size  # e.g. [16, 16]
        full_seq_len = grid_h * grid_w  # 256
        
        # Record original token lengths before padding
        token_ids_lens = [t.shape[1] for t in token_ids_list]
        
        # Pad partial token sequences to full grid size (zero-pad)
        padded_token_ids_list = []
        for t in token_ids_list:
            if t.shape[1] < full_seq_len:
                pad = torch.zeros(
                    (t.shape[0], full_seq_len - t.shape[1]),
                    device=t.device, dtype=t.dtype,
                )
                t = torch.cat([t, pad], dim=1)
            padded_token_ids_list.append(t)
        
        # Convert token IDs to quantized embeddings
        # Input: list of [1, 256] → Output: list of [1, 256, 6]
        fsq_decode_fn = partial(self.flextok_model.regularizer.indices_to_embedding)
        quant_list = packed_call(fsq_decode_fn, padded_token_ids_list)
        
        # Reshape flat embeddings [1, 256, D] → 2D spatial [1, 16, 16, D]
        # to match the decoder's expected spatial format (noised patches are [1, 16, 16, 1792])
        quant_list = [q.reshape(q.shape[0], grid_h, grid_w, q.shape[-1]) for q in quant_list]
        
        # Build the data dict with the quantized embeddings
        data_dict = {self.flextok_model.quants_write_key: quant_list}
        
        # If the decoder has a latent dropout module, set its eval_keep_k_read_key
        if hasattr(self.flextok_model.decoder, 'module_dict') and \
                "dec_nested_dropout" in self.flextok_model.decoder.module_dict:
            eval_keep_k_read_key = self.flextok_model.decoder.module_dict["dec_nested_dropout"].eval_keep_k_read_key
            data_dict[eval_keep_k_read_key] = token_ids_lens
        
        return data_dict

    @torch.no_grad()
    def decode(
        self,
        data_dict: dict[str, Any],
        timesteps: int = 20,
        generator: Optional[torch.Generator] = None,
        guidance_scale: float = 7.5,
        perform_norm_guidance: bool = True,
        num_keep_tokens: Optional[int] = None,
        vae_image_sizes: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ) -> dict[str, Any]:
        """Decode token IDs back to images using FlexTok.
        
        Args:
            data_dict: Input containing token IDs at self.write_key.
            timesteps: Number of denoising steps for flow matching.
            generator: RNG generator for decoding.
            guidance_scale: Classifier-free guidance scale.
            perform_norm_guidance: Whether to use normalized guidance.
            num_keep_tokens: Optional number of tokens to keep from sequence.
            vae_image_sizes: VAE latent grid size (e.g. 32 for 256px images with f8 VAE).
                Required for 2D grid models. If None, uses FlexTok's default.
        
        Returns:
            data_dict: Output with decoded images.
        """
        token_ids = data_dict[self.write_key]
        
        # Convert to list of tensors (FlexTok expects list)
        token_ids_list = [token_ids[i:i+1] for i in range(len(token_ids))]
        
        # Optionally truncate tokens
        if num_keep_tokens is not None:
            token_ids_list = [t[:, :num_keep_tokens] for t in token_ids_list]
        
        decode_kwargs = dict(
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            perform_norm_guidance=perform_norm_guidance,
            verbose=verbose,
        )
        if vae_image_sizes is not None:
            decode_kwargs["vae_image_sizes"] = vae_image_sizes
        
        if self._is_2d_grid:
            # For 2D grid models, bypass _prepare_data_dict_for_detokenization
            # which assumes enc_register_module and dec_nested_dropout exist
            flextok_data_dict = self._prepare_data_dict_for_2d_grid(token_ids_list)
            flextok_data_dict = self.flextok_model.decode(
                data_dict=flextok_data_dict,
                generator=generator,
                **decode_kwargs,
            )
            decoded_images_list = flextok_data_dict[self.flextok_model.image_write_key]
            decoded_images = torch.cat(decoded_images_list, dim=0)
        else:
            # For 1D models, use the standard detokenize path
            decoded_images = self.flextok_model.detokenize(
                token_ids_list,
                generator=generator,
                **decode_kwargs,
            )
        
        # Store decoded images
        data_dict["decoded_images"] = decoded_images
        
        return data_dict

