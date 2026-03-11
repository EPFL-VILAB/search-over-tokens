"""Generation wrapper for autoregressive image models.

This module provides the ImageGenerationWrapper which handles the
autoregressive generation loop with advanced sampling strategies.
"""

import logging
from typing import Any, Callable, Literal, Optional
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger("flextok_ar")

__all__ = ["sample_with_top_k_top_p", "ImageGenerationWrapper"]


def sample_with_top_k_top_p(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    num_samples: int = 1,
    generator: Optional[torch.Generator] = None,
    replacement: bool = True,
    return_probs: bool = False,
) -> torch.Tensor:
    """Sample from logits with Top-K and Top-P filtering.
    
    Adapted from VAR: https://github.com/FoundationVision/VAR
    
    Args:
        logits: Prediction logits, shape [B, L, vocab_size].
        temperature: Temperature for sampling.
        top_k: Top-K cutoff (0 = no filtering).
        top_p: Top-P (nucleus) cutoff (0.0 = no filtering).
        num_samples: Number of samples per item.
        generator: Random number generator.
        replacement: Whether to sample with replacement.
        return_probs: Whether to return probabilities.
    
    Returns:
        Sampled token IDs, shape [B, L, num_samples].
        If return_probs, also returns log probabilities.
    """
    B, L, V = logits.shape
    logits = logits / temperature
    
    # Top-K filtering
    if top_k > 0:
        idx_to_remove = logits < logits.topk(top_k, largest=True, sorted=False, dim=-1)[0].amin(
            dim=-1, keepdim=True
        )
        logits.masked_fill_(idx_to_remove, -torch.inf)
    
    # Top-P (nucleus) filtering
    if top_p > 0:
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=False)
        sorted_idx_to_remove = sorted_logits.softmax(dim=-1).cumsum_(dim=-1) <= (1 - top_p)
        sorted_idx_to_remove[..., -1:] = False
        logits.masked_fill_(
            sorted_idx_to_remove.scatter(sorted_idx.ndim - 1, sorted_idx, sorted_idx_to_remove),
            -torch.inf,
        )
    
    # Sample
    num_samples = abs(num_samples)
    samples = torch.multinomial(
        logits.softmax(dim=-1).view(-1, V),
        num_samples=num_samples,
        replacement=replacement,
        generator=generator,
    ).view(B, L, num_samples)
    
    if return_probs:
        sample_probabilities = logits.softmax(dim=-1).gather(-1, samples)
        return samples, sample_probabilities
    else:
        return samples


ImageGenerationModelTypes = Literal[
    "ar_label_to_image_model",  # Label-to-image (class-conditional)
    "ar_text_to_image_model",  # Text-to-image
]


class ImageGenerationWrapper(nn.Module):
    """Wrapper for autoregressive image generation.
    
    Handles the generation loop with support for:
    - Top-k and top-p sampling
    - Classifier-free guidance (CFG)
    - Temperature scaling
    - Partial sequence continuation
    
    Args:
        model: The AR model (MetaModel with tokenizer and AR components).
        eos_token_id: End-of-sequence token ID.
        pad_token_id: Padding token ID.
        modality: Modality type ('image').
        read_key: Key for reading predicted tokens from data_dict.
        image_token_write_key: Key for writing image tokens.
        model_type: Type of AR model ('ar_label_to_image_model' or 'ar_text_to_image_model').
        **kwargs: Additional generation parameters.
    """
    
    def __init__(
        self,
        model: nn.Module,
        eos_token_id: int = 0,
        pad_token_id: int = 0,
        modality: str = "image",
        read_key: str = "pred_image_token_ids",
        image_token_write_key: str = "image_token_ids",
        model_type: ImageGenerationModelTypes = "ar_text_to_image_model",
        **kwargs,
    ):
        super().__init__()
        
        self.model = model
        self.pad_token_id = pad_token_id
        self.read_key = read_key
        self.image_token_write_key = image_token_write_key
        self.modality = modality
        self.model_type = model_type
        
        # Determine model component names based on type
        if model_type == "ar_label_to_image_model":
            self.image_tokenizer_name = "image_tokenizer"
            self.image_model_name = "ar_image_model"
            self.text_encoder_name = None
        elif model_type == "ar_text_to_image_model":
            self.image_tokenizer_name = "image_tokenizer"
            self.image_model_name = "ar_image_model"
            self.text_encoder_name = "text_encoder"
        else:
            raise ValueError(f"Unsupported model_type {model_type}.")
    
    def get_image_tokenizer(self) -> Callable:
        """Get the image tokenizer component."""
        return self.model.models[self.image_tokenizer_name]
    
    def get_image_model(self) -> Callable:
        """Get the AR image model component."""
        return self.model.models[self.image_model_name]
    
    def get_text_encoder(self) -> Optional[Callable]:
        """Get the text encoder component (if available)."""
        if self.text_encoder_name is None:
            return None
        return self.model.models[self.text_encoder_name]
    
    @torch.inference_mode()
    def generate(
        self,
        data_dict: dict[str, Any],
        **kwargs,
    ) -> torch.Tensor:
        """Generate images.

        Args:
            data_dict: Input data (text prompts or class labels).
            **kwargs: Generation parameters.

        Returns:
            Generated images, shape [B, C, H, W].
        """
        # Generate token IDs.
        # float32 autocast matches the sot package's generate_ids call and avoids
        # numerical issues with the AR transformer's attention operations.
        with torch.cuda.amp.autocast(enabled=True, dtype=torch.float32):
            generated_ids = self.generate_ids(data_dict, **kwargs)

        # Decode tokens to images.
        # bfloat16 autocast matches the sot package's decode_tokens call and is
        # required for FlexTok's compiled flex_attention backend to run correctly.
        image_tokenizer = self.get_image_tokenizer()

        decode_call_kwargs = dict(
            data_dict={self.image_token_write_key: generated_ids},
            timesteps=kwargs.get("timesteps", 20),
            generator=kwargs.get("generator", None),
            guidance_scale=kwargs.get("tokenizer_cfg_factor", 7.5),
            perform_norm_guidance=kwargs.get("tokenizer_perform_norm_guidance", True),
            num_keep_tokens=kwargs.get("num_keep_tokens", 256),
        )
        if "vae_image_sizes" in kwargs:
            decode_call_kwargs["vae_image_sizes"] = kwargs["vae_image_sizes"]
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            decoded_images = image_tokenizer.decode(**decode_call_kwargs)["decoded_images"]

        return decoded_images
    
    @torch.inference_mode()
    def generate_ids(
        self,
        data_dict: dict[str, Any],
        **kwargs,
    ) -> torch.Tensor:
        """Generate token IDs autoregressively.
        
        Args:
            data_dict: Input data (text or class labels).
            **kwargs: Generation parameters:
                - sample: Whether to sample (vs. greedy).
                - temperature: Sampling temperature.
                - top_k: Top-k filtering.
                - top_p: Top-p (nucleus) filtering.
                - cfg_factor: Classifier-free guidance factor.
                - num_keep_tokens: Max number of tokens to generate.
                - num_samples: Number of samples per prompt.
                - replacement: Sample with replacement.
                - generator: Random generator.
        
        Returns:
            Generated token IDs, shape [B * num_samples, L].
        """
        # Extract generation parameters
        sample = kwargs.get("sample", True)
        temperature = kwargs.get("temperature", 1.0)
        top_k = kwargs.get("top_k", 0)
        top_p = kwargs.get("top_p", 0.0)
        cfg_factor = kwargs.get("cfg_factor", None)
        num_keep_tokens = kwargs.get("num_keep_tokens", 256)
        num_samples = kwargs.get("num_samples", 1)
        replacement = kwargs.get("replacement", True)
        generator = kwargs.get("generator", None)
        return_probs = kwargs.get("return_probs", False)
        
        # Determine sequence length
        token_grid_size = [num_keep_tokens, 1]
        sequence_len = np.prod(token_grid_size)
        current_len = data_dict[self.read_key].shape[1] if self.read_key in data_dict else 0
        tokens_to_generate = sequence_len - current_len
        max_iter = min(num_keep_tokens, tokens_to_generate) if num_keep_tokens else tokens_to_generate
        pad_len = sequence_len - (current_len + max_iter)
        
        # Handle text-to-image or label-to-image
        if self.model_type == "ar_text_to_image_model":
            text_encoder = self.get_text_encoder()
            text_read_key = text_encoder.read_key
            text = data_dict[text_read_key]
            assert isinstance(text, list)
            batch_size = len(text)
            
            # Prepare for CFG if needed
            if cfg_factor is not None:
                singleton_value = text_encoder.singleton_value
                text += [singleton_value] * batch_size  # Unconditional
                data_dict[text_read_key] = text
            
            # Encode text
            data_dict = self.model(data_dict, model_name=self.text_encoder_name)
            xa_mask_read_key = text_encoder.text_embeddings_mask_write_key
            xa_mask = data_dict[xa_mask_read_key]
            device = xa_mask.device
            
        elif self.model_type == "ar_label_to_image_model":
            targets = data_dict["target"]
            batch_size = targets.shape[0]
            device = targets.device
            xa_mask = None
            xa_mask_read_key = None
            
            # Prepare for CFG if needed
            if cfg_factor is not None:
                image_model = self.get_image_model()
                singleton_value = image_model.preprocessor.class_ignore_idx
                targets = torch.cat(
                    (targets, torch.full_like(targets, fill_value=singleton_value)), dim=0
                )
                data_dict["target"] = targets
        else:
            raise ValueError(f"Unsupported model_type {self.model_type}.")
        
        effective_batch_size = batch_size if cfg_factor is None else (2 * batch_size)
        
        # Initialize output tokens
        if self.read_key not in data_dict:
            data_dict[self.read_key] = torch.zeros(
                (effective_batch_size, 0), dtype=torch.long, device=device
            )
        else:
            if cfg_factor is not None:
                data_dict[self.read_key] = data_dict[self.read_key].repeat(2, 1)
        
        # Initialize sequence probabilities for tracking
        seq_probs = torch.zeros((batch_size, 1), device=device)
        
        # Initialize input_ids from data_dict (needed if max_iter == 0)
        input_ids = data_dict[self.read_key]
        
        for i in range(max_iter):
            input_ids = data_dict[self.read_key]
            
            # Update cross-attention mask if needed
            if xa_mask is not None:
                data_dict[xa_mask_read_key] = xa_mask[:, : current_len + i + 1, :]
            
            # Forward pass through AR model
            data_dict = self.model(data_dict, model_name=self.image_model_name)
            
            # Get logits
            output_key = self.get_image_model().head.write_key
            logits = data_dict[output_key].float()  # [B, L, vocab_size]
            
            # Apply classifier-free guidance
            if cfg_factor is not None:
                l_cond = logits[:batch_size]
                l_uncond = logits[batch_size:]
                logits = l_uncond + cfg_factor * (l_cond - l_uncond)

            # Sample next token
            if sample:
                last_token_logits = logits[:, -1].unsqueeze(1)  # [B, 1, vocab]
                
                # Sample - only sample multiple at the last step
                if return_probs:
                    last_token_id, last_token_probs = sample_with_top_k_top_p(
                        logits=last_token_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        generator=generator,
                        num_samples=num_samples if i == max_iter - 1 else 1,
                        replacement=replacement,
                        return_probs=True,
                    )
                    
                    # Track probabilities
                    last_token_probs = torch.log(last_token_probs)
                    seq_probs = seq_probs[:, :, None] + last_token_probs
                    seq_probs = seq_probs.permute(0, 2, 1).reshape((-1, 1))
                else:
                    last_token_id = sample_with_top_k_top_p(
                        logits=last_token_logits,
                        temperature=temperature,
                        top_k=top_k,
                        top_p=top_p,
                        generator=generator,
                        num_samples=num_samples if i == max_iter - 1 else 1,
                        replacement=replacement,
                        return_probs=False,
                    )
                
                # Reshape: [B * num_samples, 1]
                last_token_id = last_token_id.permute(0, 2, 1).reshape(-1, 1)
            else:
                # Deterministic top-k: at the last step return the top-num_samples
                # most probable tokens as distinct candidates; all other steps
                # take the single best token (k=1 → argmax).
                k = num_samples if i == max_iter - 1 else 1
                last_token_logits = logits[:, -1]  # [B, vocab]
                top_indices = last_token_logits.topk(k, dim=-1).indices  # [B, k]
                last_token_id = top_indices.reshape(-1, 1)  # [B*k, 1]

                if return_probs:
                    log_probs = last_token_logits.log_softmax(dim=-1)  # [B, vocab]
                    top_log_probs = log_probs.gather(-1, top_indices)  # [B, k]
                    if i == max_iter - 1:
                        seq_probs = (seq_probs + top_log_probs).reshape(-1, 1)  # [B*k, 1]
                    else:
                        seq_probs = seq_probs + top_log_probs  # [B, 1] (k=1)

            # For CFG, repeat the token for both conditional and unconditional
            if cfg_factor is not None:
                last_token_id = last_token_id.repeat(2, 1)

            # Expand input_ids at the last iteration to match sample count
            if i == max_iter - 1:
                input_ids = input_ids.repeat_interleave(num_samples, dim=0)
            
            # Append new token
            input_ids = torch.cat([input_ids, last_token_id], dim=1)
            data_dict[self.read_key] = input_ids
        
        # If no iterations ran, expand input_ids by num_samples before padding
        if max_iter == 0 and num_samples > 1:
            input_ids = input_ids.repeat_interleave(num_samples, dim=0)
        
        # Extend to the full sequence length with padding
        input_ids = torch.cat(
            (
                input_ids,
                torch.full(
                    size=(effective_batch_size * num_samples, pad_len),
                    fill_value=self.pad_token_id,
                    device=device,
                ),
            ),
            dim=1,
        )
        
        # Return only conditional samples (remove unconditional if CFG was used)
        output_ids = input_ids if not cfg_factor else input_ids[: batch_size * num_samples]
        
        if return_probs:
            return output_ids, seq_probs
        
        return output_ids

