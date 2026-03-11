"""Text encoder components for T2I generation.

Adapted from private l3m-ar to work with public L3M.
"""

import logging
from typing import Any, Optional
import torch
import torch.nn as nn

from l3m.model.layers.ffn import MLP

logger = logging.getLogger("flextok_ar")

__all__ = ["T5Embedder", "T5EmbedderWithMLP", "TextToEmbedMLP"]


class T5Embedder(nn.Module):
    """T5 text embedder using HuggingFace transformers.
    
    Takes text inputs and returns encoded embeddings with attention mask.
    Supports conditioning dropout for classifier-free guidance.
    
    Args:
        read_key: Key to read text from data_dict.
        text_embeddings_write_key: Key to write text embeddings to.
        text_embeddings_mask_write_key: Key to write attention mask to.
        hf_hub_path: HuggingFace model ID (e.g., 'google/flan-t5-xl').
        encoder_seqlen_max: Maximum encoder sequence length.
        decoder_seqlen: Decoder sequence length (for cross-attention mask).
        cond_dropout_p: Conditioning dropout probability for CFG.
    """
    
    def __init__(
        self,
        read_key: str,
        text_embeddings_write_key: str,
        text_embeddings_mask_write_key: str,
        hf_hub_path: str = "google/flan-t5-xl",
        encoder_seqlen_max: Optional[int] = None,
        decoder_seqlen: Optional[int] = None,
        cond_dropout_p: float = 0.0,
    ):
        super().__init__()
        self.read_key = read_key
        self.text_embeddings_write_key = text_embeddings_write_key
        self.text_embeddings_mask_write_key = text_embeddings_mask_write_key
        
        self.max_length = encoder_seqlen_max
        self.decoder_seqlen = decoder_seqlen
        self.cond_dropout_p = cond_dropout_p
        
        # Import transformers and monkey-patch to avoid apex issues
        from transformers import AutoTokenizer, T5EncoderModel
        from transformers.models.t5 import modeling_t5
        
        # Replace T5LayerNorm with standard LayerNorm if apex causes issues
        # This prevents CUDA library loading errors with apex's fused kernels
        original_t5_layer_norm = modeling_t5.T5LayerNorm
        try:
            # Try using apex version first
            logger.info(f"Loading T5 model from {hf_hub_path}...")
            self.tokenizer = AutoTokenizer.from_pretrained(hf_hub_path)
            self.model = T5EncoderModel.from_pretrained(hf_hub_path, use_cache=False).eval()
        except (ImportError, OSError) as e:
            # If apex fails, use standard LayerNorm
            logger.warning(f"APEX fused kernels not available: {e}")
            logger.info("Falling back to standard LayerNorm...")
            
            # Monkey-patch T5LayerNorm to use standard LayerNorm
            class StandardLayerNorm(nn.Module):
                def __init__(self, hidden_size, eps=1e-6):
                    super().__init__()
                    self.weight = nn.Parameter(torch.ones(hidden_size))
                    self.variance_epsilon = eps
                
                def forward(self, hidden_states):
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states / torch.sqrt(variance + self.variance_epsilon)
                    return self.weight * hidden_states
            
            modeling_t5.T5LayerNorm = StandardLayerNorm
            
            # Now load the model
            self.tokenizer = AutoTokenizer.from_pretrained(hf_hub_path)
            self.model = T5EncoderModel.from_pretrained(hf_hub_path, use_cache=False).eval()
            
            # Restore original T5LayerNorm
            modeling_t5.T5LayerNorm = original_t5_layer_norm
        
        self.singleton_value = ""  # Empty string for unconditional generation
        
        self.freeze()
    
    def freeze(self) -> "T5Embedder":
        """Freeze all parameters."""
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        return self
    
    @property
    def device(self) -> torch.device:
        """Get device of the model."""
        return next(self.parameters()).device
    
    def convert_mask_to_bias(self, mask: torch.Tensor, Mq: int) -> torch.Tensor:
        """Convert T5 attention mask to cross-attention bias.
        
        Args:
            mask: Attention mask, shape [B, Mkv], where 1 = valid, 0 = masked.
            Mq: Query sequence length.
        
        Returns:
            Attention bias, shape [B, Mq, Mkv], where 0 = valid, -inf = masked.
        """
        mask = mask.float()
        mask = torch.where(mask == 1.0, 0.0, float("-inf"))
        
        if mask.ndim == 3:
            assert mask.shape[1] == Mq
            return mask  # Already [B, Mq, Mkv]
        else:
            # Expand to [B, Mq, Mkv]
            mask = torch.repeat_interleave(mask.unsqueeze(1), Mq, dim=1)
            return mask
    
    @torch.no_grad()
    def forward(self, data_dict: dict[str, Any], **_) -> dict[str, Any]:
        """Encode text into embeddings.
        
        Args:
            data_dict: Input data containing text at self.read_key.
        
        Returns:
            data_dict: Output with text embeddings and attention mask.
        """
        texts = data_dict[self.read_key]
        
        # Apply conditioning dropout for CFG training
        if self.cond_dropout_p > 0.0 and self.training:
            import random
            texts = [
                "" if random.random() < self.cond_dropout_p else text
                for text in texts
            ]
        
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            padding="max_length" if self.max_length else "longest",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )
        
        input_ids = tokenized.input_ids.to(self.device)
        attention_mask = tokenized.attention_mask.to(self.device)
        
        # Encode
        encoder_output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        text_embeddings = encoder_output.last_hidden_state  # [B, L, D]
        
        # Convert mask to bias for cross-attention
        if self.decoder_seqlen:
            attn_mask = self.convert_mask_to_bias(attention_mask, self.decoder_seqlen)
        else:
            attn_mask = attention_mask
        
        # Write outputs
        data_dict[self.text_embeddings_write_key] = text_embeddings
        data_dict[self.text_embeddings_mask_write_key] = attn_mask
        
        return data_dict


class TextToEmbedMLP(MLP):
    """MLP for projecting text embeddings to model dimension.
    
    Args:
        text_dim: Input text embedding dimension.
        embed_dim: Output embedding dimension.
        act_layer: Activation function.
        use_bias: Whether to use bias in linear layers.
    """
    
    def __init__(
        self,
        text_dim: int,
        embed_dim: int,
        act_layer: nn.Module = None,
        use_bias: bool = False,
    ):
        if act_layer is None:
            act_layer = nn.GELU(approximate="tanh")
        
        # The MLP structure from checkpoint:
        # fc1: [hidden, input] = [2048, 2048]
        # fc2: [output, hidden] = [2304, 2048]
        # So hidden_features = text_dim (not embed_dim)
        super().__init__(
            in_features=text_dim,
            hidden_features=text_dim,  # Use text_dim as hidden, not embed_dim
            out_features=embed_dim,
            act_layer=act_layer,
            use_bias=use_bias,
        )


class T5EmbedderWithMLP(nn.Module):
    """T5 embedder with MLP projection.
    
    Combines T5Embedder with an MLP to project text embeddings
    to the model's embedding dimension.
    
    Args:
        t5_embedder: T5Embedder instance.
        mlp: MLP for projection.
    """
    
    def __init__(
        self,
        t5_embedder: T5Embedder,
        mlp: TextToEmbedMLP,
    ):
        super().__init__()
        self.t5_embedder = t5_embedder
        self.mlp = mlp
        
        # Expose these for downstream use
        self.read_key = t5_embedder.read_key
        self.text_embeddings_write_key = t5_embedder.text_embeddings_write_key
        self.text_embeddings_mask_write_key = t5_embedder.text_embeddings_mask_write_key
        self.singleton_value = t5_embedder.singleton_value
    
    def forward(self, data_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Encode and project text embeddings.
        
        Args:
            data_dict: Input data.
        
        Returns:
            data_dict: Output with projected text embeddings.
        """
        # Encode text
        data_dict = self.t5_embedder(data_dict, **kwargs)
        
        # Project embeddings
        text_embeddings = data_dict[self.text_embeddings_write_key]
        projected_embeddings = self.mlp(text_embeddings)
        data_dict[self.text_embeddings_write_key] = projected_embeddings
        
        return data_dict

