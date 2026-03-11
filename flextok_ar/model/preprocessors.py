"""AR preprocessors for image token embedding.

This module provides the ARImageEmbedPreprocessor which prepares
token embeddings for autoregressive generation.
"""

import logging
from typing import Any, Literal, Optional
import numpy as np
import torch
import torch.nn as nn

# Import from public L3M
from l3m.model.meta_models import ReadWriteBlock
from l3m.model.preprocessors.pos_embed import sinusoidal

logger = logging.getLogger("flextok_ar")

__all__ = ["ARImageEmbedPreprocessor"]


class ARImageEmbedPreprocessor(ReadWriteBlock):
    """Auto-regressive tokenized image embedding preprocessor.
    
    Creates target image tokens and shifted embeddings for AR training.
    Supports class conditioning with dropout for classifier-free guidance.
    
    Args:
        token_grid_size: Latent grid size after tokenization (e.g., [256, 1] for 1D).
        codebook_size: Image tokenizer codebook size.
        embed_dim: Transformer dimension.
        pos_embed_type: Positional embedding type ('absolute', 'sincos1d', 'sincos2d').
        init_style: Weight initialization style ('normal', 'sparse_transformer', 'uniform').
        num_classes: Number of classes for class-conditional generation.
        cond_dropout_prob: Dropout probability for classifier-free guidance.
        class_target_key: Key for reading class labels from data_dict.
        inference_read_key: Key for reading predicted tokens during generation.
        read_key: Key for reading input token IDs.
        write_key: Keys for writing [embeddings, target_tokens].
    """
    
    def __init__(
        self,
        token_grid_size: tuple[int, int],
        codebook_size: int,
        embed_dim: int,
        pos_embed_type: Optional[Literal["absolute", "sincos1d", "sincos2d"]] = None,
        init_style: Optional[Literal["normal", "sparse_transformer", "uniform"]] = "uniform",
        num_classes: Optional[int] = None,
        cond_dropout_prob: float = 0.0,
        class_target_key: str = "target",
        inference_read_key: str = "pred_image_token_ids",
        read_key: str = "image_token_ids",
        write_key: list = None,
        **kwargs,
    ):
        if write_key is None:
            write_key = ["input_embeddings", "target_token_ids"]
        
        super().__init__(read_key=read_key, write_key=write_key, **kwargs)
        
        self.class_target_key = class_target_key
        self.embeddings_key, self.target_token_ids_key = self.write_key
        self.inference_read_key = inference_read_key
        
        self.token_grid_size = token_grid_size
        self.num_tokens = np.prod(self.token_grid_size)
        self.embed_dim = embed_dim
        self.pos_embed_type = pos_embed_type
        
        # Positional embeddings
        self.positional_embedding = None
        if pos_embed_type in ["absolute", "sincos1d", "sincos2d"]:
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.num_tokens, embed_dim),
                requires_grad=(pos_embed_type == "absolute"),
            )
        elif pos_embed_type is not None:
            raise ValueError(f"Unsupported pos embedding type {pos_embed_type}.")
        
        # Token embedding
        self.init_style = init_style
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        
        # Start-of-sequence (SOS) token
        self.sos_token_embed = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        
        # Optional class embedding for class-conditional generation
        self.num_classes = num_classes
        self.cond_dropout_prob = cond_dropout_prob
        self.class_ignore_idx = -1
        self.cls_embedding: Optional[nn.Embedding] = None
        if self.num_classes is not None:
            self.cls_embedding = nn.Embedding(num_classes, embed_dim)
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights based on init_style."""
        if self.init_style == "normal":
            torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
            torch.nn.init.normal_(self.sos_token_embed, mean=0.0, std=0.02)
            if self.cls_embedding is not None:
                torch.nn.init.normal_(self.cls_embedding.weight, mean=0.0, std=0.02)
        elif self.init_style == "sparse_transformer":
            torch.nn.init.normal_(
                self.embedding.weight, mean=0.0, std=0.125 * (self.embed_dim ** -0.5)
            )
            torch.nn.init.normal_(
                self.sos_token_embed, mean=0.0, std=0.125 * (self.embed_dim ** -0.5)
            )
            if self.cls_embedding is not None:
                torch.nn.init.normal_(
                    self.cls_embedding.weight, mean=0.0, std=0.125 * (self.embed_dim ** -0.5)
                )
        elif self.init_style == "uniform":
            torch.nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
            torch.nn.init.uniform_(self.sos_token_embed, -0.1, 0.1)
            if self.cls_embedding is not None:
                torch.nn.init.uniform_(self.cls_embedding.weight, -0.1, 0.1)
        elif self.init_style is None:
            pass
        else:
            raise ValueError(f"Invalid initialization style {self.init_style}.")
        
        # Initialize positional embeddings
        if self.pos_embed_type == "absolute":
            torch.nn.init.normal_(self.positional_embedding, std=0.02)
        elif self.pos_embed_type == "sincos1d":
            sincos1d_init = sinusoidal.get_2d_sincos_pos_embed(
                self.embed_dim, (self.num_tokens, 1), cls_token=True
            )
            sincos1d_init = sincos1d_init[:-1]  # Remove CLS token
            self.positional_embedding.data.copy_(torch.from_numpy(sincos1d_init).float())
        elif self.pos_embed_type == "sincos2d":
            sincos2d_init = sinusoidal.get_2d_sincos_pos_embed(
                self.embed_dim, tuple(self.token_grid_size), cls_token=True
            )
            sincos2d_init = sincos2d_init[:-1]  # Remove CLS token
            self.positional_embedding.data.copy_(torch.from_numpy(sincos2d_init).float())
    
    def forward(self, data_dict: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Preprocess AR image tokens and embeddings.
        
        Args:
            data_dict: Input data containing:
                - self.read_key: Image token IDs, shape [B, L]
                - self.class_target_key: Optional class IDs for conditioning
        
        Returns:
            data_dict: Output data containing:
                - self.embeddings_key: Token embeddings, shape [B, L, D]
                - self.target_token_ids_key: Target token IDs, shape [B, L]
        """
        if not self.generation_mode:
            # Training mode: create targets and shifted inputs
            token_ids = data_dict[self.read_key]  # [B, L]
            target_ids = data_dict[self.read_key]  # [B, L]
        else:
            # Generation mode: use predicted tokens
            if self.inference_read_key in data_dict:
                token_ids = data_dict[self.inference_read_key]
            else:
                token_ids = data_dict[self.read_key]
            target_ids = None
        
        # Embed tokens
        token_embeds = self.embedding(token_ids)  # [B, L, D]
        
        if not self.generation_mode:
            # Remove last token's embeddings for concatenation with SOS
            token_embeds = token_embeds[:, :-1]  # [B, L-1, D]
        
        # Create input sequence: shift by one with SOS token
        input_embeds = torch.cat(
            [
                self.sos_token_embed.expand((token_embeds.shape[0], -1, -1)),
                token_embeds,
            ],
            dim=1,
        )  # [B, L, D]
        
        # Add positional embeddings
        if self.positional_embedding is not None:
            _, seq_length, _ = input_embeds.shape
            max_length = self.positional_embedding.shape[0]
            max_seq_length = min(seq_length, max_length)
            input_embeds = input_embeds + self.positional_embedding[:max_seq_length, :]
        
        # Add class conditioning if available
        if self.cls_embedding is not None:
            # [B,]
            class_ids = data_dict[self.class_target_key]
            
            # Handle ignored classes (for classifier-free guidance)
            # [B,], self.class_ignore_idx is ignored, used during l2i cfg.
            valid_mask = class_ids != self.class_ignore_idx
            class_ids[~valid_mask] = 0
            
            # Embed class indices
            # [B, D]
            cls_embeds = self.cls_embedding(class_ids)
            
            # Randomly dropout class conditioning.
            if self.training and self.cond_dropout_prob > 0.0:
                # [B,]
                keep_mask = (
                    torch.rand(cls_embeds.shape[0], device=cls_embeds.device)
                    > self.cond_dropout_prob
                )
                # [B, D]
                cls_embeds = cls_embeds * keep_mask.unsqueeze(1)
            # Apply the valid mask.
            cls_embeds = cls_embeds * valid_mask[:, None]
            
            # Add class conditioning to first token. i.e. SOS token.
            input_embeds[:, 0] += cls_embeds
        
        # Write outputs
        data_dict[self.embeddings_key] = input_embeds
        if target_ids is not None:
            data_dict[self.target_token_ids_key] = target_ids
        
        return data_dict

