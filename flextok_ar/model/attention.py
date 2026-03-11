"""Cross-attention with mask support for T2I generation.

Extends public L3M's GeneralizedAttention to support attention masks.
"""

import torch
import torch.nn as nn
from typing import Any, Optional

from l3m.model.layers.attention import GeneralizedAttention

__all__ = ["GeneralizedAttentionWithMask"]


class GeneralizedAttentionWithMask(GeneralizedAttention):
    """GeneralizedAttention with mask support for cross-attention.
    
    Extends the public L3M's GeneralizedAttention to handle attention masks,
    which is needed for cross-attending to variable-length text sequences.
    
    Args:
        Same as GeneralizedAttention.
    """
    
    def forward(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **_: Any,
    ) -> torch.Tensor:
        """Forward pass with optional attention mask.
        
        Args:
            queries: Query tensor, shape [B, N, C].
            keys: Key tensor, shape [B, M, C].
            values: Value tensor, shape [B, M, C].
            mask: Optional attention mask, shape [B, N, M] or [B, 1, N, M].
                  Values should be 0 for valid positions, -inf for masked.
        
        Returns:
            Output tensor, shape [B, N, C].
        """
        B, N, C = queries.shape
        M = keys.shape[1]
        
        # Project Q, K, V
        q = (
            self.Wq(queries)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.Wk(keys)
            .reshape(B, M, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.Wv(values)
            .reshape(B, M, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        
        # Apply QK normalization
        q, k = self.q_norm(q), self.k_norm(k)
        
        # Apply relative positional embeddings if present
        if self.relative_pos_embed is not None:
            q, k, v = self.relative_pos_embed(q, k, v)
        
        # Expand mask to match num_heads if needed
        if mask is not None:
            if mask.ndim == 3:
                # [B, N, M] -> [B, num_heads, N, M]
                mask = mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        
        # Scaled dot-product attention with mask
        x = nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=self.is_causal,
        )
        
        # Reshape and project output
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
