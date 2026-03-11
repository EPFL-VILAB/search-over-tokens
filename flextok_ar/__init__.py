"""FlexTok-AR: Autoregressive Image Generation with FlexTok Tokenizers.

This package provides AR-specific extensions for image generation using
FlexTok tokenizers and L3M training framework.
"""

__version__ = "0.1.0"

from flextok_ar.model import integration, preprocessors, generation
from flextok_ar.utils import helpers
from flextok_ar.utils.helpers import (
    load_model,
    tensor_to_pil,
    generate_t2i,
    generate_c2i,
)

__all__ = [
    "integration", "preprocessors", "generation", "helpers",
    "load_model", "tensor_to_pil", "generate_t2i", "generate_c2i",
]

