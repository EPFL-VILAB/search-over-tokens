"""Base class and factory for AR priors (models).

This module provides the abstract interface that all AR models must implement,
enabling easy extensibility and composition.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Union, Tuple, Optional
import torch

logger = logging.getLogger("soto.ar_priors")

__all__ = ["BaseARPrior", "ARPriorFactory"]


class BaseARPrior(ABC):
    """Base class for all AR priors (models).
    
    This interface allows easy integration of new AR models.
    To add a new AR model:
        1. Subclass BaseARPrior
        2. Implement the abstract methods
        3. Register with ARPriorFactory using the @ARPriorFactory.register decorator
    
    Example:
        >>> @ARPriorFactory.register("my_model")
        >>> class MyARModel(BaseARPrior):
        >>>     def generate_next_tokens(self, prompt, current_tokens, num_new_tokens, num_samples, **kwargs):
        >>>         # Your implementation - generate_tokens is automatically provided
        >>>         pass
        >>>     # Implement other abstract methods...
    """
    
    def __init__(self, config: Dict, device: str = "cuda"):
        """Initialize AR prior.

        Args:
            config: Configuration dictionary for the model.
            device: Device to run the model on.
        """
        self.config = config
        self.device = device
        self.rng = torch.Generator(device=device)

    def set_seed(self, seed: int) -> "BaseARPrior":
        """Seed the internal RNG for reproducible generation.

        Call once before a search run. Successive generate_next_tokens calls
        will still produce diverse outputs because the generator state advances
        on each call — only re-calling set_seed() resets it.

        Args:
            seed: Integer seed value.

        Returns:
            self, for chaining: ``ar_prior.set_seed(42)``
        """
        self.rng.manual_seed(seed)
        return self
    
    @abstractmethod
    def generate_next_tokens(
        self,
        prompt: Union[str, List[str]],
        current_tokens: torch.Tensor,
        num_new_tokens: int = 1,
        num_samples: int = 1,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generate next tokens given current sequence.
        
        This is used for incremental generation in search algorithms.
        Subclasses must implement this method.
        
        Args:
            prompt: Text prompt(s) or class labels
            current_tokens: Current token sequence [batch, seq_len]
            num_new_tokens: Number of new tokens to generate
            num_samples: Number of samples to generate (branching factor)
            **kwargs: Additional generation parameters:
                - return_probs: Whether to return token probabilities (default: True)
        
        Returns:
            Tuple of (new_tokens, token_probs):
            - new_tokens: New token IDs [batch * num_samples, num_new_tokens]
            - token_probs: Token probabilities [batch * num_samples] or None if not available
        """
        pass
    
    @abstractmethod
    def decode_tokens(
        self,
        tokens: torch.Tensor,
        **kwargs
    ) -> List:
        """Decode tokens to images.
        
        Args:
            tokens: Token sequences [batch, seq_len]
            **kwargs: Additional decoding parameters (e.g., timesteps, guidance_scale)
            
        Returns:
            List of PIL Images
        """
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Return vocabulary size.
        
        Returns:
            Size of the token vocabulary
        """
        pass
    
    @abstractmethod
    def get_max_tokens(self) -> int:
        """Return maximum sequence length.
        
        Returns:
            Maximum number of tokens supported
        """
        pass


class ARPriorFactory:
    """Factory for creating AR priors.
    
    This factory pattern enables dynamic registration and creation of AR models,
    making it easy to add new models without modifying core code.
    
    Example:
        >>> @ARPriorFactory.register("my_model")
        >>> class MyARModel(BaseARPrior):
        >>>     pass
        >>> 
        >>> # Later, create the model
        >>> model = ARPriorFactory.create("my_model", config={...})
    """
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register an AR prior.
        
        Args:
            name: Name to register the AR prior under.
        
        Returns:
            Decorator function that registers the class.
        """
        def decorator(ar_class):
            if name in cls._registry:
                logger.warning(f"Overwriting existing AR prior '{name}'")
            cls._registry[name] = ar_class
            return ar_class
        return decorator
    
    @classmethod
    def create(cls, name: str, config: Dict = None, **kwargs) -> BaseARPrior:
        """Create an AR prior by name.

        Args:
            name: Name of the registered AR prior.
            config: Configuration dictionary.  If omitted (or empty), the prior
                will attempt to auto-discover its default YAML config by name.
            **kwargs: Additional keyword arguments passed to the constructor.

        Returns:
            Instantiated AR prior.

        Raises:
            ValueError: If the AR prior name is not registered.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown AR prior: '{name}'. Available: {available}"
            )
        config = dict(config or {})
        config.setdefault("name", name)  # let the prior resolve its default config
        return cls._registry[name](config, **kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered AR priors.
        
        Returns:
            List of registered AR prior names.
        """
        return list(cls._registry.keys())

