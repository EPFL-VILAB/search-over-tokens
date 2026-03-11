"""Base class and factory for verifiers (reward models).

This module provides the abstract interface for verifiers that score
generated images based on prompts.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import torch
from PIL import Image

_DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / "configs" / "components" / "verifiers"

__all__ = ["BaseVerifier", "VerifierFactory"]


class BaseVerifier(ABC):
    """Base class for verifiers (reward models).

    To add a new verifier:
        1. Subclass BaseVerifier
        2. Implement the score() method
        3. Register with VerifierFactory

    Example:
        >>> @VerifierFactory.register("my_verifier")
        >>> class MyVerifier(BaseVerifier):
        >>>     def _score(self, images, prompts, **kwargs):
        >>>         # Your scoring logic (called by the public score() method)
        >>>         return scores
    """

    requires_images: bool = True
    """Whether this verifier needs decoded images. Set to False for token-only
    verifiers (e.g. LikelihoodVerifier) so search algorithms can skip decoding."""

    def __init__(self, config: Dict, device: str = "cuda"):
        """Initialize verifier.
        
        Args:
            config: Configuration dictionary.
            device: Device to run the verifier on.
        """
        self.config = config
        self.device = device
    
    def score(
        self,
        images: Union[Any, List[Image.Image], torch.Tensor],
        prompts: Union[str, List[str], None] = None,
        **kwargs
    ) -> torch.Tensor:
        """Score images, optionally conditioned on text prompts.

        Prompts are optional for verifiers that operate on images alone
        (e.g. DreamSim, Aesthetic). Text-based verifiers (e.g. CLIP,
        ImageReward) will raise an error if prompts are omitted.

        Args:
            images: Single PIL Image, single tensor, or list of PIL Images / tensor [B, C, H, W]
            prompts: Single text prompt, list of text prompts, or None for
                prompt-free verifiers.
            **kwargs: Additional scoring parameters

        Returns:
            Scores tensor of shape [B]
        """
        if not isinstance(images, (list, tuple)):
            images = [images]
        else:
            images = list(images)
        if prompts is not None:
            if not isinstance(prompts, (list, tuple)):
                prompts = [prompts]
            else:
                prompts = list(prompts)
        return self._score(images, prompts, **kwargs)

    @abstractmethod
    def _score(
        self,
        images: List,
        prompts: Optional[List[str]],
        **kwargs
    ) -> torch.Tensor:
        """Score a list of images. Subclasses implement this.

        Args:
            images: List of PIL Images or tensors.
            prompts: List of text prompts, or None for prompt-free verifiers.
        """
        pass

    def batch_score(
        self,
        images: Union[List[Image.Image], torch.Tensor],
        prompts: Union[List[str], None] = None,
        batch_size: int = 32,
        **kwargs
    ) -> torch.Tensor:
        """Score images in batches for efficiency.

        Args:
            images: List of PIL Images or tensor [B, C, H, W]
            prompts: List of text prompts, or None for prompt-free verifiers.
            batch_size: Batch size for processing

        Returns:
            Scores tensor of shape [B]
        """
        scores = []
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i+batch_size]
            batch_prompts = prompts[i:i+batch_size] if prompts is not None else None
            scores.append(self.score(batch_images, batch_prompts, **kwargs))
        return torch.cat(scores)


class VerifierFactory:
    """Factory for creating verifiers.
    
    Verifiers are loaded lazily - modules are only imported when a verifier
    is actually requested. This avoids loading dependencies for unused verifiers.
    """
    
    _registry = {}
    _module_map = {
        "clip": "sot.verifiers.clip_verifier",
        "image_reward": "sot.verifiers.image_reward_verifier",
        "dreamsim": "sot.verifiers.dreamsim_verifier",
        "aesthetic": "sot.verifiers.aesthetic_verifier",
        "pickscore": "sot.verifiers.pickscore_verifier",
        "cyclereward": "sot.verifiers.cyclereward_verifier",
        "hpsv2": "sot.verifiers.hpsv2_verifier",
        "likelihood": "sot.verifiers.likelihood_verifier",
        "grounded_sam": "sot.verifiers.spatial_verifier",
        "ensemble": "sot.verifiers.ensemble_verifier",
    }
    _loaded_modules = set()
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a verifier."""
        def decorator(verifier_class):
            if name in cls._registry:
                print(f"Warning: Overwriting existing verifier '{name}'")
            cls._registry[name] = verifier_class
            return verifier_class
        return decorator
    
    @classmethod
    def _load_verifier_module(cls, name: str):
        """Lazily load a verifier module if not already loaded."""
        if name in cls._loaded_modules:
            return
        
        if name not in cls._module_map:
            return
        
        module_path = cls._module_map[name]
        try:
            __import__(module_path)
            cls._loaded_modules.add(name)
        except ImportError as e:
            # Module will be loaded when create() is called and will raise appropriate error
            pass
    
    @classmethod
    def create(cls, name: str, config: Dict = None, **kwargs):
        """Create a verifier by name.

        The verifier module is loaded lazily on first use.
        If config is omitted, defaults are loaded from
        configs/verifiers/{name}.yaml automatically.
        """
        # Load the module if not already loaded
        if name not in cls._registry:
            cls._load_verifier_module(name)

        if name not in cls._registry:
            available = list(cls._module_map.keys())
            raise ValueError(
                f"Unknown verifier: '{name}'. Available: {available}\n"
                f"Note: The verifier module may have failed to import. "
                f"Check that required dependencies are installed."
            )

        config = dict(config or {})
        config.setdefault("name", name)

        # Auto-load defaults from YAML when no config keys besides name are set
        if len(config) == 1:  # only "name" was present
            yaml_path = _DEFAULT_CONFIG_DIR / f"{name}.yaml"
            if yaml_path.exists():
                from omegaconf import OmegaConf
                yaml_cfg = OmegaConf.load(str(yaml_path))
                # YAML top-level key is "verifier"
                if OmegaConf.select(yaml_cfg, "verifier") is not None:
                    yaml_config = OmegaConf.to_container(yaml_cfg.verifier, resolve=True)
                    config = {**yaml_config, **config}

        return cls._registry[name](config, **kwargs)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all available verifiers (from module map, not registry).
        
        This returns all known verifiers without loading their modules.
        """
        return list(cls._module_map.keys())

