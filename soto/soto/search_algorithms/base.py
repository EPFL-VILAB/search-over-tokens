"""Base class and factory for search algorithms.

This module provides the abstract interface for search algorithms that
operate over token sequences using AR priors and verifiers.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Union
from pathlib import Path
import torch

logger = logging.getLogger("soto.search_algorithms")

_DEFAULT_CONFIG_DIR = Path(__file__).parent.parent.parent / "configs" / "components" / "search_algorithms"

__all__ = ["SearchResult", "BaseSearchAlgorithm", "SearchAlgorithmFactory"]


class SearchResult:
    """Container for search results.

    Attributes:
        tokens: Generated token sequences [num_results, seq_len]
        images: List of PIL Images
        scores: Verifier scores [num_results]
        metadata: Additional information about the search
        step_images: Best image at each search step (None if not collected)
        step_scores: Best verifier score at each search step (None if not collected)
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        images: List,
        scores: torch.Tensor,
        metadata: Dict = None,
        step_images: List = None,
        step_scores: List = None,
        display_tokens: torch.Tensor = None,
    ):
        self.tokens = tokens
        self.images = images
        self.scores = scores
        self.metadata = metadata or {}
        self.step_images = step_images  # List[PIL Image], one per step
        self.step_scores = step_scores  # List[float], one per step
        self.display_tokens = display_tokens  # token seqs for all displayed images [num_results, seq_len]


class BaseSearchAlgorithm(ABC):
    """Base class for search algorithms.
    
    To add a new search algorithm:
        1. Subclass BaseSearchAlgorithm
        2. Implement the search() method
        3. Register with SearchAlgorithmFactory
    
    Example:
        >>> @SearchAlgorithmFactory.register("my_search")
        >>> class MySearch(BaseSearchAlgorithm):
        >>>     def search(self, prompt, num_results, **kwargs):
        >>>         # Your search logic
        >>>         return SearchResult(...)
    """
    
    def __init__(
        self,
        ar_prior,  # BaseARPrior
        verifier,  # BaseVerifier
        config: Dict
    ):
        """Initialize search algorithm.
        
        Args:
            ar_prior: AR prior (model) to use for generation.
            verifier: Verifier (reward model) to use for scoring.
            config: Configuration dictionary.
        """
        self.ar_prior = ar_prior
        self.verifier = verifier
        self.config = config
    
    @abstractmethod
    def search(
        self,
        prompt: Union[str, List[str]],
        num_results: int = 1,
        seed: int = None,
        **kwargs
    ) -> SearchResult:
        """Run search algorithm.

        Args:
            prompt: Text prompt(s) or class labels
            num_results: Number of results to return
            seed: Optional integer seed. When provided, resets the AR prior's
                internal RNG at the start of this call, making results
                reproducible across repeated calls with the same seed.
            **kwargs: Additional search parameters

        Returns:
            SearchResult containing best samples
        """
        pass


class SearchAlgorithmFactory:
    """Factory for creating search algorithms."""
    
    _registry = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register a search algorithm."""
        def decorator(search_class):
            if name in cls._registry:
                logger.warning(f"Overwriting existing search algorithm '{name}'")
            cls._registry[name] = search_class
            return search_class
        return decorator
    
    @classmethod
    def create(cls, name: str, ar_prior, verifier, config: Dict = None):
        """Create a search algorithm by name.

        If config is omitted, defaults are loaded from
        configs/search_algorithms/{name}.yaml (or {name}_search.yaml) automatically.
        """
        if name not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(
                f"Unknown search algorithm: '{name}'. Available: {available}"
            )

        config = dict(config or {})
        config.setdefault("name", name)

        # Auto-load defaults from YAML when no config keys besides name are set
        if len(config) == 1:  # only "name" was present
            # Search flat and one-level subdirs (e.g. janus/) for {name}.yaml or {name}_search.yaml
            search_dirs = [_DEFAULT_CONFIG_DIR] + [
                d for d in sorted(_DEFAULT_CONFIG_DIR.iterdir()) if d.is_dir()
            ]
            for stem in (name, f"{name}_search"):
                yaml_path = next(
                    (d / f"{stem}.yaml" for d in search_dirs if (d / f"{stem}.yaml").exists()),
                    None,
                )
                if yaml_path:
                    from omegaconf import OmegaConf
                    yaml_cfg = OmegaConf.load(str(yaml_path))
                    # YAML top-level key is "search"
                    if OmegaConf.select(yaml_cfg, "search") is not None:
                        yaml_config = OmegaConf.to_container(yaml_cfg.search, resolve=True)
                        config = {**yaml_config, **config}
                    break

        return cls._registry[name](ar_prior, verifier, config)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered search algorithms."""
        return list(cls._registry.keys())

