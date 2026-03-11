"""muP (Maximal Update Parametrization) utilities for inference.

Copied from l3m-ar to make flextok_ar independent.
This module provides functions to apply muP to models for proper inference.
"""

import copy
import logging
from typing import Any, Tuple, Dict

import torch
import torch.nn as nn
from mup.layer import MuReadout

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig, OmegaConf

logger = logging.getLogger("flextok_ar")

__all__ = ["maybe_instantiate_mup", "MuReadoutFSDP"]


class MuReadoutFSDP(MuReadout):
    """Version of MuReadout that works with FSDP.
    
    This extends mup.layer.MuReadout to handle the case where weights
    might be sharded with FSDP (Fully Sharded Data Parallel).
    """

    def width_mult(self):
        """Get width multiplier for muP scaling."""
        if hasattr(self.weight, "infshape"):
            width_mult = self.weight.infshape.width_mult()
        elif hasattr(self, "weight_infshape"):
            width_mult = self.weight_infshape.width_mult()
        else:
            raise AssertionError(
                "Please call set_base_shapes(...). If using torch.nn.DataParallel, "
                "switch to distributed training with "
                "torch.nn.parallel.DistributedDataParallel instead"
            )
        return width_mult


def _fix_fsdp_readout(module: MuReadoutFSDP) -> None:
    """Fix FSDP readout layer by storing infshape.
    
    Args:
        module: MuReadoutFSDP module to fix.
    """
    assert isinstance(module, MuReadoutFSDP)
    assert hasattr(module.weight, "infshape")
    module.weight_infshape = module.weight.infshape


def override_config(obj: Any, override_key: str, override_value: Any) -> Any:
    """Override any occurrence of the override_key in any nested structure.
    
    This recursively traverses dicts, lists, and OmegaConf structures to
    replace all occurrences of a specific key with a new value.
    
    Args:
        obj: Object to traverse (dict, list, or OmegaConf).
        override_key: Key to search for and replace.
        override_value: Value to replace with.
    
    Returns:
        Modified object with replacements.
    """
    if isinstance(obj, (dict, DictConfig)):
        result = {
            k: (
                override_value
                if k == override_key
                else override_config(v, override_key, override_value)
            )
            for k, v in obj.items()
        }
        return OmegaConf.create(result) if isinstance(obj, DictConfig) else result
    elif isinstance(obj, (list, ListConfig)):
        result = [override_config(item, override_key, override_value) for item in obj]
        return OmegaConf.create(result) if isinstance(obj, ListConfig) else result
    return obj


def instantiate_with_mup(
    model: nn.Module, model_config: dict, muP_base_dim: int, override_key: str
) -> nn.Module:
    """Instantiate model with muP (Maximal Update Parametrization).
    
    This creates a temporary base model with smaller dimensions, uses it to
    set the muP base shapes, then initializes the actual model with muP.
    
    Args:
        model: Model instance to apply muP to.
        model_config: Model configuration dict.
        muP_base_dim: Base dimension for muP (typically 256).
        override_key: Keys to override for base model (e.g., "dim-embed_dim").
    
    Returns:
        Model with muP applied.
    """
    import mup
    
    logger.info(f"Using muP with base dim {muP_base_dim}.")

    # Create temporary base model config with smaller dimensions
    mup_config = copy.deepcopy(model_config)
    for override_key_i in override_key.split("-"):
        logger.info(f"muP overriding key '{override_key_i}'")
        mup_config = override_config(mup_config, override_key_i, muP_base_dim)

    # Instantiate base model
    model_muP_base = instantiate(mup_config)

    # Set the base shapes, init the actual model, and delete the base model
    logger.info("Setting muP base shapes.")
    mup.set_base_shapes(model, model_muP_base, rescale_params=False)
    
    # Fix FSDP readout layers if present
    for _, module in model.named_modules():
        if isinstance(module, MuReadoutFSDP):
            _fix_fsdp_readout(module)
    
    # Initialize muP weights if the model supports it.
    # For inference with checkpoint loading, this is optional since weights
    # get overwritten by the checkpoint. The critical part is set_base_shapes
    # above, which sets infshape needed by MuReadoutFSDP.width_mult().
    if hasattr(model, "init_weights_muP"):
        logger.info("Initializing muP weights.")
        model.init_weights_muP()
    else:
        logger.info("Model does not have init_weights_muP; skipping (OK for inference with checkpoint).")
    
    del model_muP_base

    return model


def maybe_instantiate_mup(
    model: nn.Module,
    model_config: dict,
    optim_config: dict,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Apply muP to model if configured, otherwise return as-is.
    
    This function checks if muP is enabled in the optimizer config and
    applies it if necessary. For inference, this ensures the model has
    the correct scaling factors set.
    
    Args:
        model: Model instance.
        model_config: Model configuration dict.
        optim_config: Optimizer configuration dict (contains muP settings).
    
    Returns:
        Tuple of (model, model_config) with muP applied if configured.
    """
    # If no muP configuration is specified, return as-is
    if "param" not in optim_config or optim_config["param"] != "mup":
        return model, model_config

    # Deep copy model_config to avoid unintentional modifications
    model_config = copy.deepcopy(model_config)

    # Validate muP configuration
    assert (
        "muP_base_dim" in optim_config
    ), "muP_base_dim must be specified in the optimizer configuration."
    assert optim_config["muP_base_dim"] is not None, "muP_base_dim cannot be None."

    assert (
        "muP_override_key" in optim_config
    ), "muP_override_key must be specified in the optimizer configuration."
    assert (
        optim_config["muP_override_key"] is not None
    ), "muP_override_key cannot be None."

    muP_base_dim = optim_config["muP_base_dim"]

    # Apply muP to the model
    model = instantiate_with_mup(
        model=model,
        model_config=model_config,
        muP_base_dim=muP_base_dim,
        override_key=optim_config["muP_override_key"],
    )
    model_config["muP_base_dim"] = muP_base_dim

    return model, model_config

