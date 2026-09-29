# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.


import logging
import re

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def add_healpix_latents_(
    destination: torch.Tensor,
    source: torch.Tensor,
    source_level: int,
    target_level: int,
) -> torch.Tensor:
    """Add NESTED [B, cells, queries, features] latents into an owned accumulator."""
    if any(type(level) is not int or level < 0 for level in (source_level, target_level)):
        raise ValueError("HEALPix levels must be nonnegative integers.")
    if destination.ndim != 4 or source.ndim != 4:
        raise ValueError("Latents must have explicit [B, N, Q, D] axes.")
    if source.shape[1] != 12 * 4**source_level or destination.shape[1] != 12 * 4**target_level:
        raise ValueError("Latent cell counts do not match their HEALPix levels.")
    if source.shape[0] != destination.shape[0] or source.shape[2:] != destination.shape[2:]:
        raise ValueError("Latent batch, query, and feature axes must match.")
    if source_level > target_level:
        children = 4 ** (source_level - target_level)
        destination.add_(source.unflatten(1, (destination.shape[1], children)).sum(2))
    elif source_level < target_level:
        children = 4 ** (target_level - source_level)
        destination.unflatten(1, (source.shape[1], children)).add_(source.unsqueeze(2))
    else:
        destination.add_(source)
    return destination


def check_encoder_checkpoint(model: nn.Module, state: dict[str, torch.Tensor]) -> None:
    """Reject legacy or incomplete encoder weights before non-strict checkpoint loading."""
    keys = {key.removeprefix("module.") for key in state}
    if any(key.startswith("encoder.") for key in keys):
        raise ValueError(
            "Legacy encoder.* checkpoint weights are incompatible with named encoders. "
            "Use a checkpoint saved with encoders.<name>.* and matching named routing."
        )
    required = {
        key.removeprefix("module.")
        for key in model.state_dict()
        if key.removeprefix("module.").startswith("encoders.")
    }
    missing = sorted(required - keys)
    if missing:
        raise ValueError(
            "Checkpoint is missing required named encoder weights: "
            + ", ".join(missing)
            + ". Use a checkpoint with every configured encoder and matching architecture."
        )


def get_num_parameters(block):
    nps = filter(lambda p: p.requires_grad, block.parameters())
    return sum([torch.prod(torch.tensor(p.size())) for p in nps])


def freeze_weights(block):
    if hasattr(block, "name"):
        logger.info(f"Freeze block {block.name}")
    for p in block.parameters():
        p.requires_grad = False


def set_to_eval(block):
    if hasattr(block, "name"):
        logger.info(f"Set block {block.name} to eval mode")
    block.eval()


def apply_fct_to_blocks(model, blocks, fct):
    """
    Apply a function to specific blocks of a model.
    Args:
        model : model instance with attribute named_modules
        blocks : regex pattern to match block names
        fct : function to apply to matching blocks
    """

    for name, module in model.named_modules():
        name = module.name if hasattr(module, "name") else name
        # avoid the whole model element which has name ''
        if (re.fullmatch(blocks, name) is not None) and (name != ""):
            fct(module)


class ActivationFactory:
    _registry = {
        "identity": nn.Identity,
        "tanh": nn.Tanh,
        "softmax": nn.Softmax,
        "sigmoid": nn.Sigmoid,
        "gelu": nn.GELU,
        "relu": nn.ReLU,
        "leakyrelu": nn.LeakyReLU,
        "elu": nn.ELU,
        "selu": nn.SELU,
        "prelu": nn.PReLU,
        "softplus": nn.Softplus,
        "linear": nn.Linear,
        "logsoftmax": nn.LogSoftmax,
        "silu": nn.SiLU,
        "swish": nn.SiLU,
    }

    @classmethod
    def get(cls, name: str, **kwargs):
        name = name.lower()
        if name not in cls._registry:
            raise ValueError(f"Unsupported activation type: '{name}'")
        fn = cls._registry[name]
        return fn(**kwargs) if callable(fn) else fn
