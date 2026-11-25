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


def get_active_token_idxs(cf, model_params, batch) -> torch.Tensor | None:
    """Indices of the global tokens that belong to the active region of the batch.

    A HEALPix cell is active if any sample in the batch has source tokens or target
    coordinates in it (any stream, any step). The 1-ring neighbours of these cells are added,
    because the decoder reads the 9-cell neighbourhood of each target cell.

    For limited-area data (e.g. ICON, 45 of 12288 cells at level 5) this lets the global and
    forecasting engines attend only over the region that has data.

    Returns None if ``latent_active_cells_only`` is off (all cells are used).
    The returned indices refer to the token layout
    ``[aux tokens (register + class) x queries, cells x queries]`` used by the engines.
    """
    if not cf.get("latent_active_cells_only", False):
        return None

    num_cells = model_params.num_healpix_cells
    num_queries = cf.ae_local_num_queries
    device = batch.tokens_lens.device

    # source occupancy: tokens_lens has shape (steps, samples, streams, cells)
    active = batch.tokens_lens.reshape(-1, num_cells).sum(0) > 0

    # target occupancy (needed when target cells are masked in the source)
    for sample in batch.samples:
        for stream_data in sample.streams_data.values():
            for lens in stream_data.target_coords_lens:
                if len(lens) == num_cells:
                    active |= lens.to(device) > 0

    # add the 1-ring so the decoder neighbourhood of every target cell is processed
    nbours = model_params.hp_nbours.to(device).long()
    cells = torch.unique(nbours[active].flatten())

    num_aux = (cf.num_register_tokens + cf.num_class_tokens) * num_queries
    queries = torch.arange(num_queries, device=device)
    cell_token_idxs = (cells.unsqueeze(1) * num_queries + queries).flatten() + num_aux
    aux_token_idxs = torch.arange(num_aux, device=device)

    return torch.cat([aux_token_idxs, cell_token_idxs])


def apply_on_token_subset(fct, tokens, idxs, coords, *args):
    """Run ``fct(tokens, *args, coords)`` only on ``tokens[:, idxs]`` and write the result back.

    Tokens outside ``idxs`` are passed through unchanged. If ``idxs`` is None, ``fct`` runs on
    all tokens.
    """
    if idxs is None:
        return fct(tokens, *args, coords)

    coords_sub = coords[:, idxs] if coords is not None else None
    out = fct(tokens[:, idxs], *args, coords_sub)
    return tokens.index_copy(1, idxs, out.to(tokens.dtype))


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
