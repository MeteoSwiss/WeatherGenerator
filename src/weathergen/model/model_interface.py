# ruff: noqa: B006

# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import itertools
import logging
from collections import defaultdict

import torch
from omegaconf import OmegaConf
from torch.distributed.fsdp import (
    MixedPrecisionPolicy,
    fully_shard,
)
from torch.distributed.tensor import DTensor, distribute_tensor
from weathergen.common.config import Config, get_path_model, merge_configs

from weathergen.model.attention import (
    MultiCrossAttentionHeadVarlen,
    MultiCrossAttentionHeadVarlenSlicedQ,
    MultiSelfAttentionHead,
    MultiSelfAttentionHeadLocal,
    MultiSelfAttentionHeadVarlen,
)
from weathergen.model.layers import MLP
from weathergen.model.model import Model, ModelParams
from weathergen.model.utils import apply_fct_to_blocks, freeze_weights, reset_leaf_parameters
from weathergen.utils.distributed import is_root
from weathergen.utils.performance import register_nvtx_hooks
from weathergen.utils.utils import get_dtype

logger = logging.getLogger(__name__)


# same as in config: student_teacher, forecasting, masking
type TrainingMode = str


def init_model_and_shard(
    cf,
    dataset,
    run_id_contd,
    mini_epoch_contd,
    training_mode,
    device,
    with_ddp,
    with_fsdp,
    overrides={},
):
    model_creation_device = "meta" if with_ddp and with_fsdp else "cuda"
    with torch.device(model_creation_device):
        model = get_model(cf, training_mode, dataset, overrides)

    if cf.get("profiling", {}).get("nvtx_annotate", False):
        logger.info("Registering NVTX hooks for model.")
        register_nvtx_hooks(model)

    # freeze request model part
    apply_fct_to_blocks(model, cf.freeze_modules, freeze_weights)

    # TODO: this should be handled in the encoder to be close where q_cells is defined
    if "q_cells" in cf.freeze_modules:
        for encoder in model.encoders.values():
            encoder.q_cells.requires_grad = False

    if with_ddp and not with_fsdp:
        # create DDP model if running without FSDP
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            broadcast_buffers=True,
            find_unused_parameters=cf.get("ddp_find_unused_parameters", True),
            gradient_as_bucket_view=True,
            bucket_cap_mb=512,
        )

    elif with_ddp and with_fsdp:
        # with DDP *and() FSDP
        fsdp_kwargs = {
            "mp_policy": (
                MixedPrecisionPolicy(
                    param_dtype=get_dtype(cf.mixed_precision_dtype),
                    reduce_dtype=torch.float32,
                )
                if cf.with_mixed_precision
                else None
            ),
        }
        modules_to_shard = (
            MLP,
            MultiSelfAttentionHeadLocal,
            MultiSelfAttentionHead,
            MultiCrossAttentionHeadVarlen,
            MultiCrossAttentionHeadVarlenSlicedQ,
            MultiSelfAttentionHeadVarlen,
        )

        for encoder in model.encoders.values():
            for module in encoder.ae_local_engine.ae_local_blocks.modules():
                if isinstance(module, modules_to_shard):
                    fully_shard(module, **fsdp_kwargs)

            for module in encoder.ae_local_global_engine.ae_adapter.modules():
                if isinstance(module, modules_to_shard):
                    fully_shard(module, **fsdp_kwargs)

            for module in encoder.ae_global_engine.ae_global_blocks.modules():
                if isinstance(module, modules_to_shard):
                    fully_shard(module, **fsdp_kwargs)

        for module in model.forecast_engine.fe_blocks.modules():
            if isinstance(module, modules_to_shard):
                # reshard_after_forward=False keeps FE parameters unsharded
                # during the multi-step rollout loop.
                # Needed for pushforward trick.
                fully_shard(module, reshard_after_forward=False, **fsdp_kwargs)

        for module in model.latent_heads.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **fsdp_kwargs)

        full_precision_fsdp_kwargs = {
            "mp_policy": (
                MixedPrecisionPolicy(
                    param_dtype=torch.float32,
                    reduce_dtype=torch.float32,
                )
                if cf.with_mixed_precision
                else None
            ),
        }

        for module in model.target_token_engines.modules():
            if isinstance(module, modules_to_shard):
                fully_shard(module, **full_precision_fsdp_kwargs)

    if with_ddp and with_fsdp:
        fully_shard(model)
        for tensor in itertools.chain(model.parameters(), model.buffers()):
            assert tensor.device == torch.device("meta")

        # For reasons we do not yet fully understand, when using train continue in some
        # instances, FSDP2 does not register the forward function in the embedding
        # engine as a forward function. Thus, yielding a crash
        # because the input tensors are not converted to DTensors. This seems to primarily
        # occur during validation.
        for encoder in model.encoders.values():
            for embed in encoder.embed_engine.embeds.values():
                torch.distributed.fsdp.register_fsdp_forward_method(embed, "forward")

    # complete initalization and load model if inference/continuing a run
    if run_id_contd is not None:
        if is_root():
            logger.info(f"Continuing run with id={run_id_contd} at mini_epoch {mini_epoch_contd}.")
        model = load_model(cf, model, device, run_id_contd, mini_epoch_contd)
    elif cf.get("load_chkpt", {}).get("run_id", None):
        run_id = cf.load_chkpt.run_id
        mini_epoch = cf.load_chkpt.get("mini_epoch", -1)
        if is_root():
            logger.info(f"Loading checkpoint from id={run_id} at mini_epoch {mini_epoch}.")
        model = load_model(cf, model, device, run_id, mini_epoch)
    else:
        if with_ddp and with_fsdp:
            # the model was built on the meta device: materialise the storage, then fill it
            model.to_empty(device="cuda")
            model.reset_parameters()

    # model params
    model_params = ModelParams(cf).create(cf)
    model_params.reset_parameters()
    model_params = model_params.to(f"cuda:{cf.local_rank}")

    return model, model_params


def _strip_module_prefix(key: str) -> str:
    """Key without the "module." that DistributedDataParallel adds."""

    return key[len("module.") :] if key.startswith("module.") else key


def _remap_encoder_levels(cf, model, params: dict) -> dict:
    """Rename per-level checkpoint keys so an encoder is reused at a different HEALPix level.

    Encoders are keyed by level (``self.encoders[str(L)]``), so a resolution curriculum gives a
    stream's encoder a new name and its weights would be dropped as unused. ``{L: L'}`` renames
    them on load; ``{L: [L, L']}`` copies one encoder into two, for the rung where an encoder
    splits. Write the levels as STRING keys: the config is saved as JSON, whose keys are always
    strings, and config.save's _strip_interpolation cannot index an integer key.

    Only buffers are level-sized, and _geometry_buffers_to_rebuild rebuilds those, but
    ``token_size`` must not change with the level: it is the embedding's input width.
    """

    remap = cf.get("encoder_level_remap", None)
    if not remap:
        return params

    if OmegaConf.is_config(remap):
        remap = OmegaConf.to_container(remap, resolve=True)
    remap = {
        str(src): [str(d) for d in (dst if isinstance(dst, list | tuple) else [dst])]
        for src, dst in remap.items()
    }

    # regridders are left out: pure geometry, rebuilt from the new grid either way
    containers = ("encoders.", "fusion_norms.")
    model_keys = {_strip_module_prefix(key) for key in model.state_dict()}
    levels_in_model = {
        key[len(container) :].partition(".")[0]
        for container in containers
        for key in model_keys
        if key.startswith(container)
    }
    # Entries naming a level this model does not have are inherited from an earlier stage: when
    # continuing, the parent run's saved config is the merge base (config.py:451-454), so a
    # curriculum accumulates every rung's mapping. Only the applicable ones are kept.
    stale = {src: [d for d in dsts if d not in levels_in_model] for src, dsts in remap.items()}
    stale = {src: dsts for src, dsts in stale.items() if dsts}
    remap = {
        src: [d for d in dsts if d in levels_in_model]
        for src, dsts in remap.items()
        if any(d in levels_in_model for d in dsts)
    }
    assert remap, (
        f"encoder_level_remap names no level this model has: its per-level modules are at "
        f"{sorted(levels_in_model)}, and every destination was {sorted(stale)}."
    )
    if stale and is_root():
        logger.info(f"Ignoring inherited encoder_level_remap entries for absent levels: {stale}.")

    def split(key: str) -> tuple[str, str, str] | None:
        """(container, level, rest) of a per-level key, or None."""
        for container in containers:
            if key.startswith(container):
                level, _, rest = key[len(container) :].partition(".")
                return (container, level, rest) if rest else None
        return None

    renamed = {}
    num_renamed, num_dropped = 0, 0
    for key, tensor in params.items():
        # the non-sharded path may still carry a "module." prefix at this point
        prefix = "module." if key.startswith("module.") else ""
        parts = split(_strip_module_prefix(key))
        if parts is None or parts[1] not in remap:
            renamed[key] = tensor
            continue

        container, level, rest = parts
        for dst in remap[level]:
            new_key = f"{prefix}{container}{dst}.{rest}"
            assert new_key == key or new_key not in params, (
                f"encoder_level_remap {remap} would overwrite {new_key}, which the checkpoint "
                "already contains: the source and destination levels both exist in it."
            )
            if _strip_module_prefix(new_key) not in model_keys:
                # a weight of a stream the destination encoder does not serve
                num_dropped += 1
                continue
            renamed[new_key] = tensor
            num_renamed += 1

    assert num_renamed > 0, (
        f"encoder_level_remap {remap} matched no checkpoint keys; check the source levels "
        "against the run being continued."
    )
    if is_root():
        logger.info(
            f"Reusing encoder weights across levels {remap}: {num_renamed} tensors renamed, "
            f"{num_dropped} dropped as not part of the destination encoder."
        )

    return renamed


def _geometry_buffers_to_rebuild(model, params) -> tuple[set[str], set[str]]:
    """Checkpoint entries whose shape no longer matches the model.

    Grid geometry -- regrid index tables, HEALPix neighbourhoods, positional encodings -- is sized
    by the configured grids, so changing a level or an active region legitimately resizes it. Those
    buffers are dropped here and rebuilt from the current grid by their owner's reset_parameters;
    anything else is a learned weight, and rebuilding it would silently discard training.

    Returns the keys to drop, and of those the ones on a module that also owns learned weights
    (an encoder reused at a new level): _reinit_missing_modules must refill those in place.
    """

    model_sd = model.state_dict()
    buffer_names = {name for name, _ in model.named_buffers()}
    modules = dict(model.named_modules())

    stale = set()
    kept_module_buffers = set()
    for param_name, full_tensor in params.items():
        model_tensor = model_sd.get(param_name)
        if model_tensor is None or model_tensor.shape == full_tensor.shape:
            continue

        parent = param_name.rsplit(".", 1)[0]
        parent_module = modules.get(parent)
        assert param_name in buffer_names, (
            f"Shape mismatch for {param_name}: checkpoint has {tuple(full_tensor.shape)}, "
            f"model expects {tuple(model_tensor.shape)}. It is a learned weight, not grid "
            "geometry, so rebuilding it would discard training; make the config's grid "
            "geometry match the checkpoint instead."
        )
        assert parent_module is not None and hasattr(parent_module, "reset_parameters"), (
            f"Geometry buffer {param_name} changed shape but '{parent}' has no "
            "reset_parameters to rebuild it from the current grid."
        )
        if next(parent_module.parameters(), None) is not None:
            kept_module_buffers.add(param_name)
        logger.warning(
            f"Rebuilding geometry buffer {param_name} from the current grid: checkpoint "
            f"{tuple(full_tensor.shape)}, model {tuple(model_tensor.shape)}."
        )
        stale.add(param_name)

    return stale, kept_module_buffers


def _reinit_missing_modules(
    model, missing_keys, to_empty: bool, kept_module_buffers=frozenset()
) -> None:
    """Initialize the modules owning ``missing_keys``.

    These are new network parts (e.g. for fine-tuning) plus the geometry buffers dropped by
    _geometry_buffers_to_rebuild.

    ``kept_module_buffers`` are dropped buffers whose module kept its weights from the checkpoint.
    That module's own reset_parameters refills them (by convention it does not touch weights); the
    full re-initialisation below would discard those weights. On the sharded path such buffers are
    still on the meta device, since load_state_dict only gave storage to the keys it was passed.
    """

    if not missing_keys:
        return

    all_modules = dict(model.named_modules())
    missing = set(missing_keys)
    rebuilt = missing & set(kept_module_buffers)
    to_refill = defaultdict(set)
    for key in rebuilt:
        owner, name = key.rsplit(".", 1)
        to_refill[owner].add(name)

    # such a module must not also be missing one of its own tensors: initialising it would take
    # the wholesale path below and discard what was just loaded
    conflicts = {key for key in missing - rebuilt if key.rsplit(".", 1)[0] in to_refill}
    assert not conflicts, (
        f"{sorted(conflicts)} are missing from the checkpoint, but their module was kept for the "
        "weights it did provide, so re-initialising it would discard them. The checkpoint and "
        "this config disagree about more than grid geometry."
    )

    # keep the highest-level roots only, so a subtree is initialized once
    roots = set()
    for path in sorted({key.rsplit(".", 1)[0] for key in missing - rebuilt}):
        if not any(path.startswith(root + ".") for root in roots):
            roots.add(path)

    for path in sorted(roots):
        if is_root():
            logger.info(f"Initializing module not found in checkpoint: {path}")
        module = all_modules[path]
        if to_empty:
            module.to_empty(device="cuda")
            reset_leaf_parameters(module)
            module.reset_parameters()
        elif hasattr(module, "reset_parameters"):
            module.reset_parameters()

    for path, names in sorted(to_refill.items()):
        module = all_modules[path]
        for name in sorted(names):
            buffer = module.get_buffer(name)
            if buffer.is_meta:
                module.register_buffer(name, torch.empty_like(buffer, device="cuda"))
        if is_root():
            logger.info(f"Rebuilding geometry buffers of module kept from checkpoint: {path}")
        module.reset_parameters()


def load_model(cf, model, device, run_id: str, mini_epoch=-1):
    """Loads model state from checkpoint and checks for missing and unused keys.
    Args:
        run_id : model_id of the trained model
        mini_epoch : The mini_epoch to load. Default (-1) is the latest mini_epoch
    """

    path_run = get_path_model(run_id=run_id)
    mini_epoch_id = (
        f"chkpt{mini_epoch:05d}" if mini_epoch != -1 and mini_epoch is not None else "latest"
    )
    filename = f"{run_id}_{mini_epoch_id}.chkpt"

    params = torch.load(
        path_run / filename, map_location=torch.device("cpu"), mmap=True, weights_only=True
    )

    params = _remap_encoder_levels(cf, model, params)

    is_model_sharded = cf.with_ddp and cf.with_fsdp
    if is_model_sharded:
        meta_sharded_sd = model.state_dict()
        stale_buffers, kept_module_buffers = _geometry_buffers_to_rebuild(model, params)
        maybe_sharded_sd = {}
        for param_name, full_tensor in params.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            if sharded_meta_param is None:
                logger.warning(f"Parameter {param_name} from checkpoint not found in model.")
                continue
            if param_name in stale_buffers:
                continue
            if isinstance(sharded_meta_param, DTensor):
                sharded_tensor = distribute_tensor(
                    full_tensor,
                    sharded_meta_param.device_mesh,
                    sharded_meta_param.placements,
                )
                maybe_sharded_sd[param_name] = torch.nn.Parameter(sharded_tensor)
            else:
                maybe_sharded_sd[param_name] = full_tensor.to(device)
        # choose `assign=True` for sharded model since we cannot call `copy_` on meta tensor
        mkeys, ukeys = model.load_state_dict(maybe_sharded_sd, strict=False, assign=True)

        # new network parts (e.g. for fine-tuning) and the geometry buffers dropped above
        _reinit_missing_modules(
            model, mkeys, to_empty=True, kept_module_buffers=kept_module_buffers
        )

    else:
        # fix mismatch between state_dict keys that can occur between interactive/non-interactive
        model_has_prefix_module = list(model.state_dict().keys())[0].split(".")[0] == "module"
        params_has_prefix_module = list(params.keys())[0].split(".")[0] == "module"
        if model_has_prefix_module and not params_has_prefix_module:
            # add "module." prefix
            params_temp = {}
            for k in params.keys():
                params_temp["module." + k] = params[k]
            params = params_temp
        elif not model_has_prefix_module and params_has_prefix_module:
            # remove "module." prefix
            params_temp = {}
            for k in params.keys():
                params_temp[k.replace("module.", "")] = params[k]
            params = params_temp
        stale_buffers, kept_module_buffers = _geometry_buffers_to_rebuild(model, params)
        params = {k: v for k, v in params.items() if k not in stale_buffers}
        # load checkpoint
        mkeys, ukeys = model.load_state_dict(params, strict=False)
        model = model.to(device)

        _reinit_missing_modules(
            model, mkeys, to_empty=False, kept_module_buffers=kept_module_buffers
        )

    # warn about difference in checkpoint and model
    if len(mkeys) == 0 and len(ukeys) == 0:
        logger.info(f"Checkpoint {filename} loaded successfully with all weights matching.")
    if len(mkeys) > 0:
        logger.warning(f"Missing keys when loading model: {mkeys}")
    if len(ukeys) > 0:
        logger.warning(f"Unused keys when loading model: {ukeys}")

    return model


def get_model(cf: Config, training_mode: TrainingMode, dataset, overrides):
    """
    Create model

    cf :
    training_mode :
    dataset :
    """

    # TODO: how to avoid the dependence on dataset
    sources_size = dataset.get_sources_size()
    targets_num_channels = dataset.get_targets_num_channels()
    targets_coords_size = dataset.get_targets_coords_size()
    grid_shapes = dataset.get_grid_shapes()

    cf_with_overrides = merge_configs(cf, overrides)
    return Model(
        cf_with_overrides, sources_size, targets_num_channels, targets_coords_size, grid_shapes
    ).create()
