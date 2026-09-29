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

import torch
from torch.distributed.fsdp import (
    FSDPModule,
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
from weathergen.model.engines import EfficientBilinear, TargetPredictionEngine
from weathergen.model.layers import MLP
from weathergen.model.model import Model, create_model_params
from weathergen.model.norms import AdaLayerNormLayer, RMSNorm
from weathergen.model.utils import apply_fct_to_blocks, check_encoder_checkpoint, freeze_weights
from weathergen.utils.distributed import is_root
from weathergen.utils.performance import register_nvtx_hooks
from weathergen.utils.utils import get_dtype

logger = logging.getLogger(__name__)


# same as in config: student_teacher, forecasting, masking
type TrainingMode = str


@torch.no_grad()
def _materialize_model(model, device):
    """Initialize storage lost during meta construction, including concrete-owned queries."""
    model.to_empty(device=device)
    model.reset_parameters()
    for module in model.modules():
        if isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, torch.nn.RMSNorm | EfficientBilinear):
            module.reset_parameters()
        elif isinstance(module, AdaLayerNormLayer):
            module.initialise_weights()
        elif isinstance(module, TargetPredictionEngine):
            torch.nn.init.zeros_(module.pos_embed)
    for encoder in model.encoders.values():
        reset_queries = getattr(encoder, "reset_queries", None)
        if reset_queries is not None:
            reset_queries()


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
    model_creation_device = "meta" if with_ddp and with_fsdp else device
    with torch.device(model_creation_device):
        model = get_model(cf, training_mode, dataset, overrides)

    if cf.get("profiling", {}).get("nvtx_annotate", False):
        logger.info("Registering NVTX hooks for model.")
        register_nvtx_hooks(model)

    # freeze request model part
    apply_fct_to_blocks(model, model.cf.freeze_modules, freeze_weights)

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

        # One unit per branch keeps data-dependent local operations out of FSDP collectives.
        for encoder in model.encoders.values():
            encoder.sharded_training = True
            fully_shard(encoder, **fsdp_kwargs)

        for module in model.forecast_engine.modules():
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
    elif with_ddp and with_fsdp:
        _materialize_model(model, device)

    resolved_model = (
        model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
    )
    model_params = create_model_params(resolved_model.cf).to(device)

    return model, model_params


def load_model(cf, model, device, run_id: str, mini_epoch=-1):
    """Loads model state from checkpoint and checks for missing and unused keys.
    Args:
        run_id : model_id of the trained model
        mini_epoch : The mini_epoch to load. Default (-1) is the latest mini_epoch
    """

    path_run = get_path_model(cf, run_id=run_id)
    mini_epoch_id = (
        f"chkpt{mini_epoch:05d}" if mini_epoch != -1 and mini_epoch is not None else "latest"
    )
    filename = f"{run_id}_{mini_epoch_id}.chkpt"

    params = torch.load(
        path_run / filename, map_location=torch.device("cpu"), mmap=True, weights_only=True
    )

    check_encoder_checkpoint(model, params)
    # Normalize only the leading DDP namespace, including before sharded key lookup.
    params = {key.removeprefix("module."): value for key, value in params.items()}
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        params = {"module." + key: value for key, value in params.items()}

    if isinstance(model, FSDPModule):
        meta_sharded_sd = model.state_dict()
        if meta_sharded_sd.keys() - params.keys() and any(
            tensor.is_meta for tensor in meta_sharded_sd.values()
        ):
            # Initialize optional new heads before loading, never reset their restored siblings.
            _materialize_model(model, device)
        maybe_sharded_sd = {}
        for param_name, full_tensor in params.items():
            sharded_meta_param = meta_sharded_sd.get(param_name)
            if sharded_meta_param is None:
                continue
            maybe_sharded_sd[param_name] = (
                distribute_tensor(
                    full_tensor,
                    sharded_meta_param.device_mesh,
                    sharded_meta_param.placements,
                )
                if isinstance(sharded_meta_param, DTensor)
                else full_tensor.to(device)
            )
        mkeys, ukeys = model.load_state_dict(maybe_sharded_sd, strict=False, assign=True)
        ukeys.extend(key for key in params if key not in meta_sharded_sd)

    else:
        # load checkpoint
        mkeys, ukeys = model.load_state_dict(params, strict=False)
        model = model.to(device)

    # warn about difference in checkpoint and model
    if not mkeys and not ukeys:
        logger.info(f"Checkpoint {filename} loaded successfully with all weights matching.")
    if mkeys:
        logger.warning(f"Missing keys when loading model: {mkeys}")
    if ukeys:
        logger.warning(f"Unused keys when loading model: {ukeys}")

    return model


def get_model(cf: Config, training_mode: TrainingMode, dataset, overrides):
    """
    Create model

    cf :
    training_mode :
    dataset :
    """

    cf_with_overrides = merge_configs(cf, overrides)
    # Dataset size lists follow physical reader order, which overrides may rearrange/subset.
    stream_names = dataset.streams_datasets
    sources_by_name = dict(zip(stream_names, dataset.get_sources_size(), strict=True))
    channels_by_name = dict(zip(stream_names, dataset.get_targets_num_channels(), strict=True))
    coords_by_name = dict(zip(stream_names, dataset.get_targets_coords_size(), strict=True))
    sources_size = [sources_by_name[name] for name in cf_with_overrides.streams]
    targets_num_channels = [channels_by_name[name] for name in cf_with_overrides.streams]
    targets_coords_size = [coords_by_name[name] for name in cf_with_overrides.streams]
    return Model(
        cf_with_overrides, sources_size, targets_num_channels, targets_coords_size
    ).create()
