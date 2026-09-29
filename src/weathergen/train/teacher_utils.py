# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import logging

import torch
import torch.nn as nn

from weathergen.common.config import get_encoder_streams, get_path_model
from weathergen.model.engines import (
    IdentityEngine,
    LatentPredictionHeadIdentity,
    LatentPredictionHeadMLP,
    LatentPredictionHeadTransformer,
)
from weathergen.model.utils import check_encoder_checkpoint

logger = logging.getLogger(__name__)


def _create_teacher_heads(
    name: str, head_type: str, dim_embed: int, loss_conf, cf=None
) -> nn.Module:
    """Create a latent prediction head for a given SSL loss type.

    Mirrors Model._create_latent_pred_head() logic with per-loss-type token settings:
        iBOT: use_class_token=True, use_patch_token=True
        DINO: use_class_token=True, use_patch_token=False
    """
    if name == "iBOT":
        use_class_token, use_patch_token = True, True
    elif name == "DINO":
        use_class_token, use_patch_token = True, False
    else:
        raise ValueError(f"_create_teacher_heads does not support loss type {name!r}")

    if head_type == "mlp":
        return LatentPredictionHeadMLP(
            f"{name}-head", dim_embed, loss_conf, use_class_token, use_patch_token
        )
    elif head_type == "transformer":
        if cf is None:
            raise ValueError("LatentPredictionHeadTransformer requires a global config (cf)")
        return LatentPredictionHeadTransformer(
            cf, f"{name}-head", dim_embed, loss_conf, use_class_token, use_patch_token
        )
    elif head_type == "identity":
        return LatentPredictionHeadIdentity()
    else:
        raise ValueError(f"Unknown latent prediction head type {head_type!r}")


def check_teacher_input_contract(teacher_cf, input_cf, dataset=None) -> None:
    """Reject teacher configs that would reinterpret already-tokenized input batches."""
    teacher_routes = get_encoder_streams(teacher_cf)
    input_routes = get_encoder_streams(input_cf)

    def require_equal(label, teacher_value, input_value):
        if teacher_value != input_value:
            raise ValueError(
                f"Teacher input contract mismatch for {label}: "
                f"teacher={teacher_value!r}, input={input_value!r}. "
                "Cross-topology teacher retokenization is not supported."
            )

    require_equal("encoder names/order", list(teacher_routes), list(input_routes))
    require_equal("stream names/order", list(teacher_cf.streams), list(input_cf.streams))
    for key in ("fe_healpix_level", "fe_num_queries", "num_class_tokens", "num_register_tokens"):
        require_equal(key, teacher_cf.get(key, 0), input_cf.get(key, 0))
    for name, members in teacher_routes.items():
        require_equal(f"encoders.{name} membership/order", members, input_routes[name])
        require_equal(
            f"encoders.{name}.healpix_level",
            teacher_cf.encoders[name].healpix_level,
            input_cf.encoders[name].healpix_level,
        )
    for name in dict.fromkeys(stream for members in teacher_routes.values() for stream in members):
        teacher_stream = teacher_cf.streams[name]
        input_stream = input_cf.streams[name]
        for key, default in (("token_size", None), ("tokenize_spacetime", False)):
            require_equal(
                f"streams.{name}.{key}",
                teacher_stream.get(key, default),
                input_stream.get(key, default),
            )
        require_equal(
            f"streams.{name}.embed.num_tokens",
            teacher_stream.embed.get("num_tokens", 1),
            input_stream.embed.get("num_tokens", 1),
        )
        # Saved resolved channel lists take precedence over channel-selection expressions.
        channel_keys = ["train_source_channels", "val_source_channels"]
        compared_channels = False
        for key in channel_keys:
            if key in teacher_stream and key in input_stream:
                require_equal(f"streams.{name}.{key}", teacher_stream[key], input_stream[key])
                compared_channels = True
        if dataset is not None:
            key = f"{dataset._stage}_source_channels"
            channels = teacher_stream.get(key, teacher_stream.get("train_source_channels"))
            if channels is not None:
                actual = dataset.streams_datasets[name].readers[0].source_channels
                require_equal(f"streams.{name}.{key}", list(channels), list(actual))
                compared_channels = True
        if not compared_channels:
            for key in ("source", "source_exclude"):
                require_equal(
                    f"streams.{name}.{key}", teacher_stream.get(key), input_stream.get(key)
                )


def prepare_encoder_teacher(model: nn.Module, training_cfg, override_cfg) -> None:
    """Strip a model to encoder-only and create fresh SSL latent heads.

    Modifies model in-place:
    1. Removes forecast_engine, decoders, pred_heads, embed_target_coords
    2. Ensures latent_pre_norm exists
    3. Creates fresh latent_heads based on the student's SSL loss config
    """
    # Strip non-encoder components
    teacher_dim_embed = override_cfg.fe_dim_embed
    model.forecast_engine = IdentityEngine()
    model.embed_target_coords = nn.ModuleDict()
    model.target_token_engines = nn.ModuleDict()
    model.pred_heads = nn.ModuleDict()

    # Ensure latent_pre_norm exists (teacher may not have had SSL training)
    if model.latent_pre_norm is None:
        model.latent_pre_norm = nn.LayerNorm(teacher_dim_embed)

    # Create fresh latent heads from student's SSL config
    model.latent_heads = nn.ModuleDict()
    ssl_losses = [
        v for v in training_cfg.losses.values() if v.type == "LossLatentSSLStudentTeacher"
    ]
    for ssl_loss in ssl_losses:
        for name, conf in ssl_loss.loss_fcts.items():
            if name == "JEPA":
                model.latent_heads[name] = LatentPredictionHeadIdentity()
            elif name in ("iBOT", "DINO"):
                head_type = conf.get("head", "mlp").lower()
                model.latent_heads[name] = _create_teacher_heads(
                    name, head_type, teacher_dim_embed, conf, cf=override_cfg
                )
            else:
                raise ValueError(f"Unknown SSL loss type {name!r}")


def load_encoder_from_checkpoint(
    model: nn.Module,
    cf,
    teacher_run_id: str,
    teacher_mini_epoch: int | None,
    device: torch.device | str,
) -> None:
    """Load only encoder weights from a checkpoint into a model.

    Filters checkpoint to encoders.* and latent_pre_norm.* keys only. Required named
    encoder weights are validated; omitted heads remain intentionally non-strict.
    """
    path_run = get_path_model(cf, run_id=teacher_run_id)
    mini_epoch_id = (
        f"chkpt{teacher_mini_epoch:05d}"
        if teacher_mini_epoch is not None and teacher_mini_epoch != -1
        else "latest"
    )
    filename = f"{teacher_run_id}_{mini_epoch_id}.chkpt"

    params = torch.load(path_run / filename, map_location="cpu", mmap=True, weights_only=True)
    params = {key.removeprefix("module."): value for key, value in params.items()}
    check_encoder_checkpoint(model, params)

    # Filter to encoder + latent_pre_norm only
    encoder_params = {
        k: v for k, v in params.items() if k.startswith(("encoders.", "latent_pre_norm."))
    }

    model.load_state_dict(encoder_params, strict=False)
    model.to(device)

    logger.info(f"Teacher: Loaded encoder weights from checkpoint {filename}")
