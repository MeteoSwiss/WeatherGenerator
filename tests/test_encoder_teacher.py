# (C) Copyright 2025 WeatherGenerator contributors.
# Licensed under the Apache Licence Version 2.0.

from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn
from omegaconf import OmegaConf

pytest.importorskip("flash_attn", reason="flash_attn required (GPU-only)")

from weathergen.model.ema import EMAModel  # noqa: E402
from weathergen.model.engines import LatentState  # noqa: E402
from weathergen.train.target_and_aux_ssl_teacher import (  # noqa: E402
    EMATeacher,
    FrozenTeacher,
    get_target_postprocessing,
)
from weathergen.train.teacher_utils import (  # noqa: E402
    check_teacher_input_contract,
    load_encoder_from_checkpoint,
    prepare_encoder_teacher,
)


def _training_cfg(losses=None):
    return OmegaConf.create(
        {
            "losses": {
                "ssl": {
                    "type": "LossLatentSSLStudentTeacher",
                    "loss_fcts": losses or {"JEPA": {"head": "identity"}},
                }
            }
        }
    )


def _config():
    return OmegaConf.create(
        {
            "encoders": {
                "global_grid": {
                    "healpix_level": 0,
                    "ae_global_dim_embed": 4,
                    "ae_local_num_queries": 1,
                },
                "regional": {
                    "healpix_level": 2,
                    "ae_global_dim_embed": 4,
                    "ae_local_num_queries": 1,
                },
            },
            "fe_healpix_level": 0,
            "fe_num_queries": 1,
            "fe_dim_embed": 4,
            "num_class_tokens": 1,
            "num_register_tokens": 0,
            "streams": {
                "A": {
                    "encoders": ["global_grid", "regional"],
                    "token_size": 2,
                    "embed": {"net": "transformer", "num_tokens": 1},
                    "train_source_channels": ["temperature", "pressure"],
                },
                "B": {
                    "encoders": ["regional"],
                    "token_size": 2,
                    "embed": {"net": "linear", "num_tokens": 1},
                    "train_source_channels": ["wind"],
                },
            },
        }
    )


def _model():
    model = nn.Module()
    model.encoders = nn.ModuleDict()
    for name in ("global_grid", "regional"):
        branch = nn.Linear(4, 4)
        branch.q_cells = nn.Parameter(torch.arange(4, dtype=torch.float32))
        model.encoders[name] = branch
    model.forecast_engine = nn.Linear(4, 4)
    model.embed_target_coords = nn.ModuleDict({"A": nn.Linear(3, 4)})
    model.target_token_engines = nn.ModuleDict({"A": nn.Linear(4, 4)})
    model.pred_heads = nn.ModuleDict({"A": nn.Linear(4, 1)})
    model.latent_pre_norm = None
    model.latent_heads = nn.ModuleDict()
    return model


class TestPrepareEncoderTeacher:
    def test_keeps_branch_values_and_normalizes_fused_features(self):
        model = _model()
        before = {k: v.clone() for k, v in model.encoders.state_dict().items()}
        prepare_encoder_teacher(model, _training_cfg(), _config())
        x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        torch.testing.assert_close(model.forecast_engine(x, 2, None), x)
        expected = torch.nn.functional.layer_norm(x, (4,))
        actual = model.latent_pre_norm(model.forecast_engine(x, 0))
        torch.testing.assert_close(actual, expected)
        for key, value in model.encoders.state_dict().items():
            torch.testing.assert_close(value, before[key])
        state = LatentState(actual[:, :1], None, actual[:, 1:], actual)
        torch.testing.assert_close(model.latent_heads["JEPA"](state), expected[:, 1:])
        assert not model.pred_heads and not model.target_token_engines

    def test_existing_norm_affine_values_survive(self):
        model = _model()
        model.latent_pre_norm = nn.LayerNorm(4)
        with torch.no_grad():
            model.latent_pre_norm.weight.fill_(3)
            model.latent_pre_norm.bias.fill_(2)
        prepare_encoder_teacher(model, _training_cfg(), _config())
        x = torch.arange(8, dtype=torch.float32).reshape(2, 4)
        torch.testing.assert_close(
            model.latent_pre_norm(x), torch.nn.functional.layer_norm(x, (4,)) * 3 + 2
        )

    @pytest.mark.parametrize("name,slots", [("iBOT", 4), ("DINO", 1)])
    def test_class_and_patch_head_outputs(self, name, slots):
        cfg = _training_cfg(
            {name: {"head": "mlp", "out_dim": 8, "num_layers": 2, "hidden_factor": 2}}
        )
        model = _model()
        prepare_encoder_teacher(model, cfg, _config())
        patches = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        state = LatentState(torch.ones(2, 1, 4), None, patches, None)
        head = model.latent_heads[name]
        out = head(state)
        expected = head.blocks(state.class_token)
        if name == "iBOT":
            expected = torch.cat((expected, head.blocks(patches)), dim=1)
        assert out.shape == (2, slots, 8)
        torch.testing.assert_close(out, expected)


class TestLoadEncoderFromCheckpoint:
    @pytest.mark.parametrize("mini_epoch", [-1, None, 42])
    @pytest.mark.parametrize("prefix", ["", "module."])
    def test_restores_all_named_weights_without_loading_heads(self, tmp_path, mini_epoch, prefix):
        model = _model()
        model.latent_pre_norm = nn.LayerNorm(4)
        untouched = {k: v.clone() for k, v in model.pred_heads.state_dict().items()}
        state = {
            prefix + key: torch.full_like(value, index + 1)
            for index, (key, value) in enumerate(model.state_dict().items())
        }
        suffix = "latest" if mini_epoch in (-1, None) else f"chkpt{mini_epoch:05d}"
        torch.save(state, tmp_path / f"teacher_{suffix}.chkpt")
        cf = OmegaConf.create({"full_model_path": str(tmp_path)})
        load_encoder_from_checkpoint(model, cf, "teacher", mini_epoch, "cpu")
        for key, value in model.state_dict().items():
            if key.startswith(("encoders.", "latent_pre_norm.")):
                torch.testing.assert_close(value, state[prefix + key])
        for key, value in model.pred_heads.state_dict().items():
            torch.testing.assert_close(value, untouched[key])

    @pytest.mark.parametrize("invalid", ["legacy", "missing_branch", "missing_query"])
    def test_rejects_incomplete_encoder_restoration(self, tmp_path, invalid):
        model = _model()
        state = model.state_dict()
        if invalid == "legacy":
            state = {"encoder.weight": torch.ones(4, 4)}
        elif invalid == "missing_branch":
            state = {k: v for k, v in state.items() if not k.startswith("encoders.regional.")}
        else:
            del state["encoders.regional.q_cells"]
        torch.save(state, tmp_path / "teacher_latest.chkpt")
        with pytest.raises(ValueError, match="Legacy|missing required named encoder"):
            load_encoder_from_checkpoint(
                model, OmegaConf.create({"full_model_path": str(tmp_path)}), "teacher", -1, "cpu"
            )


class TestFrozenTeacher:
    def test_all_parameters_stay_frozen_through_lifecycle(self):
        model = _model()
        prepare_encoder_teacher(model, _training_cfg(), _config())
        teacher = FrozenTeacher(model, _training_cfg(), nn.ModuleDict())
        before = {k: v.clone() for k, v in model.state_dict().items()}
        teacher.reset(batch_size=3)
        teacher.update_state_post_opt_step(2, None, None)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key])
        assert all(not param.requires_grad for param in model.parameters())
        assert not model.training

    @pytest.mark.parametrize(
        "path,value",
        [
            ("encoders.regional.healpix_level", 1),
            ("fe_healpix_level", 2),
            ("streams.A.encoders", ["global_grid"]),
            ("streams.A.token_size", 3),
            ("streams.A.tokenize_spacetime", True),
            ("streams.A.embed.num_tokens", 2),
            ("streams.A.train_source_channels", ["pressure", "temperature"]),
            ("num_class_tokens", 0),
        ],
    )
    def test_rejects_tokenization_mismatch(self, path, value):
        source = _config()
        teacher = copy.deepcopy(source)
        OmegaConf.update(teacher, path, value)
        with pytest.raises(ValueError, match="Teacher input contract mismatch"):
            check_teacher_input_contract(teacher, source)

    def test_rejects_route_order_and_query_count(self):
        source = _config()
        teacher = copy.deepcopy(source)
        teacher.encoders = dict(reversed(list(teacher.encoders.items())))
        with pytest.raises(ValueError, match="encoder names/order"):
            check_teacher_input_contract(teacher, source)
        teacher = copy.deepcopy(source)
        teacher.fe_num_queries = 2
        for branch in teacher.encoders.values():
            branch.ae_local_num_queries = 2
        with pytest.raises(ValueError, match="fe_num_queries"):
            check_teacher_input_contract(teacher, source)


class TestEMATeacher:
    @pytest.mark.parametrize("ddp_prefix", [False, True])
    def test_numerical_updates_include_every_branch_and_queries(self, ddp_prefix):
        student = _model()
        student.latent_pre_norm = nn.LayerNorm(4)
        target = copy.deepcopy(student)
        prepare_encoder_teacher(target, _training_cfg(), _config())
        original = student
        if ddp_prefix:
            original = nn.Module()
            original.module = student
        ema = EMAModel(original, target, halflife_steps=2, rampup_ratio=None)
        teacher = EMATeacher(student, ema, 2, _training_cfg(), nn.ModuleDict())
        before = {k: v.clone() for k, v in target.named_parameters()}
        with torch.no_grad():
            for param in student.parameters():
                param.add_(4)
        teacher.update_state_post_opt_step(10, None, student)
        assert teacher.get_current_beta(10) == pytest.approx(0.5)
        for key, value in target.named_parameters():
            torch.testing.assert_close(value, before[key] + 2)
        teacher.reset()
        source = dict(student.named_parameters())
        for key, value in target.named_parameters():
            torch.testing.assert_close(value, source[key])

    @pytest.mark.parametrize("change", ["missing_branch", "extra_branch", "shape"])
    def test_rejects_unmatched_encoder_parameters(self, change):
        student = _model()
        teacher = copy.deepcopy(student)
        if change == "missing_branch":
            del teacher.encoders["regional"]
        elif change == "extra_branch":
            teacher.encoders["extra"] = nn.Linear(4, 4)
        else:
            teacher.encoders["regional"].q_cells = nn.Parameter(torch.ones(8))
        with pytest.raises(ValueError, match="EMA"):
            EMAModel(student, teacher)


@pytest.mark.parametrize("loss_name", ["iBOT", "DINO"])
def test_target_postprocessing_rejects_missing_configuration(loss_name):
    losses = OmegaConf.create({loss_name: {"out_dim": 8}})
    with pytest.raises(KeyError, match="center_momentum"):
        get_target_postprocessing(losses, _training_cfg())
