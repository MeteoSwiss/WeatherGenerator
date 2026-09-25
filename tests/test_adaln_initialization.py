# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import pytest
import torch
from omegaconf import OmegaConf

# Native block construction imports FlashAttention; the checks themselves stay on CPU.
pytest.importorskip("flash_attn", reason="flash_attn required to import native model blocks")

from weathergen.model.blocks import CrossAttentionBlock, SelfAttentionBlock  # noqa: E402
from weathergen.model.model import Model  # noqa: E402


def _model_with_adaln(device="cpu", cross_attention=False):
    cf = OmegaConf.create(
        dict(
            healpix_level=1,
            attention_dtype="fp32",
            streams={},
            num_register_tokens=0,
            num_class_tokens=0,
        )
    )
    with torch.device(device):
        model = Model(cf, [], [], [])
        kwargs = dict(
            dim_aux=3,
            with_adanorm=True,
            num_heads=2,
            dropout_rate=0.0,
            attention_kwargs={"norm_eps": 1e-5},
        )
        if cross_attention:
            block = CrossAttentionBlock(
                dim_q=8, dim_kv=8, with_self_attn=True, with_mlp=True, **kwargs
            )
        else:
            block = SelfAttentionBlock(dim=8, **kwargs)
        model.target_token_engines = torch.nn.ModuleDict({"probe": block})
    return model, block


def _inputs():
    x = torch.linspace(-2, 2, 40, device="cpu").reshape(5, 8)
    x[0] = 0  # Constant tokens must remain finite, including when dropout is disabled.
    c = torch.tensor([[0.5, -1.0, 2.0], [-0.3, 0.7, 1.0]], device="cpu")
    return x, c, torch.tensor([0, 2, 3], device="cpu")


@pytest.mark.parametrize(
    "initialization", ["construction", "model", "meta", "self", "cross", "projection"]
)
def test_adaln_zero_remains_identity_after_initialization(initialization):
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1234)
        model, block = _model_with_adaln(
            device="meta" if initialization == "meta" else "cpu",
            cross_attention=initialization == "cross",
        )
        if initialization == "meta":
            model.to_empty(device="cpu")
        if initialization in ("model", "meta"):
            model.reset_parameters()
        elif initialization in ("self", "cross"):
            block.initialise_weights()
        elif initialization == "projection":
            # Checkpoint loading resets newly added leaf modules through this same API.
            block.mlp_block.adaLN_modulation[-1].reset_parameters()
        model.eval()
        x, c, lens = _inputs()
        torch.testing.assert_close(block.mlp_block(x, c, x_lens=lens), x, rtol=0, atol=0)


def test_adaln_zero_can_learn_and_reload_without_resetting_conditioning():
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1234)
        model, block = _model_with_adaln()
        model.eval()
        x, c, lens = _inputs()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        loss = (block.mlp_block(x, c, x_lens=lens) - (x + 1)).square().mean()
        loss.backward()
        optimizer.step()
        learned = block.mlp_block(x, c, x_lens=lens).detach()
        assert (learned - (x + 1)).square().mean() < loss.detach()

        restored, restored_block = _model_with_adaln()
        restored.load_state_dict(model.state_dict())
        restored.eval()
        torch.testing.assert_close(
            restored_block.mlp_block(x, c, x_lens=lens), learned, rtol=0, atol=0
        )
