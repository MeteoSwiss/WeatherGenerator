#!/usr/bin/env python
# ruff: noqa: T201
"""
Minimal, self-contained walk-through of the WeatherGenerator architecture.

Everything is plain PyTorch on a synthetic 64 x 64 x 3 field so it runs on a CPU in
seconds. The class names and data flow mirror the real implementation:

    src/weathergen/model/model.py      Model.forward / Model.predict_decoders
    src/weathergen/model/encoder.py    EncoderModule.forward / assimilate_local
    src/weathergen/model/engines.py    EmbeddingEngine, LocalAssimilationEngine,
                                       Local2GlobalAssimilationEngine,
                                       GlobalAssimilationEngine, ForecastingEngine,
                                       TargetPredictionEngine, EnsPredictionHead

Simplifications vs. the real model (see README.md for the full list):
  * a regular lat/lon grid split into square "cells" replaces the HEALPix cells
  * every cell holds the same number of tokens, so attention is batched instead of
    variable-length (varlen / flash attention)
  * one stream, no masking, no register/class tokens, no SSL heads

Run:
    uv run python examples/architecture_demo/wg_mini_forward.py [--steps N]
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# --------------------------------------------------------------------------------------
# Config: same knobs as config/default_config.yml, just tiny
# --------------------------------------------------------------------------------------
@dataclass
class Config:
    # data
    grid: int = 64  # 64 x 64 points
    num_channels: int = 3  # source == target channels here
    cells_per_side: int = 8  # 8 x 8 = 64 cells  (real: 12 * 4**healpix_level)
    token_size: int = 8  # points per token  (stream cfg `token_size`)
    forecast_steps: int = 1  # latent roll-out steps
    # embedding network (stream cfg `embed`)
    embed_dim: int = 32
    embed_heads: int = 4
    embed_blocks: int = 1
    # local assimilation engine
    ae_local_dim_embed: int = 32
    ae_local_num_blocks: int = 1
    ae_local_num_heads: int = 4
    # local -> global adapter and learnable queries
    ae_local_num_queries: int = 2  # queries per cell
    ae_global_dim_embed: int = 32
    ae_adapter_num_heads: int = 4
    # global assimilation engine
    ae_global_num_blocks: int = 2
    ae_global_num_heads: int = 4
    # forecasting engine
    fe_num_blocks: int = 1
    fe_num_heads: int = 4
    # decoder (stream cfg `embed_target_coords`, `target_readout`, `pred_head`)
    tr_dim_embed: int = 32
    tr_num_layers: int = 1
    tr_num_heads: int = 4

    @property
    def num_cells(self) -> int:
        return self.cells_per_side**2

    @property
    def points_per_cell(self) -> int:
        return (self.grid // self.cells_per_side) ** 2

    @property
    def tokens_per_cell(self) -> int:
        return self.points_per_cell // self.token_size


# --------------------------------------------------------------------------------------
# Synthetic data: a smooth 3-channel field on a lat/lon grid that drifts in time
# --------------------------------------------------------------------------------------
def make_grid(cf: Config) -> torch.Tensor:
    """Returns [grid*grid, 2] (lat, lon) in degrees, row-major."""
    lat = torch.linspace(-90, 90, cf.grid)
    lon = torch.linspace(0, 360, cf.grid + 1)[:-1]
    lat, lon = torch.meshgrid(lat, lon, indexing="ij")
    return torch.stack([lat.flatten(), lon.flatten()], dim=-1)


def make_field(coords: torch.Tensor, t: float, num_channels: int) -> torch.Tensor:
    """Returns [num_points, num_channels]; channel k is a wave with its own wavenumber."""
    lat, lon = torch.deg2rad(coords[:, 0]), torch.deg2rad(coords[:, 1])
    chans = [
        torch.sin((k + 1) * lon + t) * torch.cos((k + 1) * lat) for k in range(num_channels)
    ]
    return torch.stack(chans, dim=-1)


# --------------------------------------------------------------------------------------
# Tokenization: points -> cells -> tokens
# Real: weathergen/datasets/tokenizer_utils.py (tokenize_apply_mask_source / _target)
# --------------------------------------------------------------------------------------
def cell_index(cf: Config, coords: torch.Tensor) -> torch.Tensor:
    """Which cell each point falls into (real: healpy.ang2pix)."""
    n = cf.grid // cf.cells_per_side
    row = torch.arange(cf.grid).repeat_interleave(cf.grid) // n
    col = torch.arange(cf.grid).repeat(cf.grid) // n
    return row * cf.cells_per_side + col


def coords_to_r3(coords: torch.Tensor) -> torch.Tensor:
    """(lat, lon) degrees -> unit-sphere xyz. Real: weathergen.datasets.utils.s2tor3."""
    lat, lon = torch.deg2rad(coords[:, 0]), torch.deg2rad(coords[:, 1])
    return torch.stack(
        [torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)], dim=-1
    )


def time_encoding(t: float, n: int) -> torch.Tensor:
    return torch.tensor([math.sin(t), math.cos(t)]).repeat(n, 1)


def tokenize_source(cf: Config, coords: torch.Tensor, data: torch.Tensor, t: float):
    """
    Build source tokens grouped by cell.

    Every point becomes a feature row  [time(2) | xyz(3) | data(C)]  (real tokens also
    carry stream_id and geoinfo channels), points of one cell are grouped, and each group
    of `token_size` consecutive points forms one token.

    Returns tokens [num_cells, tokens_per_cell, token_size, feat].
    """
    feats = torch.cat([time_encoding(t, len(coords)), coords_to_r3(coords), data], dim=-1)
    cells = cell_index(cf, coords)
    order = torch.argsort(cells, stable=True)  # sort points by cell
    feats = feats[order].reshape(cf.num_cells, cf.points_per_cell, -1)
    return feats.reshape(cf.num_cells, cf.tokens_per_cell, cf.token_size, -1)


def cell_neighbours(cf: Config) -> torch.Tensor:
    """
    [num_cells, 9]: each cell plus its 8 neighbours (real: ModelParams.hp_nbours, the
    HEALPix 1-ring). Lat direction clamps at the poles, lon direction wraps.
    """
    n = cf.cells_per_side
    idx = torch.arange(cf.num_cells)
    row, col = idx // n, idx % n
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r = (row + dr).clamp(0, n - 1)
            c = (col + dc) % n
            out.append(r * n + c)
    return torch.stack(out, dim=-1)


# --------------------------------------------------------------------------------------
# Building blocks (real: weathergen/model/attention.py, layers.py)
# --------------------------------------------------------------------------------------
class SelfAttentionBlock(nn.Module):
    """Pre-norm self-attention + MLP. Real: MultiSelfAttentionHead(+Varlen) followed by MLP."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class CrossAttentionBlock(nn.Module):
    """Queries attend to a key/value set. Real: MultiCrossAttentionHead(VarlenSlicedQ)."""

    def __init__(self, dim_q: int, dim_kv: int, heads: int):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim_q)
        self.norm_kv = nn.LayerNorm(dim_kv)
        self.attn = nn.MultiheadAttention(dim_q, heads, kdim=dim_kv, vdim=dim_kv, batch_first=True)
        self.norm2 = nn.LayerNorm(dim_q)
        self.mlp = nn.Sequential(
            nn.Linear(dim_q, 2 * dim_q), nn.GELU(), nn.Linear(2 * dim_q, dim_q)
        )

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        kv = self.norm_kv(kv)
        q = q + self.attn(self.norm_q(q), kv, kv, need_weights=False)[0]
        return q + self.mlp(self.norm2(q))


# --------------------------------------------------------------------------------------
# 1. Embedding engine  (engines.EmbeddingEngine -> StreamEmbedTransformer)
# --------------------------------------------------------------------------------------
class StreamEmbedTransformer(nn.Module):
    """
    One per stream. Lifts a token (token_size points x feat) to a single vector in the
    local latent space: linear per point -> attention across the points of the token ->
    pool -> project to ae_local_dim_embed.
    """

    def __init__(self, cf: Config, feat_dim: int):
        super().__init__()
        self.embed = nn.Linear(feat_dim, cf.embed_dim)
        self.blocks = nn.ModuleList(
            SelfAttentionBlock(cf.embed_dim, cf.embed_heads) for _ in range(cf.embed_blocks)
        )
        self.unembed = nn.Linear(cf.embed_dim, cf.ae_local_dim_embed)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [N, token_size, feat] -> [N, ae_local_dim_embed]
        x = self.embed(tokens)
        for blk in self.blocks:
            x = blk(x)
        return self.unembed(x.mean(dim=1))


# --------------------------------------------------------------------------------------
# 2. Local assimilation engine  (engines.LocalAssimilationEngine)
# --------------------------------------------------------------------------------------
class LocalAssimilationEngine(nn.Module):
    """Self-attention among the tokens *within* one cell (all streams mixed)."""

    def __init__(self, cf: Config):
        super().__init__()
        self.ae_local_blocks = nn.ModuleList(
            SelfAttentionBlock(cf.ae_local_dim_embed, cf.ae_local_num_heads)
            for _ in range(cf.ae_local_num_blocks)
        )

    def forward(self, tokens_c: torch.Tensor) -> torch.Tensor:
        # tokens_c: [num_cells, tokens_per_cell, D_local]; batch dim == cell -> attention
        # is confined to the cell (real code: varlen attention with cell_lens offsets)
        for blk in self.ae_local_blocks:
            tokens_c = blk(tokens_c)
        return tokens_c


# --------------------------------------------------------------------------------------
# 3. Local -> global adapter with learnable queries  (encoder.q_cells +
#    engines.Local2GlobalAssimilationEngine)
# --------------------------------------------------------------------------------------
class Local2GlobalAssimilationEngine(nn.Module):
    """
    Each cell owns `ae_local_num_queries` learnable query vectors. They cross-attend to the
    cell's locally assimilated tokens, producing a fixed-size latent per cell regardless
    of how many observations fell into it.
    """

    def __init__(self, cf: Config):
        super().__init__()
        self.q_cells = nn.Parameter(
            torch.rand(cf.num_cells, cf.ae_local_num_queries, cf.ae_global_dim_embed)
            / cf.ae_global_dim_embed
        )
        self.ae_adapter = CrossAttentionBlock(
            cf.ae_global_dim_embed, cf.ae_local_dim_embed, cf.ae_adapter_num_heads
        )

    def forward(self, tokens_c: torch.Tensor) -> torch.Tensor:
        # tokens_c: [num_cells, tokens_per_cell, D_local] -> [num_cells, Q, D_global]
        return self.ae_adapter(self.q_cells, tokens_c)


# --------------------------------------------------------------------------------------
# 4. Global assimilation engine  (engines.GlobalAssimilationEngine)
# --------------------------------------------------------------------------------------
class GlobalAssimilationEngine(nn.Module):
    """Dense self-attention over all cell latents -> globally consistent state."""

    def __init__(self, cf: Config):
        super().__init__()
        # per-cell positional encoding (real: ModelParams.pe_global / RoPE on cell coords)
        self.pe_global = nn.Parameter(
            torch.randn(cf.num_cells * cf.ae_local_num_queries, cf.ae_global_dim_embed) * 0.02
        )
        self.ae_global_blocks = nn.ModuleList(
            SelfAttentionBlock(cf.ae_global_dim_embed, cf.ae_global_num_heads)
            for _ in range(cf.ae_global_num_blocks)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [B, num_cells*Q, D_global]
        tokens = tokens + self.pe_global
        for blk in self.ae_global_blocks:
            tokens = blk(tokens)
        return tokens


# --------------------------------------------------------------------------------------
# 5. Forecasting engine  (engines.ForecastingEngine)
# --------------------------------------------------------------------------------------
class ForecastingEngine(nn.Module):
    """Advances the latent state by one step. Applied once per forecast step."""

    def __init__(self, cf: Config):
        super().__init__()
        self.fe_blocks = nn.ModuleList(
            SelfAttentionBlock(cf.ae_global_dim_embed, cf.fe_num_heads)
            for _ in range(cf.fe_num_blocks)
        )
        # real code initialises fe weights ~N(0, 1e-3) so step 0 starts near identity
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.normal_(p, std=1e-3)

    def forward(self, tokens: torch.Tensor, step: int) -> torch.Tensor:
        if step == 0:  # step 0 == analysis (no advance), same convention as Model.forward
            return tokens
        for blk in self.fe_blocks:
            tokens = blk(tokens)
        return tokens


# --------------------------------------------------------------------------------------
# 6. Decoder: target coords -> queries -> read out from the latent of the containing cell
#    and its neighbours  (model.embed_target_coords, engines.TargetPredictionEngine,
#    engines.EnsPredictionHead)
# --------------------------------------------------------------------------------------
class TargetPredictionEngine(nn.Module):
    def __init__(self, cf: Config, coord_dim: int):
        super().__init__()
        self.cf = cf
        # stream cfg `embed_target_coords: {net: linear}`
        self.embed_target_coords = nn.Linear(coord_dim, cf.tr_dim_embed)
        # position within the 3x3 neighbourhood (real: TargetPredictionEngine.pos_embed [1,9,D])
        self.pos_embed = nn.Parameter(
            torch.zeros(1, 9 * cf.ae_local_num_queries, cf.ae_global_dim_embed)
        )
        # stream cfg `target_readout: {num_layers, num_heads}`
        self.tte = nn.ModuleList(
            CrossAttentionBlock(cf.tr_dim_embed, cf.ae_global_dim_embed, cf.tr_num_heads)
            for _ in range(cf.tr_num_layers)
        )
        # stream cfg `pred_head: {num_layers, ens_size}`
        self.pred_head = nn.Sequential(
            nn.LayerNorm(cf.tr_dim_embed),
            nn.Linear(cf.tr_dim_embed, cf.tr_dim_embed),
            nn.GELU(),
            nn.Linear(cf.tr_dim_embed, cf.num_channels),
        )

    def forward(
        self, latent: torch.Tensor, target_coords: torch.Tensor, target_cells: torch.Tensor,
        nbours: torch.Tensor,
    ) -> torch.Tensor:
        """
        latent        [num_cells*Q, D_global]  (one sample)
        target_coords [T, coord_dim]           where to predict
        target_cells  [T]                      cell containing each target point
        nbours        [num_cells, 9]           1-ring neighbourhood
        returns       [T, num_channels]
        """
        cf = self.cf
        latent = latent.reshape(cf.num_cells, cf.ae_local_num_queries, -1)
        # gather the 9 neighbouring cells' latents for every target point -> [T, 9*Q, D]
        kv = latent[nbours[target_cells]].flatten(1, 2) + self.pos_embed
        # one query token per target coordinate                        -> [T, 1, D_tr]
        q = self.embed_target_coords(target_coords).unsqueeze(1)
        for blk in self.tte:
            q = blk(q, kv)
        return self.pred_head(q.squeeze(1))


# --------------------------------------------------------------------------------------
# The model  (model.Model)
# --------------------------------------------------------------------------------------
class MiniWeatherGenerator(nn.Module):
    def __init__(self, cf: Config, feat_dim: int, coord_dim: int):
        super().__init__()
        self.cf = cf
        self.embed_engine = StreamEmbedTransformer(cf, feat_dim)
        self.ae_local_engine = LocalAssimilationEngine(cf)
        self.ae_local_global_engine = Local2GlobalAssimilationEngine(cf)
        self.ae_global_engine = GlobalAssimilationEngine(cf)
        self.forecast_engine = ForecastingEngine(cf)
        self.decoder = TargetPredictionEngine(cf, coord_dim)
        self.register_buffer("hp_nbours", cell_neighbours(cf))

    def encode(self, source_tokens: torch.Tensor, verbose: bool) -> torch.Tensor:
        """EncoderModule.forward: embed -> local AE -> adapter -> global AE."""
        cf = self.cf
        log = print if verbose else (lambda *a, **k: None)

        log(f"  source tokens                {tuple(source_tokens.shape)}  "
            "[cells, tokens/cell, token_size, feat]")

        x = self.embed_engine(source_tokens.flatten(0, 1))
        x = x.reshape(cf.num_cells, cf.tokens_per_cell, -1)
        log(f"  1 embed                      {tuple(x.shape)}  [cells, tokens/cell, D_local]")

        x = self.ae_local_engine(x)
        log(f"  2 local assimilation         {tuple(x.shape)}  (attention within a cell)")

        x = self.ae_local_global_engine(x)
        log(f"  3 local->global (q_cells)    {tuple(x.shape)}  [cells, Q, D_global]")

        x = x.flatten(0, 1).unsqueeze(0)  # [1, cells*Q, D_global]
        x = self.ae_global_engine(x)
        log(f"  4 global assimilation        {tuple(x.shape)}  [B, cells*Q, D_global]")
        return x

    def forward(self, source_tokens, target_coords, target_cells, verbose: bool = False):
        """Model.forward: encode once, then roll out in latent space and decode per step.

        target_coords: one [T, coord_dim] tensor per output step (time encoding differs).
        """
        log = print if verbose else (lambda *a, **k: None)
        tokens = self.encode(source_tokens, verbose)

        preds = []
        for step in range(self.cf.forecast_steps + 1):
            tokens = self.forecast_engine(tokens, step)
            log(f"  5 forecast engine step {step}    {tuple(tokens.shape)}")
            pred = self.decoder(tokens[0], target_coords[step], target_cells, self.hp_nbours)
            log(f"  6 decode step {step}             {tuple(pred.shape)}  [targets, channels]")
            preds.append(pred)
        return preds


# --------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=0, help="optimizer steps after the forward pass")
    args = ap.parse_args()

    cf = Config()
    coords = make_grid(cf)  # [4096, 2]
    t0, dt = 0.0, 0.3

    # source: field at t0, tokenized per cell. target: field at t0 + k*dt on the same grid.
    src_field = make_field(coords, t0, cf.num_channels)  # 64x64x3
    source_tokens = tokenize_source(cf, coords, src_field, t0)
    targets = [
        make_field(coords, t0 + k * dt, cf.num_channels) for k in range(cf.forecast_steps + 1)
    ]

    # target coordinates as the decoder sees them: xyz + time encoding of the target step
    target_cells = cell_index(cf, coords)
    target_coords = [
        torch.cat([coords_to_r3(coords), time_encoding(t0 + k * dt, len(coords))], dim=-1)
        for k in range(cf.forecast_steps + 1)
    ]

    model = MiniWeatherGenerator(
        cf, feat_dim=source_tokens.shape[-1], coord_dim=target_coords[0].shape[-1]
    )
    n_params = sum(p.numel() for p in model.parameters())

    print(f"grid {cf.grid}x{cf.grid}x{cf.num_channels} -> {cf.num_cells} cells x "
          f"{cf.tokens_per_cell} tokens x {cf.token_size} points;  {n_params:,} parameters\n")
    print("forward pass:")
    with torch.no_grad():
        preds = model(source_tokens, target_coords, target_cells, verbose=True)
    loss = sum(F.mse_loss(p, t) for p, t in zip(preds, targets, strict=True)) / len(preds)
    print(f"\nMSE vs. truth (untrained): {loss.item():.4f}")

    if args.steps:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        for i in range(args.steps):
            preds = model(source_tokens, target_coords, target_cells)
            loss = sum(F.mse_loss(p, t) for p, t in zip(preds, targets, strict=True)) / len(preds)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if i % 10 == 0 or i == args.steps - 1:
                print(f"step {i:4d}  loss {loss.item():.4f}")


if __name__ == "__main__":
    main()
