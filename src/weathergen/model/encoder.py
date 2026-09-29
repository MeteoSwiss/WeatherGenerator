# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, NamedTuple

import torch
from astropy_healpix import healpy
from torch.distributed.tensor import DTensor, distribute_tensor
from torch.utils.checkpoint import checkpoint

from weathergen.common.config import Config
from weathergen.datasets.batch import BatchSamples
from weathergen.model.engines import (
    EmbeddingEngine,
    GlobalAssimilationEngine,
    Local2GlobalAssimilationEngine,
    Local2GlobalSumEngine,
    LocalAssimilationEngine,
    QueryAggregationEngine,
)
from weathergen.model.parametrised_prob_dist import LatentInterpolator
from weathergen.model.positional_encoding import positional_encoding_harmonic

if TYPE_CHECKING:
    from weathergen.model.model import ModelParams


class EncoderOutput(NamedTuple):
    healpix_level: int
    patch_tokens: torch.Tensor
    auxiliary_tokens: torch.Tensor
    coverage: torch.Tensor
    posteriors: list


class EncoderBase(torch.nn.Module, ABC):
    encoder_name: str
    healpix_level: int

    @abstractmethod
    def forward(self, model_params: ModelParams, batch: BatchSamples) -> EncoderOutput:
        """Encode one named, single-grid input batch."""
        raise NotImplementedError


class EncoderModule(EncoderBase):
    name: EncoderModule

    def __init__(self, cf: Config, sources_size, encoder_name: str) -> None:
        """Create an encoder from its scoped configuration and ordered source sizes."""
        super().__init__()
        self.cf = cf
        self.encoder_name = encoder_name
        self.healpix_level = cf.healpix_level
        self.num_healpix_cells = 12 * 4**self.healpix_level
        self.sources_size = sources_size
        self.sharded_training = False

        self.ae_aggregation_engine: QueryAggregationEngine | None = None
        self.ae_global_engine: GlobalAssimilationEngine | None = None
        self.ae_local_engine: LocalAssimilationEngine | None = None
        self.ae_local_global_engine: Local2GlobalAssimilationEngine | None = None
        self.embed_engine: EmbeddingEngine | None = None
        self.interpolator_latents: LatentInterpolator | None = None

        # embedding engine
        # determine stream names once so downstream components use consistent keys
        self.stream_names = list(cf.streams.keys())
        # separate embedding networks for differnt observation types
        self.embed_engine = EmbeddingEngine(cf, self.sources_size)

        assert cf.ae_global_att_dense_rate == 1.0, "Local attention not adapted for register tokens"
        self.num_register_tokens = cf.num_register_tokens
        self.num_class_tokens = cf.num_class_tokens

        # local assimilation engine
        self.ae_local_engine = LocalAssimilationEngine(cf)

        if cf.latent_noise_kl_weight > 0.0:
            self.interpolator_latents = LatentInterpolator(
                gamma=cf.latent_noise_gamma,
                dim=cf.ae_local_dim_embed,
                use_additive_noise=cf.latent_noise_use_additive_noise,
                deterministic=cf.latent_noise_deterministic_latents,
            )

        # local -> global assimilation engine adapter
        ae_adapter_type = cf.get("ae_adapter_type", "cross_attention")
        if ae_adapter_type == "sum":
            self.ae_local_global_engine = Local2GlobalSumEngine(cf)
        else:
            self.ae_local_global_engine = Local2GlobalAssimilationEngine(cf)

        self.q_cells = torch.nn.Parameter(
            self._create_queries(), requires_grad="q_cells" not in cf.get("freeze_modules", [])
        )

        # query aggregation engine
        self.ae_aggregation_engine = QueryAggregationEngine(cf, self.num_healpix_cells)

        # global assimilation engine
        self.ae_global_engine = GlobalAssimilationEngine(cf, self.num_healpix_cells)

    def _create_queries(self, device=None, dtype=None):
        cf = self.cf
        num_cells = self.num_healpix_cells if cf.ae_local_queries_per_cell else 1
        queries = (
            torch.rand(
                (num_cells, cf.ae_local_num_queries, cf.ae_global_dim_embed),
                device=device,
                dtype=dtype,
            )
            / cf.ae_global_dim_embed
        )
        if cf.ae_local_queries_per_cell and queries.device.type != "meta":
            cells = torch.arange(num_cells, device=queries.device)
            queries[:, :, -8:-6] = (cells / num_cells)[:, None, None]
            theta, phi = healpy.pix2ang(
                nside=2**self.healpix_level, ipix=torch.arange(num_cells, device="cpu"), nest=True
            )
            queries[:, :, -6:-3] = torch.as_tensor(
                theta, device=queries.device, dtype=queries.dtype
            ).cos()[:, None, None]
            queries[:, :, -3:] = torch.as_tensor(
                phi, device=queries.device, dtype=queries.dtype
            ).sin()[:, None, None]
            query_ids = torch.arange(cf.ae_local_num_queries, device=queries.device)
            queries[:, :, -9] = query_ids
            queries[:, :, -10] = query_ids
        return queries

    @torch.no_grad()
    def reset_queries(self):
        """Initialize queries after meta materialization, including sharded parameters."""
        queries = self._create_queries(device=self.q_cells.device, dtype=self.q_cells.dtype)
        if isinstance(self.q_cells, DTensor):
            queries = distribute_tensor(queries, self.q_cells.device_mesh, self.q_cells.placements)
        self.q_cells.copy_(queries)

    def forward(self, model_params: ModelParams, batch: BatchSamples) -> EncoderOutput:
        if batch.encoder_name != self.encoder_name or batch.healpix_level != self.healpix_level:
            raise ValueError(f"Input batch does not match encoder {self.encoder_name!r}")
        num_steps, batch_size = batch.get_num_source_steps(), len(batch)
        coverage = batch.coverage
        if coverage is None or coverage.shape != (num_steps, batch_size, self.num_healpix_cells):
            raise ValueError(f"Invalid coverage axes for encoder {self.encoder_name!r}")

        dependency = None
        if self.sharded_training and self.training and torch.is_grad_enabled():
            dependency = self.q_cells.new_zeros(())
            for parameter in self.parameters():
                if parameter.requires_grad:
                    dependency = dependency + parameter[(0,) * parameter.ndim] * 0

        num_aux = self.num_register_tokens + self.num_class_tokens
        num_queries, dim_embed = self.q_cells.shape[-2:]
        if not coverage.any():
            zero = self.q_cells.new_zeros(()) if dependency is None else dependency
            return EncoderOutput(
                self.healpix_level,
                zero.expand(batch_size, self.num_healpix_cells, num_queries, dim_embed),
                zero.expand(batch_size, num_aux, dim_embed),
                coverage.any(dim=0),
                [],
            )

        stream_cell_tokens = checkpoint(
            self.embed_engine, batch, model_params.pe_embed, use_reentrant=False
        )
        tokens_global, posteriors = checkpoint(
            self.assimilate_local,
            model_params,
            stream_cell_tokens,
            batch,
            dependency,
            use_reentrant=False,
        )
        # ponytail: dense global queries reach 196,608 cells at level 7; packing is separate work.
        tokens_global = checkpoint(
            self.ae_global_engine,
            tokens_global,
            coords=model_params.rope_coords,
            use_reentrant=False,
        ).reshape(num_steps, batch_size, -1, dim_embed)

        patches = tokens_global[:, :, num_aux:].reshape(
            num_steps, batch_size, self.num_healpix_cells, num_queries, dim_embed
        )
        patches = patches.masked_fill(~coverage[..., None, None], 0).sum(dim=0)
        auxiliary = tokens_global[:, :, :num_aux]
        auxiliary = auxiliary.masked_fill(~coverage.any(dim=-1)[..., None, None], 0).sum(dim=0)
        return EncoderOutput(
            self.healpix_level, patches, auxiliary, coverage.any(dim=0), posteriors
        )

    def interpolate_latents(self, tokens: torch.Tensor):
        """Optionally sample the local latent distribution."""
        if self.cf.latent_noise_kl_weight > 0.0:
            tokens, posteriors = self.interpolator_latents.interpolate_with_noise(
                tokens, sampling=self.training
            )
        else:
            posteriors = torch.zeros((1,), device=tokens.device)

        return tokens, posteriors

    def assimilate_local_project_chunked(self, tokens, tokens_global, cell_lens, q_cells_lens):
        """
        Apply the local assimilation engine and then the
        local-to-global adapter using a chunking in the number of tokens
        to work around to bug in flash attention, the computations is performed in chunks
        """

        # combined cell lens for all tokens in batch across all input steps
        zero_pad = torch.zeros(1, device=tokens.device, dtype=torch.int32)

        # subdivision factor for required splitting
        clen = self.num_healpix_cells // (2 if self.healpix_level <= 5 else 8)
        tokens_global_unmasked = []
        posteriors = []

        for i in range(cell_lens.shape[0] // clen):
            # make sure we properly catch all elements in last chunk
            i_end = (i + 1) * clen if i < (cell_lens.shape[0] // clen) - 1 else cell_lens.shape[0]
            l0, l1 = (
                (0 if i == 0 else cell_lens[: i * clen].cumsum(0)[-1]),
                cell_lens[:i_end].cumsum(0)[-1],
            )

            toks = tokens[l0:l1]
            # if we have a very sparse input, we may have no tokens in the chunk, toks
            # skip processing of the empty chunk in this case
            # Check if this chunk is empty
            if l0 == l1 or toks.shape[0] == 0:
                continue

            toks_global = tokens_global[i * clen : i_end]
            cell_lens_cur = torch.cat([zero_pad, cell_lens[i * clen : i_end]])
            q_cells_lens_cur = q_cells_lens[: cell_lens_cur.shape[0]]

            # local assimilation model
            toks = self.ae_local_engine(toks, cell_lens_cur, use_reentrant=False)

            toks, posteriors_c = self.interpolate_latents(toks)
            posteriors += [posteriors_c]

            # create mask for global tokens, without first element (used for padding)
            mask = cell_lens_cur[1:].to(torch.bool)
            toks_global_unmasked = toks_global[mask]
            q_cells_lens_unmasked = torch.cat([zero_pad, q_cells_lens_cur[1:][mask]])
            cell_lens_unmasked = torch.cat([zero_pad, cell_lens_cur[1:][mask]])

            # local to global adapter engine
            toks_global_unmasked = self.ae_local_global_engine(
                toks,
                toks_global_unmasked,
                q_cells_lens_unmasked,
                cell_lens_unmasked,
            )

            tokens_global_unmasked += [toks_global_unmasked]

        if not tokens_global_unmasked:
            return tokens_global[:0], posteriors
        tokens_global_unmasked = torch.cat(tokens_global_unmasked)

        return tokens_global_unmasked, posteriors

    def assimilate_local(
        self,
        model_params: ModelParams,
        tokens: torch.Tensor,
        batch: BatchSamples,
        dependency: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, list]:
        """Assimilate observed cells, retaining learned queries for masked cells."""
        cell_lens = batch.tokens_lens.sum(dim=2).flatten()
        rs = batch.get_num_source_steps() * len(batch)
        num_aux = self.num_register_tokens + self.num_class_tokens
        num_queries, dim_embed = self.q_cells.shape[-2:]

        # Inject the sharded gradient dependency before expanding dense spatial queries.
        queries = self.q_cells if dependency is None else self.q_cells + dependency
        auxiliary = positional_encoding_harmonic(queries[:1, :1].expand(rs, num_aux, dim_embed))
        patches = (queries + model_params.pe_global).repeat(rs, 1, 1)
        posteriors = []
        if tokens.numel():
            unmasked, posteriors = self.assimilate_local_project_chunked(
                tokens, patches, cell_lens, model_params.q_cells_lens
            )

        tokens_global = torch.cat(
            [auxiliary, patches.reshape(rs, self.num_healpix_cells * num_queries, dim_embed)],
            dim=1,
        )
        if not tokens.numel():
            return tokens_global, posteriors

        cell_mask = cell_lens.reshape(rs, self.num_healpix_cells).to(torch.bool)
        active = cell_mask.any(dim=-1)
        spatial_mask = cell_mask.repeat_interleave(num_queries, dim=-1)
        tokens_global[:, num_aux:][spatial_mask] = unmasked.flatten(0, 1).to(tokens_global.dtype)
        mask = torch.cat(
            [
                active[:, None].expand(-1, num_aux),
                spatial_mask,
            ],
            dim=1,
        )
        # Do not send empty step/view sequences to variable-length attention.
        batch_lens = mask.sum(dim=-1)[active]
        batch_lens = torch.cat([batch_lens.new_zeros(1), batch_lens])
        coords = model_params.rope_coords
        packed_coords = None if coords is None else coords.expand(rs, -1, -1)[mask]
        aggregated = self.ae_aggregation_engine(
            tokens_global[mask], batch_lens, use_reentrant=False, coords=packed_coords
        )
        tokens_global[mask] = aggregated.to(tokens_global.dtype)
        return tokens_global, posteriors
