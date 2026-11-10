# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import warnings

import astropy_healpix as hp
import numpy as np
import torch

from weathergen.datasets.utils import (
    healpix_verts,
    healpix_verts_rots,
)


class Tokenizer:
    """
    Base class for tokenizers.
    """

    def __init__(self, hl_source: int, hl_target: int = None, grid_source=None, grid_target=None):
        ref = torch.tensor([1.0, 0.0, 0.0])

        # Sources are tokenized on the encoder's own grid (hl_source / grid_source); targets on the
        # forecast grid (hl_target / grid_target). When the target level/grid is omitted it mirrors
        # the source, recovering the single-grid behaviour. Per-cell geometry is built for the
        # *active* cells of each level: full-globe arrays scale as 12 * 4**hl regardless of region
        # size, which is 96 GB for hpy_verts_local_target alone at level 12.
        self.hl_source = hl_source
        self.hl_target = hl_target if hl_target is not None else hl_source
        self.grid_source = grid_source
        self.grid_target = grid_target if grid_target is not None else grid_source
        # source level/grid drive source tokenization in get_tokens_windows
        self.healpix_level = self.hl_source
        self.grid = self.grid_source

        self.num_healpix_cells_source = (
            self.grid_source.num_cells if self.grid_source is not None else 12 * 4**self.hl_source
        )
        self.num_healpix_cells_target = (
            self.grid_target.num_cells if self.grid_target is not None else 12 * 4**self.hl_target
        )

        self.size_time_embedding = 6

        # global nested ids of each grid's active cells; None means "whole globe", which is also
        # what a full grid resolves to. Suffix _s = source (encoder) grid, _t = target (forecast).
        cells_s = (
            None
            if self.grid_source is None or self.grid_source.is_full
            else self.grid_source.active_to_global
        )
        cells_t = (
            None
            if self.grid_target is None or self.grid_target.is_full
            else self.grid_target.active_to_global
        )

        verts00_s, verts00_rots_s = healpix_verts_rots(self.hl_source, 0.0, 0.0, cells=cells_s)
        verts10_s, verts10_rots_s = healpix_verts_rots(self.hl_source, 1.0, 0.0, cells=cells_s)
        verts11_s, verts11_rots_s = healpix_verts_rots(self.hl_source, 1.0, 1.0, cells=cells_s)
        verts01_s, verts01_rots_s = healpix_verts_rots(self.hl_source, 0.0, 1.0, cells=cells_s)
        vertsmm_s, vertsmm_rots_s = healpix_verts_rots(self.hl_source, 0.5, 0.5, cells=cells_s)
        self.hpy_verts_rots_source = [
            verts00_rots_s.to(torch.float32),
            verts10_rots_s.to(torch.float32),
            verts11_rots_s.to(torch.float32),
            verts01_rots_s.to(torch.float32),
            vertsmm_rots_s.to(torch.float32),
        ]

        verts00_t, verts00_rots_t = healpix_verts_rots(self.hl_target, 0.0, 0.0, cells=cells_t)
        verts10_t, verts10_rots_t = healpix_verts_rots(self.hl_target, 1.0, 0.0, cells=cells_t)
        verts11_t, verts11_rots_t = healpix_verts_rots(self.hl_target, 1.0, 1.0, cells=cells_t)
        verts01_t, verts01_rots_t = healpix_verts_rots(self.hl_target, 0.0, 1.0, cells=cells_t)
        vertsmm_t, vertsmm_rots_t = healpix_verts_rots(self.hl_target, 0.5, 0.5, cells=cells_t)
        self.hpy_verts_rots_target = [
            verts00_rots_t.to(torch.float32),
            verts10_rots_t.to(torch.float32),
            verts11_rots_t.to(torch.float32),
            verts01_rots_t.to(torch.float32),
            vertsmm_rots_t.to(torch.float32),
        ]

        # local coords of each target cell's vertices/centre in the frame of each of the others
        transforms = [
            ([verts10_t, verts11_t, verts01_t, vertsmm_t], verts00_rots_t),
            ([verts00_t, verts11_t, verts01_t, vertsmm_t], verts10_rots_t),
            ([verts00_t, verts10_t, verts01_t, vertsmm_t], verts11_rots_t),
            ([verts00_t, verts11_t, verts10_t, vertsmm_t], verts01_rots_t),
            ([verts00_t, verts10_t, verts11_t, verts01_t], vertsmm_rots_t),
        ]

        self.verts_local = []
        for _verts, rot in transforms:
            # Compute local coordinates
            verts = torch.stack(_verts)
            # shape: <healpix, 4, 3>
            verts = verts.transpose(0, 1)
            # Batch multiplication by the 3x3 rotation matrices.
            # shape: <healpix, 3, 3> @ <healpix, 4, 3> -> <healpix, 4, 3>
            # Needs to transpose first to <healpix, 3, 4> then transpose back.
            t1 = torch.bmm(rot, verts.transpose(-1, -2)).transpose(-2, -1)
            t2 = ref - t1
            self.verts_local.append(t2.flatten(1, 2))

        self.hpy_verts_local_target = torch.stack(self.verts_local).transpose(0, 1)

        # add local coords wrt to center of neighboring cells
        # (since the neighbors are used in the prediction)
        ids_target = np.arange(12 * 4**self.hl_target) if cells_t is None else np.asarray(cells_t)
        with warnings.catch_warnings(action="ignore"):
            temp = hp.neighbours(ids_target, 2**self.hl_target, order="nested").transpose()
        # fix missing nbors with references to self
        for i, row in enumerate(temp):
            temp[i][row == -1] = ids_target[i]
        # a neighbour of an active cell need not itself be active, so its centre cannot be
        # gathered from vertsmm_t (which holds active cells only) and is computed here
        self.hpy_nctrs_target = (
            healpix_verts(self.hl_target, 0.5, 0.5, cells=temp.flatten())
            .reshape((len(ids_target), 8, 3))
            .transpose(1, 0)
            .to(torch.float32)
        )

    def get_size_time_embedding(self) -> int:
        """
        Get size of time embedding
        """
        return self.size_time_embedding
