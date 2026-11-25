import numpy as np
import pandas as pd
import torch
from astropy_healpix.healpy import ang2pix
from torch import Tensor

from weathergen.common.io import IOReaderData
from weathergen.datasets.utils import (
    r3tos2,
    s2tor3,
    vecs_to_rots,
)

# on some clusters our numpy version is pinned to be 1.x.x where the np.argsort does not
# the stable=True argument
numpy_argsort_args = {"stable": True} if int(np.__version__.split(".")[0]) >= 2 else {}


def theta_phi_to_standard_coords(coords):
    thetas = ((90.0 - coords[:, 0]) / 180.0) * np.pi
    phis = ((coords[:, 1] + 180.0) / 360.0) * 2.0 * np.pi

    return thetas, phis


def encode_times_source(times, time_win) -> torch.tensor:
    """Encode times in the format used for source

    Return:
        len(times) x 5
    """
    # assemble tensor as fed to the network, combining geoinfo and data
    fp32 = torch.float32
    dt = pd.to_datetime(times)
    dt_win = pd.to_datetime(time_win)
    dt_delta = dt - dt_win[0]
    year = np.atleast_1d(dt.year)
    dayofyear = np.atleast_1d(dt.dayofyear)
    minutes = np.atleast_1d(dt.hour * 60 + dt.minute)
    delta_seconds = np.atleast_1d(dt_delta.seconds)
    time_tensor = torch.cat(
        (
            torch.tensor(year, dtype=fp32).unsqueeze(1),
            torch.tensor(dayofyear, dtype=fp32).unsqueeze(1),
            torch.tensor(minutes, dtype=fp32).unsqueeze(1),
            torch.tensor(delta_seconds, dtype=fp32).unsqueeze(1),
            torch.tensor(delta_seconds, dtype=fp32).unsqueeze(1),
        ),
        1,
    )

    # normalize
    time_tensor[..., 0] /= 2100.0
    time_tensor[..., 1] = time_tensor[..., 1] / 365.0
    time_tensor[..., 2] = time_tensor[..., 2] / 1440.0
    time_tensor[..., 3] = np.sin(time_tensor[..., 3] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 4] = np.cos(time_tensor[..., 4] / (12.0 * 3600.0) * 2.0 * np.pi)

    return time_tensor


def encode_times_target(times, time_win) -> torch.tensor:
    """Encode times in the format used for target (relative time in window)

    Return:
        len(times) x 5
    """
    dt = pd.to_datetime(times)
    dt_win = pd.to_datetime(time_win)
    # for target only provide local time
    dt_delta = torch.tensor(np.atleast_1d((dt - dt_win[0]).seconds), dtype=torch.float32).unsqueeze(
        1
    )
    time_tensor = torch.cat(
        (
            dt_delta,
            dt_delta,
            dt_delta,
            dt_delta,
            dt_delta,
        ),
        1,
    )

    # normalize
    time_tensor[..., 0] = np.sin(time_tensor[..., 0] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 1] = np.cos(time_tensor[..., 1] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 2] = np.sin(time_tensor[..., 2] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 3] = np.cos(time_tensor[..., 3] / (12.0 * 3600.0) * 2.0 * np.pi)
    time_tensor[..., 4] = np.sin(time_tensor[..., 4] / (12.0 * 3600.0) * 2.0 * np.pi)

    # We add + 0.5 as for datasets with regular time steps we otherwise very often get 0 as the
    # first time and to prevent too many zeros in the input
    return time_tensor + 0.5


def hpy_cell_splits(coords: torch.tensor, hl: int):
    """Compute healpix cell id for each coordinate on given level hl

    Returns
      hpy_idxs_ord_split : list of per cell indices into thetas,phis,posr3
      thetas : thetas in rad
      phis : phis in rad
    """
    cells, groups, thetas, phis = _occupied_cell_groups(coords, hl)

    hpy_idxs_ord_split = [np.array([], dtype=np.int64) for _ in range(12 * 4**hl)]
    for b, x in zip(cells, groups, strict=True):
        hpy_idxs_ord_split[b] = x

    return (hpy_idxs_ord_split, thetas, phis)


def _occupied_cell_groups(coords: torch.tensor, hl: int):
    """Healpix cell of each point, grouped by cell, for the occupied cells only.

    Returns
      cells : sorted ids of the cells that contain at least one point
      groups : for each of these cells, the point indices (sorted by cell, stable)
      thetas, phis : coordinates in rad
    """
    thetas, phis = theta_phi_to_standard_coords(coords)
    # healpix cells for all points
    hpy_idxs = ang2pix(2**hl, thetas, phis, nest=True)

    # sort points by cell and split where the cell changes
    hpy_idxs_ord = np.argsort(hpy_idxs, **numpy_argsort_args)
    cells, starts = np.unique(hpy_idxs[hpy_idxs_ord], return_index=True)
    groups = np.split(hpy_idxs_ord, starts[1:]) if len(cells) > 0 else []

    return cells, groups, thetas, phis


def _hpy_splits_occupied(
    coords: torch.Tensor, hl: int, token_size: int, pad_tokens: bool, offset_step: int = 0
):
    """Token index lists for the occupied cells only (see hpy_splits).

    Only cells that contain points are processed, so the cost does not grow with the total
    number of healpix cells (12 * 4**hl), which matters for limited-area data at high levels.

    Returns
      cells : ids of the occupied cells
      idxs_ord : per occupied cell, the list of token index tensors
      idxs_ord_lens : per occupied cell, the list of token lengths
    """
    cells, groups, thetas, _ = _occupied_cell_groups(coords, hl)

    # pad to token size *and* offset by +1 to account for the index 0 that is added for the padding
    offset = (1 if pad_tokens else 0) + offset_step
    int32 = torch.int32
    idxs_ord, idxs_ord_lens = [], []
    for idxs in groups:
        # if token_size is exceeed split based on latitude
        # TODO: split by hierarchically traversing healpix scheme
        ts = torch.argsort(thetas[idxs], stable=True)
        # remainder for padding to token size
        if pad_tokens:
            n = len(idxs)
            r = token_size - (n % token_size if n % token_size != 0 else token_size)
        else:
            r = 0
        toks = list(
            torch.split(
                torch.cat(
                    (torch.from_numpy(np.take(idxs, ts) + offset), torch.zeros(r, dtype=int32))
                ),
                token_size,
            )
        )
        idxs_ord.append(toks)
        idxs_ord_lens.append([len(a) for a in toks])

    return cells, idxs_ord, idxs_ord_lens


def hpy_splits(
    coords: torch.Tensor, hl: int, token_size: int, pad_tokens: bool, offset_step: int = 0
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor]:
    """Compute healpix cell for each data point and splitting information per cell;
       when the token_size is exceeded then splitting based on lat is used;
       tokens can be padded

    Return :
        idxs_ord : flat list of indices (to data points) per healpix cell
        idxs_ord_lens : lens of lists per cell
        (so that data[idxs_ord].split( idxs_ord_lens) provides per cell data)
    """
    num_healpix_cells = 12 * 4**hl
    idxs_ord = [[] for _ in range(num_healpix_cells)]
    idxs_ord_lens = [[] for _ in range(num_healpix_cells)]

    cells, toks, lens = _hpy_splits_occupied(coords, hl, token_size, pad_tokens, offset_step)
    for c, t, t_lens in zip(cells, toks, lens, strict=True):
        idxs_ord[c] = t
        idxs_ord_lens[c] = t_lens

    return idxs_ord, idxs_ord_lens


def tokenize_space(
    rdata,
    token_size,
    hl,
    pad_tokens=True,
    offset_step=0,
):
    """Process one window into tokens"""

    # idx_ord_lens is length is number of tokens per healpix cell
    idxs_ord, idxs_ord_lens = hpy_splits(rdata.coords, hl, token_size, pad_tokens, offset_step)

    return idxs_ord, idxs_ord_lens


def tokenize_spacetime(
    rdata,
    token_size,
    hl,
    pad_tokens=True,
):
    """Tokenize respecting an intrinsic time step in the data, i.e. each time step is tokenized
    separately
    """

    num_healpix_cells = 12 * 4**hl
    idxs_cells = [[] for _ in range(num_healpix_cells)]
    idxs_cells_lens = [[] for _ in range(num_healpix_cells)]

    offset_step = 0
    t_unique = np.unique(rdata.datetimes)
    for _, t in enumerate(t_unique):
        # data for current time step
        mask = t == rdata.datetimes
        rdata_cur = IOReaderData(
            rdata.coords[mask], rdata.geoinfos[mask], rdata.data[mask], rdata.datetimes[mask]
        )
        cells, idxs_cur, idxs_cur_lens = _hpy_splits_occupied(
            rdata_cur.coords, hl, token_size, pad_tokens, offset_step
        )

        # collect data for all time steps (only the cells that have data at this step)
        for c, tc, tc_l in zip(cells, idxs_cur, idxs_cur_lens, strict=True):
            idxs_cells[c] = idxs_cells[c] + tc
            idxs_cells_lens[c] = idxs_cells_lens[c] + tc_l
        offset_step += mask.sum()

    return idxs_cells, idxs_cells_lens


def _masked_counts_per_cell(idxs_cells_lens, mask_tokens):
    """Per cell: number of kept tokens and number of kept data points (incl. token padding).

    Vectorised over tokens; equivalent to looping over all cells, which is slow for
    12 * 4**hl cells at high healpix levels.

    Returns
      tokens_per_cell : int64 tensor [num_cells]
      masked_points_per_cell : int32 tensor [num_cells]
    """
    num_cells = len(idxs_cells_lens)
    tokens_in_cell = np.fromiter((len(t) for t in idxs_cells_lens), np.int64, num_cells)
    token_cell = np.repeat(np.arange(num_cells), tokens_in_cell)
    token_len = np.fromiter((n for t in idxs_cells_lens for n in t), np.int64, token_cell.shape[0])
    keep = np.asarray(mask_tokens, dtype=bool)

    kept_cells = token_cell[keep]
    tokens_per_cell = np.bincount(kept_cells, minlength=num_cells)
    points_per_cell = np.bincount(kept_cells, weights=token_len[keep], minlength=num_cells)

    return (
        torch.from_numpy(tokens_per_cell.astype(np.int64)),
        torch.from_numpy(points_per_cell.astype(np.int32)),
    )


def tokenize_apply_mask_source(
    idxs_cells,
    idxs_cells_lens,
    mask_tokens,
    mask_channels,
    stream_id,
    rdata,
    time_win,
    hpy_verts_rots,
    enc_time,
):
    """
    Apply masking to the data.

    Conceptually, the data is a matrix with the rows corresponding to data points / tokens and
    the cols the channels. Thereby mask_tokens acts on the rows, grouped according to the tokens as
    specified in idxs_cells and mask_channels acts on the columns.

    """

    def return_empty(rdata, idxs_cells_lens):
        tokens_cells = [torch.tensor([])]
        tokens_per_cell = torch.zeros(len(idxs_cells_lens), dtype=torch.int32)
        return tokens_cells, tokens_per_cell

    # convert to token level, forgetting about cells
    idxs_tokens = [i for t in idxs_cells for i in t]
    idxs_lens = [i for t in idxs_cells_lens for i in t]

    # apply spatial masking on a per token level
    if mask_tokens is None:
        return return_empty(rdata, idxs_cells_lens)

    # filter tokens using mask to obtain flat per data point index list
    idxs_data = [t for t, m in zip(idxs_tokens, mask_tokens, strict=True) if m]

    if len(idxs_data) == 0:
        return return_empty(rdata, idxs_cells_lens)

    idxs_data = torch.cat(idxs_data)
    # filter list of token lens using mask and obtain flat list for splitting
    idxs_data_lens = torch.tensor([t for t, m in zip(idxs_lens, mask_tokens, strict=True) if m])

    # pad with zero at the begining of the conceptual 2D data tensor:
    # idxs_cells -> idxs_tokens -> idxs_data has been prepared so
    # that the zero-index is used to add the padding to the tokens to ensure fixed size
    times_enc = enc_time(rdata.datetimes, time_win)
    zeros_like = torch.zeros_like
    datetimes_enc_padded = torch.cat([zeros_like(times_enc[0]).unsqueeze(0), times_enc])
    geoinfos_padded = torch.cat([zeros_like(rdata.geoinfos[0]).unsqueeze(0), rdata.geoinfos])
    coords_padded = torch.cat([zeros_like(rdata.coords[0]).unsqueeze(0), rdata.coords])
    data_padded = torch.cat([zeros_like(rdata.data[0]).unsqueeze(0), rdata.data])

    # apply mask
    datetimes = datetimes_enc_padded[idxs_data]
    geoinfos = geoinfos_padded[idxs_data]
    coords = coords_padded[idxs_data]
    data = data_padded[idxs_data]

    if mask_channels is not None:
        assert False, "to be implemented"
        # data = data_padded[ : channel_mask]

    # local coords
    tokens_per_cell, masked_points_per_cell = _masked_counts_per_cell(idxs_cells_lens, mask_tokens)
    coords_local = get_source_coords_local(coords, hpy_verts_rots, masked_points_per_cell)

    # create tensor that contains all data
    stream_ids = torch.full([len(datetimes), 1], stream_id, dtype=torch.float32)
    tokens = torch.cat((stream_ids, datetimes, coords_local, geoinfos, data), 1)

    # split up tensor into tokens
    # TODO: idxs_data_lens is currently only defined when mask_tokens is not None
    idxs_data_lens = idxs_data_lens.tolist()
    tokens_cells = torch.split(tokens, idxs_data_lens)

    return tokens_cells, tokens_per_cell


def tokenize_apply_mask_target(
    stream_id,
    hl,
    idxs_cells,
    idxs_cells_lens,
    mask_tokens,
    mask_channels,
    rdata,
    time_win,
    hpy_verts_rots,
    hpy_verts_local,
    hpy_nctrs,
    enc_time,
):
    """
    Apply masking to the data.

    Conceptually, the data is a matrix with the rows corresponding to data points / tokens and
    the cols the channels. Thereby mask_tokens acts on the rows, grouped according to the tokens as
    specified in idxs_cells and mask_channels acts on the columns.

    """

    def return_empty(rdata, idxs_cells_lens):
        do = torch.zeros([0, rdata.data.shape[-1]])
        coords = torch.zeros([0, rdata.coords.shape[-1]])
        dt = np.array([], dtype=np.datetime64)
        masked_points_per_cell = torch.zeros(len(idxs_cells_lens), dtype=torch.int32)
        # data, datetimes, coords, coords_local, masked_points_per_cell
        return do, dt, coords, coords, masked_points_per_cell

    # convert to token level, forgetting about cells
    idxs_tokens = [i for t in idxs_cells for i in t]
    idxs_lens = [i for t in idxs_cells_lens for i in t]

    # apply spatial masking on a per token level
    if mask_tokens is None:
        return return_empty(rdata, idxs_cells_lens)

    # filter tokens using mask to obtain flat per data point index list
    idxs_data = [t for t, m in zip(idxs_tokens, mask_tokens, strict=True) if m]

    if len(idxs_data) == 0:
        return return_empty(rdata, idxs_cells_lens)

    idxs_data = torch.cat(idxs_data)

    # apply mask
    datetimes = np.atleast_1d(rdata.datetimes[idxs_data])
    datetimes_enc = enc_time(datetimes, time_win)
    geoinfos = rdata.geoinfos[idxs_data]
    coords = rdata.coords[idxs_data]
    data = rdata.data[idxs_data]

    if mask_channels is not None:
        assert False, "to be implemented"
        # data = data_padded[ : channel_mask]

    _, masked_points_per_cell = _masked_counts_per_cell(idxs_cells_lens, mask_tokens)

    # compute encoding of target coordinates used in prediction network
    if torch.tensor(idxs_lens).sum() > 0:
        coords_local = get_target_coords_local(
            stream_id,
            hl,
            masked_points_per_cell,
            coords,
            geoinfos,
            datetimes_enc,
            hpy_verts_rots,
            hpy_verts_local,
            hpy_nctrs,
        )
        coords_local.requires_grad = False
    else:
        coords_local = torch.tensor([])

    return data, datetimes, coords, coords_local, masked_points_per_cell


def get_source_coords_local(
    coords: Tensor,
    hpy_verts_rots: Tensor,
    masked_points_per_cell,
) -> list[Tensor]:
    """Compute simple local coordinates for a set of 3D positions on the unit sphere."""

    # remove padding from coords
    posr3 = s2tor3(*theta_phi_to_standard_coords(coords))
    posr3[0, 0] = 0.0
    posr3[0, 1] = 0.0
    posr3[0, 2] = 0.0

    rots = torch.repeat_interleave(hpy_verts_rots, masked_points_per_cell, dim=0)
    # BMM only works for b x n x m and b x m x 1
    # adding a dummy dimension to posr3
    vec_rot = torch.bmm(rots, posr3.unsqueeze(-1)).squeeze(-1)
    vec_scaled = r3tos2(vec_rot).to(torch.float32)

    # TODO: vec_scaled are small -> should they be normalized/rescaled?

    return vec_scaled


def get_target_coords_local(
    stream_id,
    hlc,
    masked_points_per_cell,
    coords,
    target_geoinfos,
    target_times,
    verts_rots,
    verts_local,
    nctrs,
):
    """Generate local coordinates for target coords w.r.t healpix cell vertices and
    and for healpix cell vertices themselves
    """

    # target_coords_lens = [len(t) for t in target_coords]
    # tcs, target_coords = tcs_optimized(target_coords)
    target_coords = s2tor3(*theta_phi_to_standard_coords(coords))
    if target_coords.shape[0] == 0:
        return torch.tensor([])

    # cell index of every target point; points are ordered by cell. This replaces splitting
    # into one tensor per cell, which is slow for 12 * 4**hl cells at high healpix levels.
    point_cell = torch.repeat_interleave(
        torch.arange(len(masked_points_per_cell)), masked_points_per_cell.to(torch.int64)
    )
    occupied_cells, point_occ = torch.unique(point_cell, return_inverse=True)

    def cell_coords_ctrs(rots):
        # = locs_to_cell_coords_ctrs(rots, per-cell split of target_coords)
        return torch.bmm(rots[point_cell], target_coords.unsqueeze(-1)).squeeze(-1)

    def ctr_coords(ctrs_r3):
        # = torch.cat(locs_to_ctr_coords(ctrs_r3, per-cell split of target_coords)), with the
        # (row-wise) rotations computed only for the occupied cells
        ctrs_rots = vecs_to_rots(ctrs_r3[occupied_cells]).to(torch.float32)
        return torch.bmm(ctrs_rots[point_occ], target_coords.unsqueeze(-1)).squeeze(-1)

    # target_geoinfos = torch.cat(target_geoinfos)
    # target_times = torch.cat(target_times)

    verts00_rots, verts10_rots, verts11_rots, verts01_rots, vertsmm_rots = verts_rots

    a = torch.zeros(
        [
            *target_coords.shape[:-1],
            1 + target_geoinfos.shape[1] + target_times.shape[1] + 5 * (3 * 5) + 3 * 8,
        ]
    )
    a[..., 0] = stream_id
    geoinfo_offset = 1
    a[..., geoinfo_offset : geoinfo_offset + target_times.shape[1]] = target_times
    geoinfo_offset += target_times.shape[1]
    a[..., geoinfo_offset : geoinfo_offset + target_geoinfos.shape[1]] = target_geoinfos
    geoinfo_offset += target_geoinfos.shape[1]

    ref = torch.tensor([1.0, 0.0, 0.0])

    tcs_lens = masked_points_per_cell.to(torch.int32)
    tcs_lens_mask = tcs_lens > 0
    tcs_lens = tcs_lens[tcs_lens_mask]

    vls = torch.cat(
        [
            vl.repeat([tt, 1, 1])
            for tt, vl in zip(tcs_lens, verts_local[tcs_lens_mask], strict=False)
        ],
        0,
    )
    vls = vls.transpose(0, 1)

    zi = 0
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - cell_coords_ctrs(verts00_rots)

    zi = 3
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[0]

    zi = 15
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - cell_coords_ctrs(verts10_rots)

    zi = 18
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[1]

    zi = 30
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - cell_coords_ctrs(verts11_rots)

    zi = 33
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[2]

    zi = 45
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - cell_coords_ctrs(verts01_rots)

    zi = 48
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[3]

    zi = 60
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + 3)] = ref - cell_coords_ctrs(vertsmm_rots)

    zi = 63
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + vls.shape[-1])] = vls[4]

    tcs_ctrs = torch.cat([ref - ctr_coords(c) for c in nctrs], -1)
    zi = 75
    a[..., (geoinfo_offset + zi) : (geoinfo_offset + zi + (3 * 8))] = tcs_ctrs

    # remaining geoinfos (zenith angle etc)
    zi = 99
    a[..., (geoinfo_offset + zi) :] = target_coords[..., (geoinfo_offset + 2) :]

    return a
