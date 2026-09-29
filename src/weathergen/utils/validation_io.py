# (C) Copyright 2025 WeatherGenerator contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

import logging

import astropy_healpix as hp
import numpy as np
import numpy.typing as npt
import torch

import weathergen.common.config as config
import weathergen.common.io as io
from weathergen.common.io import TimeRange, zarrio_writer
from weathergen.datasets.data_reader_base import TimeWindowHandler
from weathergen.model.engines import LatentState

_logger = logging.getLogger(__name__)


def write_output(
    cf,
    val_cfg,
    batch_size,
    mini_epoch,
    batch_idx,
    dn_data,
    batch,
    model_output,
    target_aux_out,
):
    """
    Interface for writing model output
    """

    # TODO: how to handle multiple physical loss terms
    outputs_physical = [
        loss_name
        for i, (loss_name, loss_term) in enumerate(val_cfg.losses.items())
        if loss_term.type == "LossPhysical"
    ]
    assert len(outputs_physical) == 1
    target_aux_out = target_aux_out[outputs_physical[0]]

    # collect all target / prediction-related information
    fp32 = torch.float32
    preds_all, targets_all, targets_coords_all, targets_times_all = [], [], [], []

    timestep_idxs = [0] if len(batch.get_output_idxs()) == 0 else batch.get_output_idxs()
    forecast_offset = timestep_idxs[0]
    targets_lens = []

    # TODO Maybe stopping at forecast_steps explained #1657
    for t_idx in timestep_idxs:
        preds_all += [[]]
        targets_all += [[]]
        targets_coords_all += [[]]
        targets_times_all += [[]]
        targets_lens += [[]]
        for sname in cf.streams.keys():
            # handle spoof data: do not write since it might corrupt validation (spoofing invisible
            # there)
            if target_aux_out.physical[t_idx][sname]["is_spoof"][0]:
                targets = target_aux_out.physical[t_idx][sname]["target"]
                # for-loop to make sure we have a consistent number of samples
                preds_s = [np.zeros((1, 0, t.shape[1])) for t in targets]
                targets_s = [np.zeros((0, t.shape[1])) for t in targets]
                t_coords_s = [np.zeros((0, 2)) for t in targets]
                t_times_s = [np.array([]).astype("datetime64[ns]") for t in targets]

            else:
                preds = model_output.get_physical_prediction(t_idx, sname)
                targets = target_aux_out.physical[t_idx][sname]["target"]

                preds_s, targets_s, t_coords_s, t_times_s = [], [], [], []

                # handle forcing streams or if sample is empty
                if preds is None:
                    # preds are empty so create copy of target and add ensemble dimension
                    assert targets[0].shape[0] == 0, "Empty preds but non-empty targets."
                    preds = [target.clone().unsqueeze(0) for target in targets]

                for i_batch, (pred, target) in enumerate(zip(preds, targets, strict=True)):
                    target_data = target_aux_out.physical[t_idx][sname]
                    t_coords = target_data["target_coords"][i_batch]
                    t_times = target_data["target_times"][i_batch]

                    idxs_inv = target_aux_out.physical[t_idx][sname]["idxs_inv"][i_batch]
                    if idxs_inv is not None:
                        pred = pred[:, idxs_inv]
                        target = target[idxs_inv]
                        t_coords = t_coords[idxs_inv]
                        t_times = t_times[idxs_inv]

                    # denormalize data if requested and map to storage format
                    preds_s += [dn_data(sname, pred.to(fp32)).detach().cpu().numpy()]
                    targets_s += [dn_data(sname, target.to(fp32)).detach().cpu().numpy()]

                    # extract original target coords and times from target data
                    t_coords_s += [t_coords.cpu().numpy()]
                    t_times_s += [t_times.astype("datetime64[ns]")]

            targets_lens[-1] += [[]]
            targets_lens[-1][-1] += [t.shape[0] for t in targets_s]

            preds_all[-1] += [np.concatenate(preds_s, axis=1)]
            targets_all[-1] += [np.concatenate(targets_s)]
            targets_coords_all[-1] += [np.concatenate(t_coords_s)]
            targets_times_all[-1] += [np.concatenate(t_times_s)]

    if len(preds_all) == 0 or np.array([p.shape[1] for pp in preds_all for p in pp]).sum() == 0:
        _logger.warning("Writing no data since predictions are empty.")
        return

    # collect source information
    sources = []
    for sample in batch.get_source_samples().get_samples():
        sources += [[]]
        for _, stream_data in sample.streams_data.items():
            # TODO: support multiple input steps
            sources[-1] += [stream_data.source_raw[0]]

    sample_idxs = [
        list(sample.streams_data.values())[0].sample_idx
        for sample in batch.get_source_samples().get_samples()
    ]

    # more prep work

    # output stream names to be written, use specified ones or all if nothing specified
    stream_names = list(cf.streams.keys())
    stream_infos = list(cf.streams.values())
    if val_cfg.get("output").get("streams") is not None:
        output_stream_names = val_cfg.output.streams
    else:
        output_stream_names = stream_names

    write_latents = io.LATENT_STREAM in output_stream_names
    output_streams: dict[str, int] = {
        name: stream_names.index(name) for name in output_stream_names if name != io.LATENT_STREAM
    }
    _logger.debug(f"Using output streams: {output_streams} from streams: {stream_names}")

    target_channels: list[list[str]] = [list(stream.val_target_channels) for stream in stream_infos]
    source_channels: list[list[str]] = [list(stream.val_source_channels) for stream in stream_infos]

    geoinfo_channels = [[] for _ in stream_infos]  # TODO obtain channels

    # calculate global sample indices for this batch by offsetting by sample_start
    sample_start = batch_idx * batch_size

    # write output

    start_date = val_cfg.start_date
    end_date = val_cfg.end_date

    twh = TimeWindowHandler(
        start_date,
        end_date,
        val_cfg.time_window_len,
        val_cfg.time_window_step,
    )
    source_windows = (twh.window(idx) for idx in sample_idxs)
    source_intervals = [TimeRange(window.start, window.end) for window in source_windows]

    latents_all = get_latent_output(batch, model_output) if write_latents else None

    data = io.OutputBatchData(
        sources,
        source_intervals,
        targets_all,
        preds_all,
        targets_coords_all,
        targets_times_all,
        targets_lens,
        output_streams,
        target_channels,
        source_channels,
        geoinfo_channels,
        latents=latents_all,
        sample_start=sample_start,
        forecast_offset=forecast_offset,
    )

    store_path = config.get_path_results(cf, mini_epoch, batch_idx)

    with zarrio_writer(store_path) as zio:
        for subset in data.items():
            zio.write_zarr(subset)
        # Write latent data directly to zarr store without using OutputItem validation
        if data.latents:
            _write_latent_data_to_zarr(
                zio,
                data,
                cf,
                batch,
                batch_idx,
                batch_size,
            )


def _write_latent_data_to_zarr(zio, data, cf, batch, batch_idx, batch_size):
    """Write dense forecast-grid latents, bypassing physical OutputItem validation."""
    sample_start = batch_idx * batch_size
    num_register_tokens = cf.num_register_tokens
    num_class_tokens = cf.num_class_tokens
    num_extra_tokens = num_register_tokens + num_class_tokens
    num_queries = cf.fe_num_queries
    coords_len = 12 * 4**cf.fe_healpix_level
    spatial_tokens = coords_len * num_queries
    lon, lat = _get_healpix_coords(cf)
    coords_array = np.stack([lat, lon], axis=1).astype(np.float32)
    geoinfo_array = np.zeros((coords_len, 0), dtype=np.float32)
    times_array = np.full((coords_len,), np.datetime64("NaT"), dtype="datetime64[ns]")

    for t_idx, latents_in_step in enumerate(data.latents):
        for sample_idx_in_batch, latents_in_sample in enumerate(latents_in_step):
            if not latents_in_sample:
                continue

            # Calculate global sample index
            global_sample_idx = sample_start + sample_idx_in_batch

            # Reserve latent step 0 for the initial encoded state.
            group_path = f"{global_sample_idx}/{io.LATENT_STREAM}/{t_idx + 1}"

            # ZipStore needs all attributes at creation to avoid duplicate zarr.json entries.
            group_attrs = {
                "num_extra_tokens": int(num_extra_tokens),
                "num_register_tokens": int(num_register_tokens),
                "num_class_tokens": int(num_class_tokens),
                "spatial_points": int(coords_len),
                "coords_order": "lat_lon",
                "num_queries": int(num_queries),
            }
            for name in ("tokens", "latent_state", "z_pre_norm", "patch_tokens"):
                if name in latents_in_sample:
                    group_attrs["total_points"] = int(np.asarray(latents_in_sample[name]).shape[0])
                    break

            group = zio.data_root.get(group_path)
            if group is None:
                group = zio.data_root.create_group(group_path, attributes=group_attrs)
            latent_names = {_latent_output_name(name) for name in latents_in_sample}
            for latent_name, latent_data in latents_in_sample.items():
                latent_array = np.asarray(latent_data)
                output_name = _latent_output_name(latent_name)
                if (
                    output_name == "tokens"
                    and latent_array.shape[0] == spatial_tokens + num_extra_tokens
                ):
                    registers, classes, latent_array = np.split(
                        latent_array, [num_register_tokens, num_extra_tokens]
                    )
                    # Prefer explicitly supplied normalized auxiliary fields over raw prefixes.
                    for name, array in (("register_tokens", registers), ("class_token", classes)):
                        if name not in latent_names:
                            _write_array(group, name, array)
                elif (
                    output_name not in ("tokens", "class_token", "register_tokens")
                    and num_class_tokens
                    and latent_array.shape[0] == spatial_tokens + num_class_tokens
                ):
                    _write_array(
                        group, f"{output_name}_class_token", latent_array[:num_class_tokens]
                    )
                    latent_array = latent_array[num_class_tokens:]

                if (
                    output_name not in ("class_token", "register_tokens")
                    and num_queries > 1
                    and latent_array.shape[0] == spatial_tokens
                ):
                    latent_array = latent_array.reshape(
                        coords_len, num_queries, *latent_array.shape[1:]
                    )
                _write_array(group, output_name, latent_array)

            _write_array(group, "coords", coords_array)
            _write_array(group, "geoinfo", geoinfo_array)
            _write_array(group, "times", times_array)


def _write_array(group, name: str, data: npt.NDArray) -> None:
    # ZipStore cannot truly delete; overwriting creates duplicate entries.
    if name not in group:
        group.create_array(name, data=data)


def _latent_output_name(name: str) -> str:
    return {
        "latent_state": "tokens",
        "latent_state_class_token": "class_token",
        "latent_state_register_tokens": "register_tokens",
    }.get(name, name)


_HEALPIX_COORDS_CACHE: dict[int, tuple[npt.NDArray, npt.NDArray]] = {}


def _get_healpix_coords(cf) -> tuple[npt.NDArray, npt.NDArray]:
    healpix_level = cf.fe_healpix_level
    cached = _HEALPIX_COORDS_CACHE.get(healpix_level)
    if cached is not None:
        return cached

    num_healpix_cells = 12 * 4**healpix_level
    ipix = np.arange(num_healpix_cells)
    lon, lat = hp.healpix_to_lonlat(ipix, 2**healpix_level, order="nested")
    coords = (lon.to_value("deg"), lat.to_value("deg"))
    _HEALPIX_COORDS_CACHE[healpix_level] = coords
    return coords


def get_latent_output(batch, model_output):
    """Collect tensor outputs per sample; encoder posterior dictionaries are not spatial outputs."""

    # collect latent outputs per forecast step and per sample
    fp32 = torch.float32

    timestep_idxs = [0] if len(batch.get_output_idxs()) == 0 else batch.get_output_idxs()

    n_samples = len(batch.get_source_samples())

    latents_all: list[list[dict]] = []
    for t_idx in timestep_idxs:
        latents_all.append([])
        latent_pred = model_output.get_latent_prediction(t_idx)
        for i_sample in range(n_samples):
            per_sample: dict = {}
            for lname, lval in latent_pred.items():
                if isinstance(lval, LatentState):
                    fields = {
                        "tokens": lval.z_pre_norm,
                        "register_tokens": lval.register_tokens,
                        "class_token": lval.class_token,
                    }
                    for field_name, tensor in fields.items():
                        if tensor is not None:
                            sample_tensor = tensor[i_sample]
                            per_sample[field_name] = sample_tensor.detach().to(fp32).cpu().numpy()
                elif isinstance(lval, torch.Tensor):
                    per_sample[lname] = lval[i_sample].detach().to(fp32).cpu().numpy()
            latents_all[-1].append(per_sample)

    return latents_all
