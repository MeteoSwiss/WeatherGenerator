# (C) Copyright 2026 WeatherGenerator contributors.

import copy
from types import SimpleNamespace

import astropy_healpix.healpy as hp
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from weathergen.common.config import (
    _sanitize_time_keys,
    get_encoder_config,
    get_encoder_streams,
)
from weathergen.common.io import IOReaderData
from weathergen.datasets.multi_stream_data_sampler import MultiStreamDataSampler, _Stream


def _config(forecast_level=0):
    streams = {}
    for i, (name, routes) in enumerate(
        [("A", ["coarse", "fine"]), ("B", ["coarse"]), ("C", ["fine"])]
    ):
        streams[name] = {
            "name": name,
            "stream_id": i,
            "encoders": routes,
            "token_size": 1,
            "embed": {"num_tokens": 1},
            "train_source_channels": ["x"],
            "train_target_channels": ["x"],
            "val_source_channels": ["x"],
            "val_target_channels": ["x"],
        }
    views = {
        str(i): {
            "num_steps_input": 2,
            "masking_strategy": "random",
            "masking_strategy_config": {"rate": rate},
        }
        for i, rate in enumerate((0.0, 0.5, 1.0))
    }
    targets = {str(i): {"num_steps_input": 2, "masking_strategy": "forecast"} for i in range(3)}
    cf = OmegaConf.create(
        {
            "encoders": {
                "coarse": {
                    "healpix_level": 0,
                    "ae_local_dim_embed": 16,
                    "ae_global_dim_embed": 32,
                    "ae_local_num_queries": 1,
                },
                "fine": {
                    "healpix_level": 2,
                    "ae_local_dim_embed": 8,
                    "ae_global_dim_embed": 32,
                    "ae_local_num_queries": 1,
                },
            },
            "fe_healpix_level": forecast_level,
            "fe_dim_embed": 32,
            "fe_num_queries": 1,
            "streams": streams,
            "rank": 0,
            "world_size": 1,
            "data_loading": {"rng_seed": 71, "num_workers": 0},
            "training_config": {
                "training_mode": ["masking", "student_teacher"],
                "start_date": "2020-01-01T00:00",
                "end_date": "2020-02-01T00:00",
                "time_window_len": "06:00:00",
                "time_window_step": "06:00:00",
                "samples_per_mini_epoch": 12,
                "shuffle": False,
                "model_input": views,
                "target_input": targets,
                "forecast": {"offset": 0, "time_step": "06:00:00", "policy": None, "num_steps": 0},
                "losses": {
                    "physical": {
                        "type": "LossPhysical",
                        "loss_fcts": {
                            "mse": {
                                "target_source_correspondence": {
                                    i: {i: "independent"} for i in range(3)
                                },
                            }
                        },
                    }
                },
            },
        }
    )
    return _sanitize_time_keys(cf)


class _ReaderBoundarySampler(MultiStreamDataSampler):
    """Only raw reader windows are synthetic; routing/masking/tokenization are real."""

    spoofed = ()
    missing = ()

    def _init_stream_datasets(self, cf):
        return {
            name: _Stream(info, [SimpleNamespace(stream_info=info)])
            for name, info in cf.streams.items()
        }

    def _get_data_windows(self, base_idx, num_forecast_steps, num_steps_input_max, stream_ds):
        info = stream_ds[0].stream_info

        def window(idx):
            theta, phi = hp.pix2ang(4, np.arange(192), nest=True)
            coords = np.stack((90 - np.rad2deg(theta), np.rad2deg(phi) - 180), -1).astype("float32")
            data = (np.arange(192, dtype="float32") + idx * 1000 + info.stream_id * 10000)[:, None]
            if info.name in self.missing:
                coords, data = coords[:0], data[:0]
            return IOReaderData(
                coords,
                np.empty((len(data), 0), dtype="float32"),
                data,
                np.full(
                    len(data), self.time_window_handler.window(idx).start, dtype="datetime64[ms]"
                ),
                is_spoof=info.name in self.spoofed,
            )

        return (
            [window(i) for i in range(base_idx - num_steps_input_max + 1, base_idx + 1)],
            [window(base_idx + i) for i in range(self._get_output_length(num_forecast_steps))],
        )


def test_routing_architecture_and_validation():
    cf = _config()
    assert get_encoder_streams(cf) == {"coarse": ["A", "B"], "fine": ["A", "C"]}
    scoped = get_encoder_config(cf, "fine")
    assert list(scoped.streams) == ["A", "C"]
    assert scoped.ae_local_dim_embed == 8
    assert cf.encoders.coarse.ae_local_dim_embed == 16
    for path, value in (
        ("encoders.fine.healpix_level", -1),
        ("fe_healpix_level", True),
        ("streams.A.encoders", ["coarse", "coarse"]),
        ("streams.C.encoders", ["unknown"]),
        ("streams.A.encoders", "coarse"),
        ("encoders.fine.ae_global_dim_embed", 64),
        ("encoders.fine.ae_local_num_queries", 2),
    ):
        bad = copy.deepcopy(cf)
        OmegaConf.update(bad, path, value, merge=False)
        with pytest.raises(ValueError):
            get_encoder_streams(bad)
    cf.streams.A.encoders = ["coarse"]
    cf.streams.C.encoders = []
    with pytest.raises(ValueError, match="member stream"):
        get_encoder_streams(cf)


def test_overlapping_grids_steps_views_and_subsetting():
    cf = _config()
    sampler = _ReaderBoundarySampler(cf, cf.training_config, "val")
    sampler.reset()
    batch = sampler._get_batch(4, 0)
    assert batch.source_samples.tokens_lens is None
    for samples in (batch.source_samples, batch.target_samples):
        assert list(samples.samples[0].streams_data) == ["A", "B", "C"]
        for name, members, cells in (("coarse", ["A", "B"], 12), ("fine", ["A", "C"], 192)):
            child = samples.encoder_batches[name]
            assert child.tokens_lens.shape == (2, 3, 2, cells)
            assert child.coverage.shape == (2, 3, cells)
            assert child.coverage.all()
            assert list(child.samples[0].streams_data) == members
            for step, index in enumerate((4, 3)):
                for member in members:
                    stream = child.samples[2].streams_data[member]
                    actual = stream.source_tokens_cells[step][..., -1].flatten().sort().values
                    expected = (
                        torch.arange(192) + index * 1000 + cf.streams[member].stream_id * 10000
                    )
                    torch.testing.assert_close(actual, expected.float())
                    assert stream.source_raw[step] is None
        subset = samples.get_subset([2, 0])
        assert [s.view_meta.global_params["idx"] for s in subset.samples] == [2, 0]
        assert [s.view_meta.global_params["correspondence"] for s in subset.samples] == [
            samples.samples[i].view_meta.global_params["correspondence"] for i in (2, 0)
        ]
        for name, child in samples.encoder_batches.items():
            selected = subset.encoder_batches[name]
            torch.testing.assert_close(selected.tokens_lens, child.tokens_lens[:, [2, 0]])
            torch.testing.assert_close(selected.coverage, child.coverage[:, [2, 0]])
            for j, i in enumerate((2, 0)):
                for member in child.samples[i].streams_data:
                    original = child.samples[i].streams_data[member]
                    chosen = selected.samples[j].streams_data[member]
                    assert chosen.sample_idx == original.sample_idx
                    for step in range(2):
                        torch.testing.assert_close(
                            chosen.source_tokens_cells[step], original.source_tokens_cells[step]
                        )
    assert not batch.source_samples.sources_empty()
    assert not batch.source_samples.sources_nan()
    assert not batch.source_samples.samples[0].view_meta.mask.any()
    assert batch.source_samples.samples[2].view_meta.mask.all()
    for name in cf.streams:
        assert len(batch.target_samples.samples[2].streams_data[name].target_tokens[0]) == 192
        assert len(batch.source_samples.samples[2].streams_data[name].source_raw[0].data) == 192


def test_coarse_all_mask_and_fused_any_visibility_exclude_spoofs():
    cf = _config(2)
    sampler = _ReaderBoundarySampler(cf, cf.training_config, "val")
    sampler.spoofed = ("B",)
    sampler.missing = ("C",)
    sampler.reset()
    batch = sampler._get_batch(4, 0)
    sources = batch.source_samples
    for view, sample in enumerate(sources.samples):
        keep = sample.meta_info["A"].mask
        coarse = sources.encoder_batches["coarse"]
        fine = sources.encoder_batches["fine"]
        assert coarse.coverage[:, view].all() and fine.coverage[:, view].all()
        torch.testing.assert_close(
            coarse.tokens_lens[0, view, 0].bool(), keep.reshape(12, 16).all(-1)
        )
        torch.testing.assert_close(fine.tokens_lens[0, view, 0].bool(), keep)
        assert not coarse.tokens_lens[:, view, 1].any()
        assert not fine.tokens_lens[:, view, 1].any()
        torch.testing.assert_close(sample.view_meta.mask, keep)
    # Entirely absent fine branch stays zero even though the coarse branch is populated.
    sampler.spoofed = ("A", "C")
    sampler.missing = ()
    batch = sampler._get_batch(4, 0)
    assert not batch.source_samples.encoder_batches["fine"].coverage.any()
    assert not batch.source_samples.sources_empty()


@pytest.mark.parametrize("source_level,target_level", [(0, 0), (2, 0), (0, 2)])
def test_nested_addition_and_autograd(source_level, target_level):
    from weathergen.model.utils import add_healpix_latents_

    ns, nt = 12 * 4**source_level, 12 * 4**target_level
    source = torch.arange(2 * ns * 2 * 3, dtype=torch.float64).reshape(2, ns, 2, 3).requires_grad_()
    destination = torch.full((2, nt, 2, 3), 10.0, dtype=torch.float64)
    before = source.detach().clone()
    expected = destination.clone()
    if ns >= nt:
        for cell in range(nt):
            expected[:, cell] += before[:, cell * (ns // nt) : (cell + 1) * (ns // nt)].sum(1)
        factor = 1
    else:
        for cell in range(nt):
            expected[:, cell] += before[:, cell // (nt // ns)]
        factor = nt // ns
    add_healpix_latents_(destination, source, source_level, target_level)
    torch.testing.assert_close(destination, expected)
    torch.testing.assert_close(source.detach(), before)
    destination.sum().backward()
    torch.testing.assert_close(source.grad, torch.full_like(source, factor))


def test_partial_coverage_sum_not_mean():
    from weathergen.model.utils import add_healpix_latents_

    fine = torch.zeros(1, 192, 1, 1)
    fine[0, :16, 0, 0] = torch.arange(1, 17)
    for last_child, expected in ((16, 146), (0, 130)):
        fine[0, 15] = last_child
        coarse = torch.full((1, 12, 1, 1), 10.0)
        add_healpix_latents_(coarse, fine, 2, 0)
        assert coarse[0, 0, 0, 0] == expected
        torch.testing.assert_close(coarse[0, 1:], torch.full((11, 1, 1), 10.0))
    with pytest.raises(ValueError):
        add_healpix_latents_(coarse, fine[:, :-1], 2, 0)


def test_ssl_aligns_native_views_and_expands_cell_masks_over_queries():
    from weathergen.datasets.batch import SampleMetaData
    from weathergen.train.loss_modules.loss_module_ssl import LossLatentSSLStudentTeacher

    cf = _config()
    cf.fe_num_queries = 2
    cf.num_class_tokens = 1
    visible = torch.arange(12) < 6
    source_info = [
        SampleMetaData({}, visible, {"idx": i, "correspondence": i, "loss": ["JEPA"]})
        for i in (2, 0)
    ]
    target_info = [
        SampleMetaData({}, torch.ones(12, dtype=torch.bool), {"idx": i, "loss": ["JEPA"]})
        for i in (0, 2)
    ]
    student = torch.tensor([4.0, 0.0]).reshape(2, 1, 1).expand(2, 24, 1).clone().requires_grad_()
    teacher = torch.tensor([2.0, 5.0]).reshape(2, 1, 1).expand(2, 24, 1)
    preds = SimpleNamespace(latent=[{"JEPA": student}])
    targets = SimpleNamespace(latent={"JEPA": teacher}, aux_outputs=target_info)
    loss_module = LossLatentSSLStudentTeacher(
        cf, cf.training_config, "train", "cpu", JEPA={"weight": 1.0, "loss_extra_args": {}}
    )
    metadata = (None, source_info, None, target_info)
    loss = loss_module.compute_loss(preds, targets, metadata).loss
    torch.testing.assert_close(loss, torch.tensor(3.0))
    loss.backward()
    assert not student.grad[:, :12].any()
    assert (student.grad[:, 12:] < 0).all()
    source_info[0].global_params["correspondence"] = 9
    with pytest.raises(ValueError):
        loss_module.compute_loss(preds, targets, metadata)
    source_info[0].global_params["correspondence"] = 2
    source_info[0].mask = visible[:-1]
    with pytest.raises(ValueError):
        loss_module.compute_loss(preds, targets, metadata)
