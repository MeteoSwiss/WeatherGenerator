from pathlib import Path

import numpy as np
import pytest
import torch
from astropy_healpix import healpy
from omegaconf import OmegaConf

pytest.importorskip("flash_attn", reason="flash_attn required to import native model blocks")

from weathergen.common.io import IOReaderData  # noqa: E402
from weathergen.datasets.batch import BatchSamples  # noqa: E402
from weathergen.datasets.stream_data import StreamData  # noqa: E402
from weathergen.datasets.tokenizer_masking import TokenizerMasking  # noqa: E402
from weathergen.model.engines import BilinearDecoder  # noqa: E402
from weathergen.model.model import Model, ModelOutput  # noqa: E402


def _geometry(coords, values=None, frequencies=(), fourier_enabled=True, **options):
    coords = np.asarray(coords, dtype=np.float32)
    data = IOReaderData(
        coords=coords,
        geoinfos=np.empty((len(coords), 0), dtype=np.float32),
        data=np.zeros((len(coords), 1), dtype=np.float32) if values is None else values[:, None],
        datetimes=np.full(len(coords), np.datetime64("1985-01-01T00:00", "ns")),
    )
    tokenizer = TokenizerMasking(2, None, **options)
    stream = {
        "stream_id": 1,
        "token_size": 32,
        "embed_target_coords": {
            "fourier_frequencies": frequencies,
            "fourier_enabled": fourier_enabled,
        },
    }
    tokens = tokenizer.get_tokens_windows(stream, [data], False)[0]
    window = np.array(["1985-01-01T00:00", "1985-01-01T03:00"], dtype="datetime64[ns]")
    mask = np.ones(192, dtype=bool)
    geometry = tokenizer.get_target_coords(stream, data, tokens, window, mask)
    ordered_values = tokenizer.get_target_values(stream, data, tokens, window, mask)[0]
    return geometry, ordered_values


def _model(coord_dim=9):
    cf = OmegaConf.create(
        dict(
            decoder_type="Linear",
            ae_local_num_queries=1,
            healpix_level=2,
            attention_dtype="fp32",
            streams={"field": {}},
            num_register_tokens=1,
            num_class_tokens=0,
        )
    )
    with torch.random.fork_rng(devices=[]):
        model = Model(cf, [], [], [])
        model.embed_target_coords = torch.nn.ModuleDict(
            {"field": torch.nn.Linear(coord_dim, 1, bias=False)}
        )
        model.target_token_engines = torch.nn.ModuleDict(
            {"field": BilinearDecoder("field", 1, 1, 1)}
        )
        model.pred_heads = torch.nn.ModuleDict({"field": torch.nn.Identity()})
    with torch.no_grad():
        model.embed_target_coords["field"].weight.zero_()
        model.embed_target_coords["field"].weight[0, 0] = 1
        model.target_token_engines["field"].bilin.weight.fill_(1)
    return model


def _predict(model, geometries, tokens):
    batch = BatchSamples(["field"], len(geometries), 1, [0])
    for i, geometry in enumerate(geometries):
        stream = StreamData(i, 1, 1, 192)
        if geometry is not None:
            features, counts = geometry
            stream.add_target_coords("train", 0, features, counts, False)
        batch.samples[i].add_stream_data("field", stream)
    batch.to_device("cpu")
    output = model.predict_decoders(None, 0, tokens, batch, ModelOutput(1))
    return output.get_physical_prediction(0, "field")


def test_fourier_readout_uses_only_owning_cells_and_supports_ablation():
    model = _model(15)
    frequency = np.pi / 2
    with torch.no_grad():
        model.embed_target_coords["field"].weight[0, 8] = 1  # Global Y sine feature.
    field = np.sin(np.arange(192)).astype(np.float32)
    coords = np.array([[0, 179.999], [0, -179.999], [45, -10], [50, 5]])
    theta = np.deg2rad(90 - coords[:, 0])
    phi = np.deg2rad(coords[:, 1] + 180)
    owners = healpy.ang2pix(4, theta, phi, nest=True)
    expected_field = field[owners] * (1 + np.sin(frequency * np.sin(theta) * np.sin(phi)))
    geometry, expected = _geometry(
        coords, expected_field.astype(np.float32), frequencies=[frequency], global_coords=True
    )
    tokens = torch.cat((torch.tensor([999.0]), torch.from_numpy(field))).reshape(1, 193, 1)
    prediction = _predict(model, [geometry], tokens)[0]
    torch.testing.assert_close(prediction[0], expected, atol=2e-6, rtol=0)
    ablated_geometry, ablated_expected = _geometry(
        coords, field[owners], frequencies=[frequency], fourier_enabled=False, global_coords=True
    )
    ablated_prediction = _predict(model, [ablated_geometry], tokens)[0]
    torch.testing.assert_close(ablated_prediction[0], ablated_expected, atol=2e-6, rtol=0)


def test_readout_keeps_samples_and_gradients_separate_with_empty_sample_and_auxiliary_token():
    model = _model()
    geometry, _ = _geometry([[45, -10], [50, 5]], global_coords=True)
    tokens = torch.tensor([3.0, 17.0, 7.0]).reshape(3, 1, 1).repeat(1, 193, 1)
    tokens[:, 0] = 999
    tokens.requires_grad_()
    prediction = _predict(model, [geometry, None, geometry], tokens)
    torch.testing.assert_close(prediction[0], torch.full((1, 2, 1), 3.0))
    assert prediction[1].shape == (1, 0, 1)
    torch.testing.assert_close(prediction[2], torch.full((1, 2, 1), 7.0))
    (prediction[2] - 5).square().sum().backward()
    assert torch.count_nonzero(tokens.grad[:2]) == 0
    assert torch.count_nonzero(tokens.grad[:, 0]) == 0
    assert tokens.grad[2, 1:].abs().sum() > 0


def test_full_precision_readout_preserves_small_spatial_variations_under_autocast():
    model = _model()
    model.cf.decoder_full_precision = True
    with torch.no_grad():
        model.embed_target_coords["field"].weight[0, -3] = -1
    expected_field = (1 - np.cos(np.deg2rad([1.0, 2.0]))).astype(np.float32)
    geometry, expected = _geometry([[0, -179], [0, -178]], expected_field, global_coords=True)
    tokens = torch.ones((1, 193, 1), requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        prediction = _predict(model, [geometry], tokens)[0]
    torch.testing.assert_close(prediction[0], expected, atol=1e-7, rtol=0)
    prediction.sum().backward()
    torch.testing.assert_close(tokens.grad.sum(), expected.sum(), atol=1e-7, rtol=0)


def test_hard_readout_supports_branch_absolute_and_global_fourier_coordinates():
    coords = np.array([[45.0, -10.0], [50.0, 5.0], [60.0, 20.0]])
    theta = np.deg2rad(90 - coords[:, 0])
    phi = np.deg2rad(coords[:, 1] + 180)
    owners = healpy.ang2pix(4, theta, phi, nest=True)
    field = 1 + np.arange(192, dtype=np.float32) / 192
    tokens = torch.cat((torch.tensor([999.0]), torch.from_numpy(field))).reshape(1, 193, 1)

    model = _model(105)
    with torch.no_grad():
        model.embed_target_coords["field"].weight.zero_()
        model.embed_target_coords["field"].weight[0, [98, 97, 96, 95]] = torch.arange(1.0, 5.0)
    absolute = (
        np.sin(coords[:, 0])
        + 2 * np.cos(coords[:, 0])
        + 3 * np.sin(coords[:, 1])
        + 4 * np.cos(coords[:, 1])
    )
    geometry, expected = _geometry(
        coords,
        (absolute * field[owners]).astype(np.float32),
        decoder_absolute_coords=True,
    )
    torch.testing.assert_close(_predict(model, [geometry], tokens)[0][0], expected)

    frequency = np.pi / 2
    model = _model(15)
    with torch.no_grad():
        model.embed_target_coords["field"].weight.zero_()
        model.embed_target_coords["field"].weight[0, 6] = 1
    expected_field = np.sin(frequency * np.sin(theta) * np.cos(phi)) * field[owners]
    geometry, expected = _geometry(
        coords,
        expected_field.astype(np.float32),
        frequencies=[frequency],
        global_coords=True,
    )
    torch.testing.assert_close(_predict(model, [geometry], tokens)[0][0], expected)


def test_materialized_queries_and_bilinear_readout_can_predict_and_learn():
    cf = OmegaConf.load(Path(__file__).resolve().parents[1] / "config/default_config.yml")
    cf.streams = {}
    cf.healpix_level = 2
    cf.decoder_type = "Linear"
    cf.ae_local_dim_embed = cf.ae_global_dim_embed = 16
    cf.ae_local_num_blocks = cf.ae_global_num_blocks = cf.fe_num_blocks = 0
    cf.ae_adapter_num_heads = 2
    cf.ae_adapter_embed = 8
    cf.ae_local_queries_per_cell = True
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(1234)
        with torch.device("meta"):
            model = Model(cf, [], [], []).create()
            model.embed_target_coords["field"] = torch.nn.Linear(9, 1, bias=False)
            model.target_token_engines["field"] = BilinearDecoder("field", 1, 16, 1)
            model.pred_heads["field"] = torch.nn.Identity()
        model.streams = {"field": {}}
        model.to_empty(device="cpu")
        with torch.no_grad():
            model.encoder.q_cells.fill_(float("nan"))
            model.target_token_engines["field"].bilin.weight.fill_(float("nan"))
        model.reset_parameters()
        model.eval()
        geometry, _ = _geometry([[45, -10], [50, 5]], global_coords=True)
        prediction = _predict(model, [geometry], model.encoder.q_cells.transpose(0, 1))[0]
        assert torch.isfinite(prediction).all()
        (prediction - 1).square().mean().backward()
        for parameter in (
            model.encoder.q_cells,
            model.target_token_engines["field"].bilin.weight,
        ):
            assert torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0
