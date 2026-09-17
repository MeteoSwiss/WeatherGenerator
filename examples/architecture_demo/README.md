# WeatherGenerator architecture on a 64 x 64 x 3 toy field

`wg_mini_forward.py` is a plain-PyTorch re-implementation of the WeatherGenerator data flow,
small enough to read in one sitting and to run on a laptop CPU in seconds. It does **not**
import the real model; it mirrors its structure so that each stage can be matched 1:1 to the
production code.

```bash
uv run python examples/architecture_demo/wg_mini_forward.py            # forward pass, print shapes
uv run python examples/architecture_demo/wg_mini_forward.py --steps 40 # + a few optimizer steps
```

## Data flow

```
 64x64x3 field on a lat/lon grid
        |  tokenize: point -> [time | xyz | data], group by cell, token_size points per token
        v
 source tokens        [cells, tokens/cell, token_size, feat]     (64, 8, 8, 8)
        |  1. StreamEmbedTransformer  (one per stream)
        v
 local tokens         [cells, tokens/cell, D_local]              (64, 8, 32)
        |  2. LocalAssimilationEngine: self-attention within a cell
        v
 local tokens         [cells, tokens/cell, D_local]              (64, 8, 32)
        |  3. Local2GlobalAssimilationEngine: learnable q_cells cross-attend to cell tokens
        v
 cell latents         [cells, Q, D_global]                       (64, 2, 32)
        |  4. GlobalAssimilationEngine: dense self-attention over all cells
        v
 global latent        [B, cells*Q, D_global]                     (1, 128, 32)
        |  5. ForecastingEngine, once per forecast step (step 0 = analysis, no-op)
        v
 latent at step k     [B, cells*Q, D_global]
        |  6. TargetPredictionEngine: target coord -> query -> cross-attention to the
        |     latents of the containing cell + its 8 neighbours -> pred_head
        v
 prediction           [targets, channels]                        (4096, 3)
```

Steps 1-4 are the **encoder** (`EncoderModule`), run once on the input. Steps 5-6 are
repeated for every output step, entirely in latent space; only the decoder touches
physical space again.

## Mapping to the real code

| Demo | Real implementation | Config keys |
|------|---------------------|-------------|
| `Config` | `config/default_config.yml` + stream YAML | |
| `tokenize_source`, `cell_index` | `datasets/tokenizer_utils.py` `tokenize_apply_mask_source`; cells are HEALPix pixels via `healpy.ang2pix` | `healpix_level`, stream `token_size` |
| `cell_neighbours` | `ModelParams.hp_nbours` (HEALPix 1-ring, 9 entries) | |
| `StreamEmbedTransformer` | `engines.EmbeddingEngine` -> `embeddings.StreamEmbedTransformer` / `StreamEmbedLinear` | stream `embed: {net, dim_embed, num_blocks, num_heads}` |
| `LocalAssimilationEngine` | `engines.LocalAssimilationEngine` (varlen attention over `cell_lens`) | `ae_local_dim_embed`, `ae_local_num_blocks`, `ae_local_num_heads` |
| `Local2GlobalAssimilationEngine.q_cells` | `EncoderModule.q_cells` | `ae_local_num_queries`, `ae_local_queries_per_cell` |
| `Local2GlobalAssimilationEngine.ae_adapter` | `engines.Local2GlobalAssimilationEngine` (`MultiCrossAttentionHeadVarlenSlicedQ`) or `Local2GlobalSumEngine` | `ae_adapter_type`, `ae_adapter_num_heads`, `ae_adapter_num_blocks` |
| `GlobalAssimilationEngine` | `engines.GlobalAssimilationEngine` (+ `QueryAggregationEngine`, register/class tokens) | `ae_global_dim_embed`, `ae_global_num_blocks`, `ae_global_num_heads`, `ae_global_att_dense_rate` |
| `ForecastingEngine` | `engines.ForecastingEngine` (weights init N(0, 1e-3)) | `fe_num_blocks`, `fe_num_heads`, `forecast_steps`, `forecast_att_dense_rate` |
| `TargetPredictionEngine.embed_target_coords` | `Model.embed_target_coords[stream]` | stream `embed_target_coords: {net, dim_embed}` |
| `TargetPredictionEngine.tte` | `engines.TargetPredictionEngine` (`decoder_type`: PerceiverIO, AdaLayerNorm, ...) | stream `target_readout: {num_layers, num_heads}`, `decoder_type` |
| `TargetPredictionEngine.pred_head` | `engines.EnsPredictionHead` | stream `pred_head: {num_layers, ens_size}` |
| `MiniWeatherGenerator.forward` | `model.Model.forward` -> `predict_decoders` | |

## What the demo leaves out

- **HEALPix.** The real model tiles the sphere with `12 * 4**healpix_level` equal-area cells.
  The demo uses an 8 x 8 grid of square cells; neighbours wrap in longitude and clamp at the poles.
- **Variable-length cells.** Observations are irregular, so real cells hold different numbers of
  tokens; attention is variable-length (`cell_lens`, flash-attn varlen). Here every cell has 8 tokens
  and attention is simply batched over the cell dimension.
- **Multiple streams.** Each stream (ERA5, CERRA, SYNOP, satellites, ...) has its own embedding
  network and its own decoder; tokens of all streams are scattered into the same cells before the
  local assimilation engine. Streams with `forcing: True` skip the decoder, `diagnostic: True` skip the encoder.
- **Masking.** Training masks a fraction of source tokens (`masking_rate`, `masking_strategy`) and
  predicts the masked points; the decoder queries are the *masked* coordinates. The demo predicts the full grid.
- **Register / class tokens, SSL heads.** Extra global tokens (`num_register_tokens`,
  `num_class_tokens`), `QueryAggregationEngine`, latent prediction heads (JEPA/DINO-style losses) and
  latent noise (`latent_noise_kl_weight`) are omitted.
- **Token content.** Real tokens carry `stream_id`, time encodings, local coordinates relative to the
  cell centre, geoinfo channels and the data; positional encodings are harmonic / RoPE
  (`positional_encoding.py`). The demo uses `[sin t, cos t | x, y, z | data]` and a learned PE.
- **Multi-step input, batches, mixed precision, activation checkpointing, ensembles (`ens_size`).**
