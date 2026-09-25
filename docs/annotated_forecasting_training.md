# The Annotated WeatherGenerator: one forecasting training step

A walk through `run_train.py train --config config/config_forecasting.yml`, in the spirit of
*The Annotated Transformer*: source excerpts interleaved with the tensor shapes actually observed
while stepping the run under `debugpy` on a single GH200 (2026-09-22, commit of this checkout).

Everything marked **observed** was read from live frames. Everything else is read from source with
`file:line` references. Shapes are for `batch_size_per_gpu = 1` (the config default); other ranks
see identical per-rank shapes.

How the numbers were captured (reproducible with the `.vscode/launch.json` "base train" entry):

```
program : src/weathergen/run_train.py
args    : train --config config/config_forecasting.yml --options data_loading.num_workers=0
```

`num_workers=0` is the only deviation from `launch.json`: it runs `MultiStreamDataSampler` in the
main process so the debugger can see the sample being built (with `num_workers=12` it is built in
forked workers). Breakpoints used: `trainer.py:276,287,456,480,502,519,569`,
`encoder.py:129,133,140,191,210,323`, `model.py:689,705,824,836`.

---

## 0. The whole step on one page

```
 ERA5 anemoi zarr  (64281 × 113 × 1 × 40320)  6-hourly, o96 grid                   config_forecasting.yml
   │  DataReaderAnemoi: 80 source ch, 81 target ch, 9 geoinfo ch                    + streams/era5.yml
   ▼                                                                                 + private (paths)
 MultiStreamDataSampler._get_batch(idx=36234, fsteps=3)
   │  source window  [2003-10-20 12:00, 18:00)          → 40320 pts × 97 cols
   │  target windows  t+6h, t+12h, t+18h                → 3 × 20000 pts (random subset of 40320)
   │  HEALPix level 5 → 12288 cells, tokens of 8 points
   ▼
 ModelBatch
   source: source_tokens_cells[0] (12296, 8, 97)  source_tokens_lens[0] (12288,)  target_coords[k] (20000,114)
   target: target_tokens[k] (20000, 81)  target_coords_raw[k] (20000, 2)  target_times_raw[k] (20000,)
   ▼
 Model.forward                                                     shape (observed)          params
   EmbeddingEngine / StreamEmbedTransformer                        (12296, 2048) bf16          5.3 M
   assimilate_local: q_cells + pe_global → SlicedQ cross-attn      (12288, 1, 2048) → (1,12288,2048)   50.3 M
   GlobalAssimilationEngine 4 × [MSA(32h) , MLP]                   (1, 12288, 2048) fp32     134.3 M
   sum over input steps (1 step → identity)                        (1, 12288, 2048)
   for step in 1,2,3:
     ForecastingEngine 16 × [MSA(16h), MLP] (+LN after block 7)    (1, 12288, 2048)          537.0 M
     predict_decoders: 1-ring gather (110592, 2048)
       embed_target_coords Linear(114→512)                         (20000, 512)                58 K
       TargetPredictionEngineClassic 2 × [CA, SA, MLP] (AdaLN)     (20000, 512)              12.6 M
       EnsPredictionHead Linear(512→81)                            (1, 20000, 81)              42 K
   ▼                                                                                   Σ 739 652 438
 ModelOutput.physical[step]['ERA5'] = [ (1, 20000, 81) ]   for step = 1,2,3
   ▼
 PhysicalTargetAndAux.compute → TargetAuxOutput.physical[step]['ERA5']{target (20000,81), coords, times}
   ▼
 LossPhysical: mse(target, pred.mean(ens)) · cos(lat) · channel_weights → mean over steps  → loss 0.4140 (observed)
   ▼
 GradScaler.scale(loss).backward → unscale → clip_grad_norm_(1.0)  (total_norm 1.578 observed)
 AdamW.step → LR.step (1e-6 → 1.0019e-6) → EMA.update
 GPU memory: 21.6 GB peak after forward, 35.7 GB peak after backward (observed, no FSDP, batch 1)
```

---

## 1. Configuration resolution

`run_train.run_train` (`src/weathergen/run_train.py:158-196`):

```python
cf = config.load_merge_configs(
    args.private_config, None, None, args.base_config, *args.config, cli_overwrite
)
cf = config.set_run_id(cf, args.run_id, False)
cf.data_loading.rng_seed = int(time.time())
devices = Trainer.init_torch(multiprocessing_method=mp_method)
cf = Trainer.init_ddp(cf)
cf.streams = config.load_streams(Path(cf.streams_directory))
```

`load_merge_configs` (`packages/common/src/weathergen/common/config.py:414-473`) merges, ascending
precedence: `config/default_config.yml` → private config (paths; resolved via
`WEATHERGEN_PRIVATE_CONF` or `platform-env.py`, `config.py:568-622`) → `config_forecasting.yml` →
`--options`. Time strings become OmegaConf resolvers so `cf.training_config.forecast.time_step`
yields `np.timedelta64(6,'h')` (`_sanitize_time_keys`, `config.py:113-135`).

`Trainer.init_ddp` (`train/trainer_base.py:66-144`) reads `WORLD_SIZE`/`SLURM_NTASKS`. **Observed:**
`world_size=1, with_ddp=False, with_fsdp=True` — with one rank no process group is created and
`init_model_and_shard` skips `fully_shard`, so this run holds plain fp32 `Parameter`s.
On a multi-GPU allocation the same config shards every attention/MLP leaf with FSDP2
(`model/model_interface.py:84-136`).

Values that shape everything below (**observed** after merge):

| key | value | consequence |
|---|---|---|
| `healpix_level` | 5 | `num_healpix_cells = 12·4⁵ = 12288` |
| `training_config.forecast` | `offset=1, num_steps=3, time_step=6h, policy=fixed` | `output_steps=4`, `output_idxs=[1,2,3]` |
| `training_config.model_input` | `{forecasting: {masking_strategy: forecast}}` | 1 source sample, keep-all cell mask |
| `training_config.losses` | `{physical: {type: LossPhysical, loss_fcts: {mse: {}}}}` | one loss term, weight 1 |
| `time_window_len/step` | 6h / 6h | one analysis per window |
| `samples_per_mini_epoch` | 4096 (val 256) | `len(dataset)=4096`, `len(dataset_val)=256` |
| `ae_local_dim_embed / ae_global_dim_embed` | 2048 / 2048 | latent width |
| `ae_local_num_blocks / ae_aggregation_num_blocks` | 0 / 0 | those engines are identity |
| `ae_global_num_blocks / fe_num_blocks` | 4 / 16 | dense attention depth |
| `decoder_type` | `PerceiverIOCoordConditioning` | `TargetPredictionEngineClassic` |
| `with_mixed_precision / attention_dtype` | True / bf16 | autocast bf16, q/k cast to bf16 |

---

## 2. Data: from zarr to `ModelBatch`

### 2.1 Reader

`MultiStreamDataSampler.__init__` (`datasets/multi_stream_data_sampler.py:95-154`) builds one
`DataReaderAnemoi` per stream file. **Observed** for `ERA5`:

```
ds.shape          (64281, 113, 1, 40320)      # (time, variable, ensemble, gridpoint)
period            6 h                         # 64281 = 44 yrs × 365.25 × 4
data_start/end    1979-01-01T00 … 2022-12-31T00
latitudes.shape   (40320,)                    # o96 reduced Gaussian ≈ 1°
len(source_channels) = 80   len(target_channels) = 81   len(geoinfo_channels) = 9
```

`source_channels` = 113 vars minus `source_exclude` minus geoinfo/constant vars
(`data_reader_anemoi.py:256-295`); `target_channels` is the same set with `skt` in and
`slor, sdor` out (`era5.yml:14-15`). `geoinfo_channels` (`z, lsm, slor, sdor, insolation, cos/sin
local time, cos/sin julian day`) ride along every point as conditioning.

`mean.shape = stdev.shape = (113,)` come from the zarr `statistics`; all channels are fed as
`(x-mean)/stdev`. **Observed** `target_tokens[1]`: mean −0.0086, std 0.9996 — i.e. the loss below is
in normalised units and an untrained model has MSE ≈ 1 per channel.

### 2.2 Time windows and forecast steps

`TimeWindowHandler.window(idx)` = `[start + idx·step, +len)`. **Observed** sample `idx=36234` →
`[2003-10-20T12:00, 2003-10-20T18:00)`. `_get_data_windows` (`:561-611`):

```python
for idx in range(base_idx - num_steps_input_max + 1, base_idx + 1):      # 1 input step → [36234]
    rdata = collect_datasources(stream_ds, idx, "source", self.rng)
for timestep_idx in range(self.output_offset, num_output_steps):          # [1, 2, 3]
    step_forecast_dt = base_idx + (self.time_step * timestep_idx) // self.step_timedelta
    rdata = collect_datasources(stream_ds, step_forecast_dt, "target", self.rng)
```

**Observed** target times: `target_times_raw[1][0] = 2003-10-20T18:00`,
`[2] = 2003-10-21T00:00`, `[3] = 2003-10-21T06:00`. Index 0 of every per-step list is the
(empty) "current" slot because `offset=1`; the model never decodes it.

`collect_datasources(..., "target")` subsamples `max_num_targets = 20000` of the 40320 points
(`:66-76`), so each target step is a *different* random 20000-point subset.

### 2.3 HEALPix tokenisation

`TokenizerMasking.get_tokens_windows` → `tokenize_spacetime` → `hpy_splits`
(`datasets/tokenizer_utils.py:124-173`): every point gets its nested HEALPix cell at level 5,
points within a cell are sorted, chunked into groups of `token_size = 8`, and (sources only) the
last chunk is padded with index 0, which addresses an all-zero row prepended to the data
(`tokenize_apply_mask_source`, `:271-274`).

**Observed** source tokens:

```
source_tokens_cells[0]   (12296, 8, 97) float32
source_tokens_lens[0]    (12288,) int64   sum 12296, min 1, max 2   (8 cells hold >8 points → 2 tokens)
padded rows              58048 of 98368   (= 98368 − 40320 real gridpoints)
```

The 97 columns of a source row (`tokenize_apply_mask_source`, `:299-300`):

```
tokens = torch.cat((stream_ids, datetimes, coords_local, geoinfos, data), 1)
          1 + 5 + 2 + 9 + 80 = 97
```

* `stream_ids` — `stream_id = 0` for ERA5 (**observed** column 0 is all zero).
* `datetimes` (`encode_times_source`, `:27-60`) — `[year/2100, doy/365, minutes/1440, sin, cos of Δt in window]`.
  **Observed** first row: `[0.954, 0.803, 0.5, 0, 1]` = 2003, day 293, 12:00, Δt=0.
* `coords_local` — the point rotated into its cell's local frame, as 2 spherical angles
  (`get_source_coords_local`, `:397-418`). Padded rows keep a non-zero `(0.021, 2.356)` here because
  `r3tos2` of the zero vector is not zero; every other column of a padded row is 0.
* `geoinfos` (9), `data` (80).

**Observed** target-coordinate rows: `target_coords[k]` is `(20000, 114)` for `k=1..3` and `(0,)` at
`k=0`. Layout (`get_target_coords_local`, `:448-522`):

```
1 (stream id) + 5 (encode_times_target: sin/cos Δt + 0.5) + 9 geoinfo
+ 5 × (3 + 12)   local coords w.r.t. the 4 cell vertices + centre, each with its 4 neighbour-vertex offsets
+ 3 × 8          offsets to the 8 neighbouring cell centres
= 114
```

`target_coords_lens[k]` is `(12288,)` = number of target points per cell (**observed** max 8,
sum 20000). This is what makes the varlen decoder attention work.

### 2.4 Masking

`Masker._generate_cell_mask` with `masking_strategy: forecast` (`datasets/masking.py:514-560`)
produces an all-`True` cell mask (**observed** `mask.sum() = 12288` for both source and target).
Forecasting is therefore "masked-token modelling" where the mask is *in time*, not in space:
all cells are input, all cells (at later times) are target.

### 2.5 The batch object

`ModelBatch` (`datasets/batch.py:278-506`) separates what the model sees from what the loss sees.
**Observed** (`/tmp/wg_batch.json` dump at `trainer.py:456`):

```
ModelBatch
├ output_offset 1, output_steps 4, output_idxs [1,2,3]
├ source2target_matching_idxs [0]     target2source_matching_idxs [[0]]
├ source_samples : BatchSamples
│   ├ tokens_lens (1, 1, 1, 12288) int64     # (input_steps, samples, streams, cells)
│   └ samples[0]
│       ├ meta_info['ERA5'] = SampleMetaData(params={masking_strategy: forecast}, mask (12288,) bool,
│       │                      global_params={idx:0, correspondence:0, loss:['mse'], relationship:'independent'})
│       └ streams_data['ERA5'] : StreamData
│           source_tokens_cells [ (12296, 8, 97) ]     source_tokens_lens [ (12288,) ]
│           target_coords       [ (0,), (20000,114) ×3 ] target_coords_lens [ (12288,) ×4 ]
│           target_tokens       [ (0,) ×4 ]             # sources carry no target values
└ target_samples : BatchSamples
    ├ tokens_lens (1, 1, 1, 12288) int32  (all zero: no source tokens on the target side)
    └ samples[0].streams_data['ERA5']
            target_tokens      [ (0,), (20000,81) ×3 ]
            target_coords_raw  [ [], (20000,2) ×3 ]     # lat/lon in degrees (observed row: 2.34, -135.0) → cosine_latitude weights
            target_times_raw   [ [], (20000,) ×3 ]
            source_tokens_cells [ None ]
```

`trainer.train` hands `batch.get_source_samples()` to the model and
`batch.get_target_samples(...)` to the target/aux calculator (`train/trainer.py:463-478`).
The `correspondence`/`idx` pair in `global_params` is how the loss later pairs prediction 0 with
target 0 (`loss_module_physical.py:332-343`).

---

## 3. Non-trainable geometry: `ModelParams`

`ModelParams.reset_parameters` (`model/model.py:163-262`). **Observed** shapes:

```
pe_embed     (64, 2048)        bf16   sinusoid over token rank within a cell (max 64 tokens/cell)
pe_global    (12288, 1, 2048)  bf16   sinusoid over cell index (+ query index)
hp_nbours    (12288, 9)        int    [self, 8 nested HEALPix neighbours]; hp_nbours[0] = [0,4437,4439,2,3,1,5803,5802,9215]
q_cells_lens (12289,)          int    [0, 1, 1, …]  cumsum-ready lens: one query per cell
rope_coords / rope_cell_coords = None   (rope_2D: False)
```

There is no learned positional embedding and no time-step embedding anywhere; position enters
via `pe_embed` (inside a cell), `pe_global` (which cell), and the explicit local coordinates in
the token columns.

---

## 4. `Model.forward`

```python
def forward(self, model_params: ModelParams, batch: ModelBatch) -> ModelOutput:      # model.py:672
    output = ModelOutput(batch.get_output_len())                                       # 4 slots
    tokens, posteriors = self.encoder(model_params, batch)                            # (1, 12288, 2048)
    shape = (len(batch), batch.get_num_source_steps(), *tokens.shape[1:])
    tokens = tokens.reshape(shape).sum(axis=1)                                        # sum over input steps
    for step in batch.get_output_idxs():                                              # [1, 2, 3]
        tokens = self.forecast_engine(tokens, step, model_params.rope_coords)
        output = self.predict_decoders(model_params, step, tokens, batch, output)
        output = self.predict_latent(model_params, step, tokens, batch, output)
    return output
```

Module tree (**observed** `named_children`) and parameter counts:

```
Model                                   739 652 438
├ encoder : EncoderModule               189 950 965
│  ├ embed_engine        EmbeddingEngine               5 348 341
│  ├ ae_local_engine     LocalAssimilationEngine               0   (0 blocks)
│  ├ ae_local_global_engine Local2GlobalAssimilationEngine 50 341 888
│  ├ ae_aggregation_engine QueryAggregationEngine             0   (0 blocks)
│  └ ae_global_engine    GlobalAssimilationEngine     134 258 688
├ forecast_engine : ForecastingEngine   537 034 752
├ embed_target_coords['ERA5'] NamedLinear     58 368
├ target_token_engines['ERA5'] TargetPredictionEngineClassic 12 562 704
├ pred_heads['ERA5'] EnsPredictionHead        41 553
├ latent_heads (empty)                             0
└ latent_pre_norm LayerNorm(2048)              4 096
```

### 4.1 Embedding: one small transformer per stream

`EncoderModule.forward` (`model/encoder.py:120-140`) first calls
`EmbeddingEngine.forward` (`model/engines.py:81-132`):

```python
sdata = torch.cat(sdata).to(tokens_all.dtype)                 # (12296, 8, 97) → bf16
x_embeds += [self.embeds[stream_name](sdata).flatten(0, 1)]   # (12296, 2048)
...
tokens_all = torch.cat(x_embeds)                              # single stream: no scatter needed
pe_idxs = self.get_pe_idxs_vectorized(batch)                  # rank of each token inside its cell
tokens_all = tokens_all + pe_embed[pe_idxs]
```

`StreamEmbedTransformer` (`model/embeddings.py:21-129`), built from `era5.yml: embed:
{net: transformer, dim_embed: 512, num_blocks: 2, num_heads: 8, num_tokens: 1}` and
`embed_unembed_mode: block`. The trick is the transpose:

```python
x = peh(checkpoint(self.embed, x_in.transpose(-2, -1), use_reentrant=False))   # (12296, 97, 8) → (12296, 97, 512)
for layer in self.layers:                                                       # 2 × [MSA(8 heads), MLP]
    x = checkpoint(layer, x, use_reentrant=False)
out = [ue(ln(x[:, i])) for i, (ue, ln) in enumerate(zip(self.unembed, self.ln_final))]   # 97 × Linear(512→21)
out = torch.stack(out, dim=1).flatten(-2, -1)                                    # (12296, 2037)
if out.shape[-1] < self.dim_out:
    out = torch.nn.functional.pad(out, [0, self.dim_out - out.shape[-1]])        # → (12296, 2048)
```

So inside a token the *97 columns* are the sequence and the *8 points* are the feature vector:
attention mixes channels, not points. Each column is then read out to
`2048 // 97 = 21` latent dims, concatenated and zero-padded by 11. **Observed** output
`stream_cell_tokens: (12296, 2048) bf16`.

### 4.2 Local → global: learnable per-cell queries

`EncoderModule.assimilate_local` (`encoder.py:275-355`) turns 12296 ragged tokens into exactly
one latent per cell:

```python
cell_lens = torch.sum(batch.tokens_lens, 2).flatten()                    # (12288,)  tokens per cell
rs = num_steps_input * len(batch)                                         # 1
tokens_global = self.q_cells.repeat(num_tokens, 1, 1) + model_params.pe_global   # (12288, 1, 2048)
tokens_global = tokens_global.repeat(rs, 1, 1)
tokens_global_unmasked, posteriors = self.assimilate_local_project_chunked(
    tokens, tokens_global, cell_lens, model_params.q_cells_lens)
```

`q_cells` is a single learned vector `(1, 1, 2048)` (**observed**); adding `pe_global` gives
every cell a distinct query. `assimilate_local_project_chunked` (`:156-216`) processes cells in
two halves (`clen = 12288 // 2`, a flash-attention workaround). **Observed** chunk 0:

```
toks            (6148, 2048) bf16     # tokens of cells 0..6143  (l0=0, l1=6148)
toks_global     (6144, 1, 2048)
cell_lens_cur   (6145,)  [0, 1, 1, 1, …]   # leading 0 for cumsum
toks (after ae_local_engine)  unchanged     # 0 local blocks
toks_global_unmasked (6144, 1, 2048) fp32   # after ae_local_global_engine
```

`Local2GlobalAssimilationEngine` (`engines.py:247-316`) is `[CrossAttn, MLP, CrossAttn]` where
the cross-attention is `MultiCrossAttentionHeadVarlenSlicedQ` (`model/attention.py:410-524`):

```python
qs = [self.lnorm_q(head_proj(x_q_i).reshape(s)).to(self.dtype)              # one Linear per query slice (1 here)
      for head_proj, x_q_i in zip(self.proj_heads_q, x_q.transpose(1, 0))]  # (6144, 16 heads, 128)
ks = self.lnorm_k(self.proj_heads_k(x_kv).reshape(s)).to(self.dtype)         # (6148, 16, 128)
vs = self.proj_heads_v(x_kv).reshape(s)
cum_x_q_lens  = torch.cumsum(x_q_lens, 0, dtype=torch.int32)                 # q_cells_lens: 1 query per cell
cum_x_kv_lens = torch.cumsum(x_kv_lens, 0, dtype=torch.int32)                # cell_lens: 1–2 tokens per cell
outs += [flash_attn_varlen_func(qs_i, ks, vs, cum_x_q_lens, cum_x_kv_lens, x_q_lens.max(), x_kv_lens.max(), ...)]
```

Varlen attention with matching cumulative lengths means *query of cell c attends only to the
tokens of cell c*. Sizes: `ae_adapter_num_heads = 16`, `ae_adapter_embed = 128` head dim, residual
on the query side. After both chunks are concatenated and the (empty) register/class tokens and the
(identity) aggregation engine are applied:

```python
tokens_global[mask] = tokens_global_unmasked.to(tokens_global.dtype)   # encoder.py:345  (mask all True here)
tokens_global = tokens_global.reshape([rs, num_tokens_tot, 1, 2048]).flatten(1, 2)
```

**Observed** at `encoder.py:323`: `tokens_global_unmasked (12288, 2048) fp32`, final
`tokens_global (1, 12288, 2048) fp32`. Cells with no input tokens would keep their
`q_cells + pe_global` prior — that is the mechanism used by the masked (MTM/JEPA) configs;
here every cell has data.

### 4.3 Global assimilation: dense attention over 12288 cells

`GlobalAssimilationEngine` (`engines.py:455-529`): with `ae_global_att_dense_rate = 1.0` every one
of the 4 blocks is a dense `MultiSelfAttentionHead(2048, 32 heads)` (head dim 64) + `MLP(2048→4096→2048)`.
`MultiSelfAttentionHead.forward` (`attention.py:590-617`):

```python
x = self.lnorm(x)                                                    # LayerNorm(2048, no affine, eps 1e-4)
qs = self.lnorm_q(self.proj_heads_q(x).reshape(s)).to(self.dtype)    # (1, 12288, 32, 64) bf16, per-head LN
ks = self.lnorm_k(self.proj_heads_k(x).reshape(s)).to(self.dtype)
vs = self.proj_heads_v(x).reshape(s).to(self.dtype)
outs = flash_attn_func(qs, ks, vs, softcap=self.softcap, dropout_p=dropout_rate)
out = self.proj_out(outs.flatten(-2, -1)) + x_in
```

Full 12288² attention, bf16 via flash-attn; each block is `torch.utils.checkpoint`ed
(`engines.py:528`). **Observed** in/out `(1, 12288, 2048) fp32` (residual stream stays fp32 under
autocast). The `MultiSelfAttentionHeadLocal` block-sparse variant (`block_factor: 64`) is compiled
only when a dense rate < 1 is set and is not instantiated in this run.

### 4.4 Forecast engine: autoregressive rollout in latent space

```python
tokens = tokens.reshape(shape).sum(axis=1)             # (1, 1, 12288, 2048) → (1, 12288, 2048)
for step in [1, 2, 3]:
    tokens = self.forecast_engine(tokens, step, None)
```

`ForecastingEngine.forward` (`engines.py:623-636`):

```python
if self.training:
    noise_std = self.cf.get("fe_impute_latent_noise_std", 0.0)          # 1e-4
    tokens = tokens + torch.randn_like(tokens) * torch.norm(tokens) * noise_std
for block in self.fe_blocks:                                             # 16 × [MSA(16 heads, dim 128), MLP], LN after block 7
    tokens = checkpoint(block, tokens, coords, aux_info, use_reentrant=False)
```

Note that `fstep` is accepted and ignored: the same 537 M parameters advance the state by one
`time_step` (6 h) each call, and step 2 is applied to the *output* of step 1. Weights are
initialised `N(0, 0.001)` (`engines.py:614-621`) so at init the engine is close to identity.
**Observed** after step 1: `(1, 12288, 2048) fp32`, `‖tokens‖ = 5017`. `pushforward` is off, so
all three steps keep their graph and gradients flow through the whole rollout.

### 4.5 Decoding to arbitrary coordinates

`Model.predict_decoders` (`model.py:734-839`) is a Perceiver-IO-style read-out: each target point
becomes a query that attends to the latents of its own cell and the 8 neighbours.

```python
s = [batch_size, 12288, 1, 2048]
idxs = model_params.hp_nbours.unsqueeze(0).repeat((batch_size, 1, 1)).flatten(0, 1)     # (12288, 9)
tokens_nbors = tokens.reshape(s).flatten(0, 1)[idxs.flatten()].flatten(0, 1)             # (110592, 2048)
tokens_nbors_lens = torch.full((12288 + 1,), 9, dtype=torch.int32); tokens_nbors_lens[0] = 0
```

**Observed** `tokens_nbors (110592, 2048) fp32` = 12288 × 9. Then per stream:

```python
t_coords = torch.cat([batch.samples[i].streams_data['ERA5'].target_coords[step] ...])   # (20000, 114)
tc_tokens = checkpoint(self.embed_target_coords['ERA5'], t_coords)                       # Linear(114→512) → (20000, 512) bf16
tcls = torch.cat([sample.streams_data['ERA5'].target_coords_lens[step] ...])             # (12288,) points per cell
tcs_lens = torch.cat([torch.zeros(1), tcls])                                             # (12289,) [0, 2, 2, 1, 1, …]
tc_tokens = self.target_token_engines['ERA5'](
    latent=tokens_nbors, output=tc_tokens, latent_lens=tokens_nbors_lens,
    output_lens=tcs_lens, coordinates=t_coords)                                          # (20000, 512)
pred = self.pred_heads['ERA5'](tc_tokens)                                                # (1, 20000, 81)
pred = torch.split(pred, t_coords_lens, dim=1)                                           # list per batch sample
output.add_physical_prediction(step, 'ERA5', pred)
```

`TargetPredictionEngineClassic` (`engines.py:694-798`), `target_readout: {num_layers: 2, num_heads: 4}`:

```
2 × [ MultiCrossAttentionHeadVarlen   q: targets of cell c (512)   kv: 9 latents of cell c (2048→512), AdaLN(q | coords)
      MultiSelfAttentionHeadVarlen    among the targets of cell c                                 AdaLN(coords)
      MLP(512→1024→512)                                                                            AdaLN(coords)  ]
```

Every norm in the read-out is an `AdaLayerNorm` conditioned on the raw 114-dim coordinate row
(`Linear(114→456) → SiLU → Linear(456→1024)` producing scale/shift), i.e. "coord conditioning".
`EnsPredictionHead` with `ens_size: 1, num_layers: 1` is a single `Linear(512→81)`; the leading
dimension is the ensemble axis.

**Observed** at `model.py:836`, step 1: `tc_tokens (20000, 512) bf16`, `pred (1, 20000, 81) bf16`.

### 4.6 `ModelOutput`

**Observed** at `trainer.py:480`:

```
preds.physical = [ {}, {'ERA5': [(1,20000,81)]}, {'ERA5': [(1,20000,81)]}, {'ERA5': [(1,20000,81)]} ]
preds.latent   = [ {'posteriors': [0., 0.]}, {'latent_state': LatentState}, {…}, {…} ]
```

`predict_latent` stores the raw latent per step (`LatentState.z_pre_norm`); `latent_heads` is
empty for this loss set, and `latent_pre_norm` only runs at `step == 0`, which never happens with
`offset=1` — both are inert here and exist for the SSL configs.

---

## 5. Targets and loss

### 5.1 Targets

`PhysicalTargetAndAux.compute` (`train/target_and_aux_module_base.py:98-132`) just regroups the
target side of the batch. **Observed** `targets_and_auxs['physical'].physical`:

```
[ {}, {'ERA5': {target: [(20000,81)], target_times: [(20000,)], target_coords: [(20000,2)],
                target_metda_data: [dict], is_spoof: [False], idxs_inv: [None]}}, {…step 2…}, {…step 3…} ]
```

### 5.2 `LossPhysical`

`LossCalculator.compute_loss` (`train/loss_calculator.py:81-114`) sums `weight × module.loss`
over configured loss terms — one here. `LossPhysical.compute_loss`
(`train/loss_modules/loss_module_physical.py:241-460`), reduced to what executes for this config:

```python
for stream_name, stream_info in self.cf.streams.items():                        # ERA5
    stream_loss_weight, weights_channels = self._get_weights(stream_name, stream_info)
    #   TRAIN: loss_weight 1.0, weights_channels = target_channel_weights (81,)  e.g. q_10 0.2 … q_1000 0.8, surface 1.0
    for timestep_idx, (preds_cur, target_cur) in enumerate(zip(preds.physical, targets.physical)):
        if not preds_cur.get(stream_name): continue                              # step 0 skipped
        for pred, pred_params in zip(preds_batch, output_info):                  # per batch sample
            target_idx = ... global_params["correspondence"] ↔ global_params["idx"]
            substep_masks    = self._get_substep_masks(...)                      # unique target_times → 1 mask (all True)
            weights_locations = self._get_location_weights(...)                  # cosine_latitude(target_coords_raw)
            for loss_fct, loss_fct_weight, loss_fct_name in self.loss_fcts:      # [(mse, 1.0, 'mse')]
                pred = pred.reshape([pred.shape[0], *target.shape])              # (1, 20000, 81)
                loss_lfct, loss_lfct_chs = self._loss_per_loss_function(
                    loss_fct, target, pred, substep_masks, weights_channels, weights_locations)
                loss_st_corr = loss_st_corr + spoof_weight * loss_fct_weight * loss_lfct * output_step_weight
            loss_timestep = loss_timestep + loss_st_corr
        loss_stream = loss_stream + loss_timestep
        ctr_timesteps += 1
    loss = loss + (stream_loss_weight * loss_stream) / ctr_timesteps             # mean over 3 steps
loss = loss / ctr_streams                                                        # mean over streams
```

and `mse` → `lp_loss(p=2)` (`train/loss_modules/loss_functions.py:126-224`):

```python
mask_nan = ~torch.isnan(target)
pred = pred.mean(0)                                            # ensemble mean → (20000, 81)
diff_p = (where(mask_nan, target, 0) - where(mask_nan, pred, 0)) ** 2
diff_p = (diff_p.transpose(1, 0) * weights_points).transpose(1, 0)   # cos(lat) per point
loss_chs = diff_p.mean(0)                                      # (81,) per-channel MSE
loss = torch.mean(loss_chs * weights_channels)                 # channel-weighted scalar
```

So, for this run,

$$
\mathcal{L} \;=\; \frac{1}{3}\sum_{k=1}^{3}\;\frac{1}{C}\sum_{c=1}^{81} w_c \cdot
\frac{1}{N}\sum_{i=1}^{20000} \cos(\phi_i)\,\big(y^{(k)}_{i,c}-\hat y^{(k)}_{i,c}\big)^2 .
$$

**Observed**: `loss = 0.4140` (fp32 scalar, `requires_grad=True`); `losses_unweighted_hist` holds
per-channel/per-step values used only for logging, e.g. `10u: {1: 0.78, 2: 0.71, 3: 0.79}`,
`2t: {1: 0.46, 2: 0.43, 3: 0.42}` — a freshly initialised model already sits below 1 because
weighted channels (upper-level `q_*`, `t_*` with weights 0.2–0.8) pull the average down.

---

## 6. The optimisation step

`Trainer.train` (`train/trainer.py:434-571`):

```python
batch.to_device(self.device)
with torch.autocast(device_type=f"cuda:{cf.local_rank}", dtype=bf16, enabled=True):
    preds = self.model(model_params=self.model_params, batch=batch.get_source_samples())
    targets_and_auxs[loss_name] = target_aux.compute(istep, batch.get_target_samples(target_idxs), ...)
loss = self.loss_calculator.compute_loss(preds=preds, targets_and_aux=targets_and_auxs, metadata=extract_batch_metadata(batch))

self.optimizer.zero_grad()
self.grad_scaler.scale(loss).backward()
self.grad_scaler.unscale_(self.optimizer)
total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
self.grad_scaler.step(self.optimizer)
self.grad_scaler.update()
self.lr_scheduler.step()
if self.validate_with_ema:
    self.ema_model.update(self.cf.general.istep * batch_size_total, batch_size_total)
self.cf.general.istep += 1
```

**Observed** on the first step:

| quantity | value |
|---|---|
| `GradScaler.get_scale()` | 65536 (default; a `GradScaler` is created because `optimizer.grad_scaling` defaults to True) |
| `total_norm` (pre-clip, unscaled) | 1.578 → clipped to 1.0 |
| AdamW `eps`, `weight_decay` | 2e-8, 0.1 (`fused=True`) |
| AdamW `betas` | (0.95, 0.9875) |
| LR before → after `lr_scheduler.step()` | 1.0e-6 → 1.0019e-6 |
| `lr_steps` | 262 144 = 4096 samples × 64 mini-epochs / batch 1 |
| warmup / decay / cooldown steps | 256 (cosine, `OneCycleLR`) / 261 376 (constant at 5e-5) / 512 (linear) |
| EMA | `EMAModel`, half-life 600k samples, ramp-up ratio 0.09 |
| peak GPU memory after forward / after backward | 21.6 GB / 35.7 GB (`torch.cuda.max_memory_allocated`) |
| wall time of step 0 | 232 s (debugger + in-process data loading; not representative) |

Two things worth knowing that the config does not make obvious:

* `beta1`. `trainer.py:326-331` derives `beta1 = 1 − κ(1 − 0.98125)` with `κ = world_size × batch`
  (= 0.98125 here), but the observed `betas[0]` is **0.95**: `OneCycleLR` is constructed with
  the default `cycle_momentum=True` (`train/lr_scheduler.py:105-112`) and overwrites `beta1`
  every step during warmup (0.95 → 0.85). After warmup the "constant" decay policy stops calling
  the scheduler (`lr_scheduler.py:213-219`), so `beta1` stays at the last value `OneCycleLR` wrote
  [INFERENCE: 0.85 at the end of warmup, from `OneCycleLR` semantics; not observed].
* `with_fsdp: True` only takes effect with `world_size > 1`. In this single-process run there is no
  sharding and no DDP; the 739.7 M fp32 parameters (2.96 GB) plus AdamW state fit comfortably.

Logging cadence (`train_logging`): terminal every 10 batches, metrics every 20, `_latest.chkpt`
every 250; after each mini-epoch `validate` runs 256 samples through `ema_model.forward_eval` and
`save_model(mini_epoch)` writes `<run_id>_chkpt<N>.chkpt` (`trainer.py:391-411, 573-656, 704-730`).

---

## 7. Attention variants in play

| class (`model/attention.py`) | kernel | used by | ragged? |
|---|---|---|---|
| `MultiSelfAttentionHead` (:527) | `flash_attn_func` on `(B, S, H, D)` | stream embedder (512/8h), global AE (2048/32h), forecast engine (2048/16h) | no — dense over 12288 cells |
| `MultiCrossAttentionHeadVarlenSlicedQ` (:410) | `flash_attn_varlen_func`, one Q projection per query slice | local→global adapter (q 2048, kv 2048, 16h×128) | yes — cell lens |
| `MultiCrossAttentionHeadVarlen` (:305) | `flash_attn_varlen_func` | decoder read-out (q 512, kv 2048→512, 4h×128) | yes — targets/cell vs 9 latents/cell |
| `MultiSelfAttentionHeadVarlen` (:27) | `flash_attn_varlen_func` | decoder self-attention among targets of a cell; local AE / aggregation (0 blocks here) | yes |
| `MultiSelfAttentionHeadLocal` (:210) | `flex_attention` block mask, `block_factor` | only when `*_att_dense_rate < 1` | — (not built) |

All of them: pre-LayerNorm without affine (`norm_eps 1e-4`), bias-free q/k/v/out projections,
per-head LayerNorm on q and k (`with_qk_lnorm`), q/k cast to bf16 before the kernel, residual add
in fp32, dropout 0.1 inside the kernel during training.

---

## 8. Where each config knob lands

```
era5.yml
  token_size 8 ───────────────► StreamEmbedTransformer.embed = Linear(8 → 512)
  embed.dim_embed 512, num_blocks 2, num_heads 8 ─► 2 × [MSA, MLP] over the 97 channel-columns
  embed_target_coords.dim_embed 512 ───────────────► NamedLinear(114 → 512), TTE width
  target_readout.num_layers 2, num_heads 4 ────────► TargetPredictionEngineClassic depth / heads
  pred_head.ens_size 1, num_layers 1 ──────────────► EnsPredictionHead = [Linear(512 → 81)]
  channel_weights ─────────────────────────────────► weights_channels in lp_loss (train only)
  location_weight cosine_latitude ─────────────────► weights_points in lp_loss
  max_num_targets 20000 ───────────────────────────► points per target step
config_forecasting.yml
  healpix_level 5 ─────────────────────────────────► 12288 cells everywhere
  ae_local_dim_embed 2048 ─────────────────────────► embedder dim_out (97×21 + pad), pe_embed
  ae_adapter_num_heads 16, ae_adapter_embed 128 ───► SlicedQ cross-attention heads / head dim
  ae_global_num_blocks 4, ae_global_num_heads 32 ──► GlobalAssimilationEngine
  fe_num_blocks 16, fe_num_heads 16, fe_layer_norm_after_blocks [7] ─► ForecastingEngine
  fe_impute_latent_noise_std 1e-4 ─────────────────► train-time latent noise before each rollout step
  forecast.offset 1, num_steps 3, time_step 6h ────► output_idxs [1,2,3]; target windows t+6h..t+18h
  with_flash_attention / attention_dtype bf16 ─────► flash-attn kernels, q/k dtype
  training_config.optimizer / learning_rate_scheduling ─► §6
```
