# AdaLN initialization defects

## Findings and fixes

The user reported a possible difference in AdaLN initialization between model paths. CPU reproduction confirmed two defects in the gated `AdaLayerNormLayer` implementation and its callers. The existing ungated `AdaLayerNorm` is a different formulation; different initialization between those two formulations alone is not evidence of a bug, and the ungated implementation is unchanged.

### Zero modulation was lost during reinitialization

`AdaLayerNormLayer` is documented as AdaLN-Zero: its shift, scale and residual gate start at zero, so its initial output must equal its input. Ordinary construction restored those zeros, but later initialization did not:

- `Model.reset_parameters()` recursively calls `nn.Linear.reset_parameters()`. This overwrote the modulation projection with ordinary random initialization.
- Fresh FSDP startup constructs the model on `meta`, materializes it with `to_empty`, and calls that model reset. Constructor-time zeroing therefore did not survive fresh distributed initialization.
- `SelfAttentionBlock.initialise_weights()` and `CrossAttentionBlock.initialise_weights()` recursively applied Xavier initialization. Their constructor-only zero-restoration calls did not protect subsequent calls to these methods.
- Initialization of newly added checkpoint leaf modules also invokes the projection's `reset_parameters()` method.

The fix puts the zero-reset policy on a small private `nn.Linear` subclass used only for gated AdaLN modulation. This preserves zero initialization through ordinary model and leaf resets without changing parameter names, shapes or the linear forward operation. Attention-block initialization now restores AdaLN zeros during the post-order initialization traversal rather than only in the constructors.

Changed implementation files:

- `src/weathergen/model/norms.py`: `_ZeroInitLinear` and `AdaLayerNormLayer`.
- `src/weathergen/model/blocks.py`: both attention-block initialization traversals.

`model.py` and `model_interface.py` require no special-case changes: their existing leaf-reset calls now use the modulation projection's correct reset policy.

### Dropout was accidentally used as LayerNorm epsilon

All five gated-AdaLN construction sites in the attention blocks passed `dropout_rate` as the fourth positional argument. That argument is actually `norm_eps`. Consequently, dropout `0.1` produced epsilon `0.1`, and dropout `0.0` produced epsilon `0.0`.

With epsilon zero, a constant token produces NaNs during normalization. Multiplying that branch by a zero residual gate does not remove the NaNs. The callers now pass the configured `attention_kwargs["norm_eps"]` explicitly by keyword. This does not change the attention/MLP dropout settings.

## Reproduction and verification

The CPU probe used native attention-block construction, the native gated MLP branch, and `Model.reset_parameters()`. It did not run FlashAttention kernels or initialize CUDA.

| Initialization path | Maximum identity error before | After |
|---|---:|---:|
| Ordinary construction, nonconstant inputs | 0 | 0 |
| CPU model reset | 0.503978 | 0 |
| `meta` → CPU materialization → model reset | 0.232905 | 0 |
| Self-attention block reinitialization | 0.538359 | 0 |
| Cross-attention block reinitialization | 0.521757 | 0 |
| Modulation projection leaf reset | 0.670453 | 0 |

With zero dropout, constant-token output changed from nonfinite to finite. Both tested dropout settings now use the requested epsilon `1e-5` rather than the dropout probability.

Regression checks are in `tests/test_adaln_initialization.py`. All seven checks failed before the fix and passed afterward. They cover the distinct initialization paths, constant-token identity, learning away from the initial identity, and restoring learned conditioning through a state-dict round trip.

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv/bin/python -m pytest tests/test_adaln_initialization.py -q
```

Ruff formatting and checks also passed for the two changed implementation files and the regression module. Full distributed/FSDP execution was not run: the test exercises the same `meta` materialization and reset operations on CPU, and the same projection-reset primitive used for new checkpoint modules, not a distributed checkpoint-loading session.

Before/after probe output, the runnable probe text and original source hashes are retained under `/capstor/scratch/cscs/rradev/cerra_decoder_artefacts/adaln_initialization/`.

## Existing checkpoints and CERRA experiments

- State-dict parameter names and shapes remain compatible. Loading learned modulation parameters does not zero them; this is covered by the regression check.
- **Correcting epsilon can change the outputs of existing gated-AdaLN checkpoints**, because epsilon is constructor configuration rather than a saved parameter. Preserve the original code snapshot for exact old-checkpoint inference; use fresh, matched-budget runs for comparisons involving the corrected gated blocks.
- Initialization RNG consumption can also differ for affected gated-AdaLN models. This is not a promise of identical fresh-run weights.
- The original CERRA checkpoints (`caea0001`, 400 updates; `caea0002`, 2,560 updates) contain neither gated-AdaLN modulation parameters nor ungated adaptive-normalization parameters. Their configurations use the Linear/Bilinear decoder with adaptive prediction MLP normalization disabled. These defects therefore do not explain the observed seams in those runs, and the already-completed 2,560-step MLP result does not need repeating for this fix.
- The matched **2,560-update** linear rerun is now complete as experiment **09**, with an independent checkpoint reload. The historical 400-step run remains documented but is excluded from the matched comparison. See the current [CERRA experiment report](cerra_decoder_artefacts.md) for metrics, maps and native loss curves.
