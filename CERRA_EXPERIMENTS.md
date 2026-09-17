# CERRA output-variable experiments

Base config: `config/forecasting_cerra_3h.yml` (CERRA 5.5 km, 3 h, `trim_edge: 100`).
All four experiments use the **same 72 input channels** (7 surface: `skt sp 10si 10wdir 2r 2t msl`;
5 upper-air variables `r t u v z` x 13 levels 50-1000 hPa) and differ only in what is decoded.
Each run is a chain of 4 x 12 h Slurm jobs on JUPITER booster (`--chain-jobs 4`, QOS `normal` caps a job at 12 h).

## Experiments

| # | Name | Streams dir | Encoder / decoder layout | Output | Question |
|---|------|-------------|--------------------------|--------|----------|
| 1 | r850 | `config/streams/cerra_r850/` | 1 stream `CERRA`, 1 encoder, 1 decoder | `r_850` | Reference. Known to work well; high-resolution structures visible in the output. |
| 2 | z500 | `config/streams/cerra_z500/` | 1 stream `CERRA`, 1 encoder, 1 decoder | `z_500` | Same setup on a smooth, large-scale field: does the model still produce fine-scale structure, or is the r850 result specific to a high-variance target? |
| 3 | r850+z500 (single decoder) | `config/streams/cerra_r850_z500/` | 1 stream `CERRA`, 1 encoder, 1 decoder with 2 output channels | `r_850`, `z_500` | Does sharing one decoder between a fine-scale and a smooth target degrade either (loss balancing, smoothing of r850)? |
| 4 | r850 + z500 (two decoders) | `config/streams/cerra_2dec_r850_z500/` | `CERRA_in` (`forcing: True`, encoder only) + `CERRA_r850` and `CERRA_z500` (`diagnostic: True`, one decoder each) | `r_850` (dec 1), `z_500` (dec 2) | Same targets as #3 but with separate decoder heads on a shared latent: isolates decoder capacity/interference from encoder effects. |

## Runs

| # | Streams dir | run_id | Slurm job ids (part1..4, cleanup) | Command | Status |
|---|-------------|--------|-----------------------------------|---------|--------|
| 1 | `cerra_r850` | - | - | - | not submitted |
| 2 | `cerra_z500` | `gzri183a` | 1856371, 1856372, 1856373, 1856374, cleanup 1856375 | `../WeatherGenerator-private/hpc/launch-slurm.py --stage train --chain-jobs 4 --base-config config/forecasting_cerra_3h.yml --options streams_directory=./config/streams/cerra_z500/` | **failed** 2026-09-17 14:16, part1 after 2 min: `PermissionError` on `/e/data1/slmet/ml_training/cerra-...-3h-v2.zarr` (user not in group `slmet`). Parts 2-4 will start and fail the same way; cancel with `scancel 1856372 1856373 1856374 1856375`. |
| 3 | `cerra_r850_z500` | - | - | - | not submitted |
| 4 | `cerra_2dec_r850_z500` | - | - | - | not submitted |

`vvv73tsa` (2026-09-17 14:07, z500): launcher aborted before `sbatch` (15 tracked files under `config/` deleted in the working tree; `git ls-files` copy failed). No jobs.

### Submitting the remaining runs

All four at once (parallel chains of 4):

```bash
../WeatherGenerator-private/hpc/launch-slurm.py --pipeline config/cerra_variants_pipeline.yml
```

One at a time:

```bash
../WeatherGenerator-private/hpc/launch-slurm.py --stage train --chain-jobs 4 \
    --base-config config/forecasting_cerra_3h.yml \
    --options streams_directory=./config/streams/<dir>/
```

Artifacts per run: source/config copy in `/e/scratch/weatherai/slurm/slurm_weathergen_<run_id>_dir/`,
logs in `.../WeatherGenerator/logs/<run_id>/`, results in `/e/scratch/weatherai/shared_work`.

## Prerequisite before resubmitting

Read access to the CERRA zarr requires membership in group `slmet` (`/e/data1/slmet` is `drwxrws--- root slmet`).
`radev1` is currently in `jusers jupiter_booster weatherai e-ext-2025e01-128` only. Either get added to `slmet`
or point `data_path_anemoi` (in `WeatherGenerator-private/hpc/jupiter/config/paths.yml`) at a copy readable by `weatherai`.

Note: the Slurm script reports `COMPLETED 0:0` for the crashed part1 because the `train` branch of
`hpc/jupiter/weathergen_slurm.sh` does not propagate `srun`'s exit code. Check `logs/<run_id>/output.<jobid>.txt`, not `sacct`.
