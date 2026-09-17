# CERRA output-variant training runs

Launched 2026-09-17 16:50 on santis from `~/WeatherGenerator`. Pipeline ID `n7mrncxn`.

## Launch command

```bash
env -u UENV_MOUNT_LIST -u UENV_VIEW -u UENV_LABEL -u UENV_REPO -u UENV_TELEMETRY -u UENV_WARN_MIGRATE \
  ../WeatherGenerator-private/hpc/launch-slurm.py --pipeline config/pipeline_cerra_streams.yml --no-register
```

- `env -u UENV_*`: the login shell had `prgenv-gnu/25.6:v2` loaded; santis rejects `sbatch` from inside a uenv
  session, and `weathergen_slurm.sh` starts its own `uenv run` in the job anyway.
- `--no-register`: no MLflow registration/upload.
- Pipeline file: `config/pipeline_cerra_streams.yml` (`parallelize: true`, `chain_jobs: 4` per stage).
- Slurm script: `hpc/santis/weathergen_slurm.sh` (1 node, 4 GPUs, `--time=24:00:00`, partition `normal`, account `ch17`).
- Copies: `/iopsstor/scratch/cscs/thunter/slurm/slurm_weathergen_<run_id>_dir/`
- Results/logs/models: `/iopsstor/scratch/cscs/thunter/shared_work/`

## Common setup

All 4 experiments share `config/forecasting_cerra_3h.yml` and differ only in `streams_directory`.

| | |
|---|---|
| Dataset | `cerra-rr-an-oper-se-al-ec-mars-5p5km-1985-2023-3h-v2` (anemoi), `trim_edge: 100` |
| Input | same 72 channels: `skt sp 10si 10wdir 2r 2t msl` + `r/t/u/v/z` at 13 levels (50–1000 hPa) |
| Task | forecasting, `masking_strategy: forecast`, `time_window 03:00:00`, `forecast.num_steps: 1`, `offset: 1` |
| Train period | 1985-01-01 – 2022-12-31; validation 2023-10-01 – 2023-12-31 |
| Schedule | 128 mini-epochs × 4096 samples, lr_max 5e-5, warmup 256, cooldown 512 |
| Model | healpix L5, `ae_local/global dim 2048`, `ae_global 4 blocks`, `fe 16 blocks`, decoder `PerceiverIOCoordConditioning` |
| Stream tokens | `token_size 32`, `max_num_targets 375000`, embed transformer 2 blocks / dim 256 |
| Jobs | 4 chained jobs per run (`part1` fresh `train`, `part2-4` `train_continue` from latest checkpoint, `afterany`) + 1 cleanup job |

## Experiments

### 1. `cerrar850` — single decoder, r_850

| | |
|---|---|
| run_name | `cerrar850` |
| run_id | `fx6yh9m6` |
| streams_directory | `./config/streams/cerra_r850/` |
| streams | `CERRA` (stream_id 0): source = 72 ch, target = `['r_850']` |
| slurm jobs | `870129` (part1), `870130` (part2), `870131` (part3), `870132` (part4), `870133` (cleanup) |
| job names | `weathergen_fx6yh9m6_part{1..4}_cerrar850_n7mrncxn` |

### 2. `cerraz500` — single decoder, z_500

| | |
|---|---|
| run_name | `cerraz500` |
| run_id | `vumx8vah` |
| streams_directory | `./config/streams/cerra_z500/` |
| streams | `CERRA` (stream_id 0): source = 72 ch, target = `['z_500']` |
| slurm jobs | `870134` (part1), `870135` (part2), `870136` (part3), `870137` (part4), `870138` (cleanup) |
| job names | `weathergen_vumx8vah_part{1..4}_cerraz500_n7mrncxn` |

### 3. `cerrar850z500` — single decoder, r_850 + z_500

| | |
|---|---|
| run_name | `cerrar850z500` |
| run_id | `mmfsjhph` |
| streams_directory | `./config/streams/cerra_r850_z500/` |
| streams | `CERRA` (stream_id 0): source = 72 ch, target = `['r_850', 'z_500']` |
| slurm jobs | `870139` (part1), `870140` (part2), `870141` (part3), `870142` (part4), `870143` (cleanup) |
| job names | `weathergen_mmfsjhph_part{1..4}_cerrar850z500_n7mrncxn` |

### 4. `cerra2dec` — shared encoder, two independent decoders (r_850, z_500)

| | |
|---|---|
| run_name | `cerra2dec` |
| run_id | `y9sqjerj` |
| streams_directory | `./config/streams/cerra_2dec_r850_z500/` |
| streams | `CERRA_in` (stream_id 0, `forcing: True`): source = 72 ch, target = `[]`; `CERRA_r850` (stream_id 1, `diagnostic: True`): target = `['r_850']`; `CERRA_z500` (stream_id 2, `diagnostic: True`): target = `['z_500']` |
| slurm jobs | `870144` (part1), `870145` (part2), `870146` (part3), `870147` (part4), `870148` (cleanup) |
| job names | `weathergen_y9sqjerj_part{1..4}_cerra2dec_n7mrncxn` |

## Monitoring

```bash
squeue -u $USER -o "%i %j %T %r"
tail -f /iopsstor/scratch/cscs/thunter/shared_work/logs/<run_id>/output.<jobid>.txt
```
