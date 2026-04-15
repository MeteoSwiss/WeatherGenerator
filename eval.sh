#!/bin/bash
#SBATCH --job-name=wg-eval
#SBATCH --partition=normal
#SBATCH --account=ch17
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail
cd /users/tsivalingam/projects/WeatherGenerator
uv run evaluate --config config/evaluate/eval_config.yml 