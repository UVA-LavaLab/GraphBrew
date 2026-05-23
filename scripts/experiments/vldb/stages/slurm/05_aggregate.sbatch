#!/bin/bash
#SBATCH --job-name=gbrew-aggregate
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=results/slurm_logs/%x-%j.out
# ---------------------------------------------------------------------------
# Stage 05 — generate paper figures + LaTeX tables from collected JSON.
# Pure I/O. Run after 03 (and optionally 04) finish.
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p results/slurm_logs
module load gcc miniforge || true
source .venv/bin/activate 2>/dev/null || true

python3 scripts/experiments/vldb/stages/05_aggregate.py --exp 0
