#!/bin/bash
#SBATCH --job-name=gbrew-cachesim
#SBATCH --partition=standard
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=results/slurm_logs/%x-%j.out
# ---------------------------------------------------------------------------
# Stage 04 — cache simulation (CPU-speed independent).
# Slow per-graph. Wall-clock of the host does NOT affect results — only the
# simulated cache stats are recorded. Safe to run on slow / shared nodes
# or a different machine than stage 03.
# Override via env: EXP (default 1), GRAPHS, EXTRA_ARGS
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p results/slurm_logs
module load gcc miniforge || true
source .venv/bin/activate 2>/dev/null || true

EXP="${EXP:-1}"
GRAPHS="${GRAPHS:-}"
EXTRA="${EXTRA_ARGS:-}"

GRAPH_FLAG=""
if [[ -n "$GRAPHS" ]]; then GRAPH_FLAG="--graphs $GRAPHS"; fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
python3 scripts/experiments/vldb/stages/04_cache_sim.py --exp "$EXP" $GRAPH_FLAG $EXTRA
