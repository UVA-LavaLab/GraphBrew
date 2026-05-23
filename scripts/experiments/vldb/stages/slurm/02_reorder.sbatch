#!/bin/bash
#SBATCH --job-name=gbrew-reorder
#SBATCH --partition=standard
#SBATCH --time=06:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=results/slurm_logs/%x-%j.out
# ---------------------------------------------------------------------------
# Stage 02 — pre-generate reorder mappings (one-time, reused by 03 + 04).
# CPU-bound. NO network. Safe on compute nodes (--skip-download equivalent).
# Override via env: EXP, GRAPHS, EXTRA_ARGS
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p results/slurm_logs
module load gcc miniforge || true
source .venv/bin/activate 2>/dev/null || true

EXP="${EXP:-2}"
GRAPHS="${GRAPHS:-}"
EXTRA="${EXTRA_ARGS:-}"

GRAPH_FLAG=""
if [[ -n "$GRAPHS" ]]; then GRAPH_FLAG="--graphs $GRAPHS"; fi

# Pre-flight: graph .sg must exist (stage 01 must have run first).
if [[ -n "$GRAPHS" ]]; then
  for g in $GRAPHS; do
    if [[ ! -f "results/graphs/$g/$g.sg" ]]; then
      echo "ERROR: results/graphs/$g/$g.sg missing — run stage 01 first." >&2
      exit 2
    fi
  done
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
python3 scripts/experiments/vldb/stages/02_reorder.py --exp "$EXP" $GRAPH_FLAG $EXTRA
