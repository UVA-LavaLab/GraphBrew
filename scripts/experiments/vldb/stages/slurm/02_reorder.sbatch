#!/bin/bash
#SBATCH --job-name=gbrew-reorder
#SBATCH --partition=standard
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=240G
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
GRAPH_DIR="${GRAPH_DIR:-/media/NVMeData/00_GraphDatasets/GraphBrew}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/media/NVMeData/00_GraphDatasets/GraphBrew/artifacts}"
THREADS="${THREADS:-16}"
CPU_LIST="${CPU_LIST:-0-15}"

GRAPH_FLAG=""
if [[ -n "$GRAPHS" ]]; then GRAPH_FLAG="--graphs $GRAPHS"; fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
python3 scripts/experiments/vldb/stages/02_reorder.py \
  --exp "$EXP" \
  --graph-dir "$GRAPH_DIR" \
  --artifact-root "$ARTIFACT_ROOT" \
  --threads "$THREADS" \
  --cpu-list "$CPU_LIST" \
  $GRAPH_FLAG $EXTRA
