#!/bin/bash
#SBATCH --job-name=gbrew-cpuperf
#SBATCH --partition=standard
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --nodelist=jaguar
#SBATCH --mem=128G
#SBATCH --output=results/slurm_logs/%x-%j.out
# ---------------------------------------------------------------------------
# Stage 03 — CPU wall-clock benchmark sweep.
# Real-hardware timing. Use a fast, isolated partition for repeatable numbers.
# Requires stage 01 (.sg files) and stage 02 (.lo mappings) completed.
# Override via env: EXP, GRAPHS, MEASUREMENT_GENERATION, EXTRA_ARGS
#   sbatch --export=ALL,EXP=2,GRAPHS="com-Orkut" 03_cpu_perf.sbatch
# Final paper timing must remain on jaguar; do not fan out across node classes.
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p results/slurm_logs
module load gcc miniforge || true
source .venv/bin/activate 2>/dev/null || true

EXP="${EXP:-2}"
GRAPHS="${GRAPHS:-}"
EXTRA="${EXTRA_ARGS:-}"
GENERATION="${MEASUREMENT_GENERATION:-}"
GRAPH_DIR="${GRAPH_DIR:-/media/NVMeData/00_GraphDatasets/GraphBrew}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/media/NVMeData/00_GraphDatasets/GraphBrew/artifacts}"
THREADS="${THREADS:-16}"
CPU_LIST="${CPU_LIST:-0-15}"
EXPECTED_HOST="${EXPECTED_HOST:-jaguar}"

if [[ "$(hostname)" != "$EXPECTED_HOST" ]]; then
  echo "ERROR: final CPU timing requires host $EXPECTED_HOST" >&2
  exit 2
fi

GRAPH_FLAG=""
if [[ -n "$GRAPHS" ]]; then GRAPH_FLAG="--graphs $GRAPHS"; fi
GENERATION_FLAG=""
if [[ -n "$GENERATION" ]]; then
  GENERATION_FLAG="--measurement-generation $GENERATION"
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
python3 scripts/experiments/vldb/stages/03_cpu_perf.py \
  --exp "$EXP" \
  --graph-dir "$GRAPH_DIR" \
  --artifact-root "$ARTIFACT_ROOT" \
  --threads "$THREADS" \
  --cpu-list "$CPU_LIST" \
  $GRAPH_FLAG $GENERATION_FLAG $EXTRA
