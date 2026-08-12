#!/bin/bash
#SBATCH --job-name=gbrew-prep
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --output=results/slurm_logs/%x-%j.out
# ---------------------------------------------------------------------------
# Stage 01 — data preparation (download + .mtx -> .sg conversion).
# REQUIRES INTERNET. Run on a login node or on a partition with egress.
# Override via env: EXP, GRAPHS, EXTRA_ARGS
#   sbatch --export=ALL,EXP=2,GRAPHS="cit-Patents com-Orkut" 01_prep.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p results/slurm_logs
module load gcc miniforge || true
source .venv/bin/activate 2>/dev/null || true

EXP="${EXP:-2}"
GRAPHS="${GRAPHS:-}"
EXTRA="${EXTRA_ARGS:-}"
GRAPH_DIR="${GRAPH_DIR:-/media/Data/00_GraphDatasets/GraphBrew}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/media/Data/00_GraphDatasets/GraphBrew/artifacts}"
THREADS="${THREADS:-16}"
CPU_LIST="${CPU_LIST:-0-15}"

GRAPH_FLAG=""
if [[ -n "$GRAPHS" ]]; then GRAPH_FLAG="--graphs $GRAPHS"; fi

python3 scripts/experiments/vldb/stages/01_prep.py \
  --exp "$EXP" \
  --graph-dir "$GRAPH_DIR" \
  --artifact-root "$ARTIFACT_ROOT" \
  --threads "$THREADS" \
  --cpu-list "$CPU_LIST" \
  $GRAPH_FLAG $EXTRA
