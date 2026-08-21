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

EXTRA="${EXTRA_ARGS:-}"
GRAPH_DIR="${GRAPH_DIR:-/media/NVMeData/00_GraphDatasets/GraphBrew}"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-/media/NVMeData/00_GraphDatasets/GraphBrew/artifacts}"
PRIVATE_ARGS=()
if [[ -n "${PAPER_DIR:-}" ]]; then
  PRIVATE_ARGS+=(--paper-dir "$PAPER_DIR")
  if [[ "${PUBLISH_PAPER_FIGURES:-0}" == "1" ]]; then
    PRIVATE_ARGS+=(--publish-paper-figures)
  fi
  if [[ "${PACKAGE_PAPER:-0}" == "1" ]]; then
    PRIVATE_ARGS+=(--package-paper)
    if [[ -n "${PAPER_PACKAGE_MANIFEST:-}" ]]; then
      PRIVATE_ARGS+=(--paper-package-manifest "$PAPER_PACKAGE_MANIFEST")
    fi
  fi
fi
python3 scripts/experiments/vldb/stages/05_aggregate.py \
  --exp 0 \
  --graph-dir "$GRAPH_DIR" \
  --artifact-root "$ARTIFACT_ROOT" \
  "${PRIVATE_ARGS[@]}" \
  $EXTRA
