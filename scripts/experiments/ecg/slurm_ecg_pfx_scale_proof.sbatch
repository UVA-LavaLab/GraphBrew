#!/bin/bash
#SBATCH --job-name=ecg-pfx-scale
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=results/slurm_logs/ecg_pfx_scale_%A_%a.out
#SBATCH --error=results/slurm_logs/ecg_pfx_scale_%A_%a.err

set -euo pipefail

cd "${GRAPHBREW_ROOT:-$PWD}"
source .venv/bin/activate
mkdir -p results/slurm_logs

if [[ -n "${SHARDS:-}" ]]; then
  line_number=$((SLURM_ARRAY_TASK_ID + 1))
  line=$(sed -n "${line_number}p" "$SHARDS")
  if [[ -z "$line" ]]; then
    echo "No scale-proof row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 3
  fi
  IFS=$'\t' read -r SCALE ROOT BACKEND OUT_ROOT <<< "$line"
else
  : "${SCALE:?SCALE must be set when SHARDS is not provided}"
  : "${ROOT:?ROOT must be set when SHARDS is not provided}"
  BACKEND="${BACKEND:-both}"
  OUT_ROOT="${OUT_ROOT:-results/ecg_experiments/ecg_pfx_scale_proof/slurm_${SCALE}_${ROOT}_${BACKEND}}"
fi

python3 scripts/experiments/ecg/ecg_pfx_scale_proof.py \
  --scale "$SCALE" \
  --roots "$ROOT" \
  --backend "$BACKEND" \
  --out-root "$OUT_ROOT" \
  --timeout-sniper "${TIMEOUT_SNIPER:-1200}" \
  --timeout-gem5 "${TIMEOUT_GEM5:-1800}"