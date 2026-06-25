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
  if [[ ! -r "$SHARDS" ]]; then
    echo "SHARDS is not readable: $SHARDS" >&2
    exit 2
  fi
  if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID must be set by sbatch --array when SHARDS is provided" >&2
    exit 2
  fi
  line_number=$((SLURM_ARRAY_TASK_ID + 1))
  line=$(sed -n "${line_number}p" "$SHARDS")
  if [[ -z "$line" ]]; then
    echo "No scale-proof row for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}" >&2
    exit 3
  fi
  field_count=$(awk -F '\t' '{print NF; exit}' <<< "$line")
  if [[ "$field_count" -ne 4 ]]; then
    echo "Invalid scale-proof row ${line_number}: expected 4 tab-separated fields, got ${field_count}" >&2
    echo "$line" >&2
    exit 4
  fi
  IFS=$'\t' read -r SCALE ROOT BACKEND OUT_ROOT <<< "$line"
  for field_name in SCALE ROOT BACKEND OUT_ROOT; do
    if [[ -z "${!field_name}" ]]; then
      echo "Invalid scale-proof row ${line_number}: empty ${field_name}" >&2
      echo "$line" >&2
      exit 4
    fi
  done
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