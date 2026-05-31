#!/usr/bin/env bash
# headline_1mb_gem5_sweep.sh — populate gem5 ECG cells at the literature
# 1MB canonical L3 row, for graphs the workstation can run.
#
# Mirrors scripts/experiments/ecg/sweeps/headline_1mb_ecg_sweep.sh but
# drives gem5 instead of cache_sim. Outputs to a NEW sweep root so the
# existing /tmp/graphbrew-grasp-gem5-sweep (4kB/32kB/256kB/2MB stress
# config) is preserved untouched.
#
# Cache hierarchy: GRASP HPCA20 canonical
#   L1d 32kB/8w, L2 256kB/8w, L3 1MB/16w, 64B lines
# Reorder: -o 5 (DBG)
# Policies: literature roster + ECG variant
#   LRU, SRRIP, GRASP, POPT, ECG:DBG_PRIMARY
#
# Workstation tiers (ECG-Final-Runs.md / ECG-Sniper-Runs.md sizing):
#   email-Eu-core (tiny, ~5-10 min/cell):  LOCAL    — launch by default
#   cit-Patents   (3.8M v, 16M e, ~30 min/cell):
#     LOCAL but long — use GRAPH_FILTER=cit-Patents to gate
#
# Total runtime estimates per graph (5 pol x 4 apps x 1 L3):
#   email-Eu-core: ~40 min
#   cit-Patents:   ~10 hours
#
# Output: $OUT_ROOT/<graph>-<app>/DBG/roi_matrix.csv
# Default $OUT_ROOT=/tmp/graphbrew-headline-1mb-gem5
#
# Usage:
#   bash scripts/experiments/ecg/sweeps/headline_1mb_gem5_sweep.sh
#   GRAPH_FILTER=email-Eu-core bash scripts/.../headline_1mb_gem5_sweep.sh
#   GRAPH_FILTER=cit-Patents APP_FILTER=pr bash scripts/.../headline_1mb_gem5_sweep.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="${OUT_ROOT:-/tmp/graphbrew-headline-1mb-gem5}"
LOG="${OUT_ROOT}/runner.log"
mkdir -p "$OUT_ROOT"
date +"%Y-%m-%d %T --- start headline-1MB gem5 sweep ---" | tee -a "$LOG"

declare -A GRAPH_PATHS=(
  [email-Eu-core]=results/graphs/email-Eu-core/email-Eu-core.sg
  [cit-Patents]=results/graphs/cit-Patents/cit-Patents.sg
)

declare -A GRAPH_APPS=(
  [email-Eu-core]="pr bfs sssp bc"
  [cit-Patents]="pr bfs sssp bc"
)

declare -A APP_OPTS=(
  [pr]="-f {graph_path} -s -o 5 -n 1 -i 2"
  [bfs]="-f {graph_path} -s -o 5 -n 1 -r 0"
  [sssp]="-f {graph_path} -s -o 5 -n 1 -r 0"
  [bc]="-f {graph_path} -s -o 5 -n 1 -r 0 -i 1"
)

POLICIES=(LRU SRRIP GRASP POPT "ECG:DBG_PRIMARY")
TARGET_LABEL="ECG_DBG_PRIMARY"

cell_is_complete() {
  local csv="$1" expected="$2"
  [ -f "$csv" ] || return 1
  local actual
  actual="$(awk -F, '
    NR==1 { for (i=1;i<=NF;i++) if($i=="policy_label") pli=i; next }
    pli { print $pli }
  ' "$csv" | sort -u | wc -l)"
  [ "$actual" -ge "$expected" ]
}

cell_count=0
skip_count=0
for short in "${!GRAPH_PATHS[@]}"; do
  if [ -n "${GRAPH_FILTER:-}" ] && [ "$short" != "${GRAPH_FILTER}" ]; then
    continue
  fi
  path="${GRAPH_PATHS[$short]}"
  if [ ! -f "$path" ]; then
    date +"%T MISS  ${short} (sg not on disk: $path)" | tee -a "$LOG"
    continue
  fi
  for app in ${GRAPH_APPS[$short]}; do
    if [ -n "${APP_FILTER:-}" ] && [ "$app" != "${APP_FILTER}" ]; then
      continue
    fi
    outdir="${OUT_ROOT}/${short}-${app}/DBG"
    csv="${outdir}/roi_matrix.csv"
    if cell_is_complete "$csv" "${#POLICIES[@]}"; then
      date +"%T SKIP  ${short}/${app} (gem5 CSV has >= ${#POLICIES[@]} policy_labels)" | tee -a "$LOG"
      skip_count=$((skip_count + 1))
      continue
    fi
    mkdir -p "$outdir"
    opts="${APP_OPTS[$app]//\{graph_path\}/$path}"
    date +"%T BEGIN ${short}/${app} gem5 L3=1MB policies=${POLICIES[*]}" | tee -a "$LOG"
    if python3 scripts/experiments/ecg/roi_matrix.py \
        --suite gem5 \
        --benchmark "$app" \
        --options "$opts" \
        --policies "${POLICIES[@]}" \
        --l1d-size 32kB --l1d-ways 8 \
        --l2-size 256kB --l2-ways 8 \
        --l3-sizes 1MB --l3-ways 16 \
        --line-size 64 \
        --timeout-gem5 14400 \
        --out-dir "$outdir" \
        --no-build >> "$LOG" 2>&1; then
      date +"%T OK    ${short}/${app}" | tee -a "$LOG"
      cell_count=$((cell_count + 1))
    else
      date +"%T FAIL  ${short}/${app}" | tee -a "$LOG"
    fi
  done
done

date +"%Y-%m-%d %T --- finished headline-1MB gem5 sweep: ${cell_count} new cells, ${skip_count} skipped ---" | tee -a "$LOG"
