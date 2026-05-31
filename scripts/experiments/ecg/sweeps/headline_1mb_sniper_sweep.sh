#!/usr/bin/env bash
# headline_1mb_sniper_sweep.sh — populate Sniper ECG cells at the
# literature 1MB canonical L3 row, for graphs the workstation can run.
#
# Companion to headline_1mb_{ecg,gem5}_sweep.sh. Uses SIFT file-backed
# wrapper paths (per ECG-Sniper-Runs.md). Output dir mirrors the
# existing Sniper anchor layout.
#
# Cache hierarchy: GRASP HPCA20 canonical
#   L1d 32kB/8w, L2 256kB/8w, L3 1MB/16w, 64B lines
# Reorder: -o 5 (DBG)
# Policies: literature roster + ECG variant
#   LRU, SRRIP, GRASP, POPT, ECG:DBG_PRIMARY
#
# Workstation tiers (per ECG-Sniper-Runs.md sizing):
#   email-Eu-core: SIFT file-backed smoke — ~5-15 min/cell
#   cit-Patents:   SIFT same-graph — ~30 min/cell or higher (~50 GiB RSS
#                  documented for full same-graph runs; use with caution)
#
# Output: $OUT_ROOT/<graph>-<app>/DBG/roi_matrix.csv
# Default $OUT_ROOT=/tmp/graphbrew-headline-1mb-sniper
#
# Usage:
#   bash scripts/experiments/ecg/sweeps/headline_1mb_sniper_sweep.sh
#   GRAPH_FILTER=email-Eu-core bash scripts/.../headline_1mb_sniper_sweep.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="${OUT_ROOT:-/tmp/graphbrew-headline-1mb-sniper}"
LOG="${OUT_ROOT}/runner.log"
mkdir -p "$OUT_ROOT"
date +"%Y-%m-%d %T --- start headline-1MB Sniper sweep ---" | tee -a "$LOG"

declare -A GRAPH_PATHS=(
  [email-Eu-core]=results/graphs/email-Eu-core/email-Eu-core.sg
  [cit-Patents]=results/graphs/cit-Patents/cit-Patents.sg
)

declare -A GRAPH_APPS=(
  [email-Eu-core]="pr bfs sssp"
  [cit-Patents]="pr bfs sssp"
)

# Sniper full-wrapper bc smoke is not yet validated per ECG-Sniper-Runs.md
# so we keep bc out of the Sniper headline scope for now. cache_sim and
# gem5 cover bc.

declare -A APP_OPTS=(
  [pr]="-f {graph_path} -s -o 5 -n 1 -i 2"
  [bfs]="-f {graph_path} -s -o 5 -n 1 -r 0"
  [sssp]="-f {graph_path} -s -o 5 -n 1 -r 0"
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
      date +"%T SKIP  ${short}/${app} (Sniper CSV has >= ${#POLICIES[@]} policy_labels)" | tee -a "$LOG"
      skip_count=$((skip_count + 1))
      continue
    fi
    mkdir -p "$outdir"
    opts="${APP_OPTS[$app]//\{graph_path\}/$path}"
    date +"%T BEGIN ${short}/${app} Sniper L3=1MB policies=${POLICIES[*]}" | tee -a "$LOG"
    if python3 scripts/experiments/ecg/roi_matrix.py \
        --suite sniper \
        --benchmark "$app" \
        --options "$opts" \
        --policies "${POLICIES[@]}" \
        --l1d-size 32kB --l1d-ways 8 \
        --l2-size 256kB --l2-ways 8 \
        --l3-sizes 1MB --l3-ways 16 \
        --line-size 64 \
        --timeout-sniper 14400 \
        --sniper-workload benchmark \
        --allow-sniper-benchmark-workload \
        --sniper-frontend sift \
        --sniper-enable-graph-policies \
        --out-dir "$outdir" \
        --no-build >> "$LOG" 2>&1; then
      date +"%T OK    ${short}/${app}" | tee -a "$LOG"
      cell_count=$((cell_count + 1))
    else
      date +"%T FAIL  ${short}/${app}" | tee -a "$LOG"
    fi
  done
done

date +"%Y-%m-%d %T --- finished headline-1MB Sniper sweep: ${cell_count} new cells, ${skip_count} skipped ---" | tee -a "$LOG"
