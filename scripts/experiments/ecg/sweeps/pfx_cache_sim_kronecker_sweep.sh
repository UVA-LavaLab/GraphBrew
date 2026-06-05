#!/usr/bin/env bash
# pfx_cache_sim_kronecker_sweep.sh — synthetic-graph extrapolation proof.
#
# Sprint 6f Path C: validate that the ECG_PFX vs DROPLET efficiency claim
# from the literature corpus (sprint 6e: 3.27× fewer requests for same
# L3 miss reduction) holds on synthetic Kronecker graphs at 4M and 16M
# vertices — well beyond the largest literature graph (soc-LiveJournal1
# at 4.8M).
#
# Output: /tmp/graphbrew-ecg-pfx-cache_sim-kronecker/<graph>-<app>/{baselines,pfx_combined,droplet_combined}/

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="${OUT_ROOT:-/tmp/graphbrew-ecg-pfx-cache_sim-kronecker}"
LOG="${OUT_ROOT}/runner_pfx_kron_sweep.log"
mkdir -p "$OUT_ROOT"
date +"%Y-%m-%d %T --- start kron sweep ---" | tee -a "$LOG"

declare -A GRAPH_PATHS=(
  [kron-s22]=results/graphs/kron-s22/kron-s22.sg
  [kron-s24]=results/graphs/kron-s24/kron-s24.sg
)
declare -A GRAPH_APPS=(
  [kron-s22]="pr bfs"
  [kron-s24]="pr"
)
declare -A BFS_SRC=( [kron-s22]=0 [kron-s24]=0 )
declare -A APP_OPTS=(
  [pr]="-f {graph_path} -s -o 5 -n 1 -i 2"
  [bfs]="-f {graph_path} -s -o 5 -n 1 -r {bfs_src}"
)

cell_complete() {
  local csv="$1"
  [ -f "$csv" ] || return 1
  local status
  status="$(awk -F, '
    NR==1 { for (i=1;i<=NF;i++) if($i=="status") si=i; next }
    si { print $si; exit }
  ' "$csv" 2>/dev/null || echo '')"
  case "$status" in
    ok|active_no_fill|inactive) return 0 ;;
    *) return 1 ;;
  esac
}

run_ct=0; skip_ct=0; fail_ct=0
for short in "${!GRAPH_PATHS[@]}"; do
  if [ -n "${GRAPH_FILTER:-}" ] && [ "$short" != "${GRAPH_FILTER}" ]; then continue; fi
  path="${GRAPH_PATHS[$short]}"
  if [ ! -f "$path" ]; then
    date +"%T MISS  ${short} ($path)" | tee -a "$LOG"
    continue
  fi
  for app in ${GRAPH_APPS[$short]}; do
    if [ -n "${APP_FILTER:-}" ] && [ "$app" != "${APP_FILTER}" ]; then continue; fi
    bfs_src="${BFS_SRC[$short]:-0}"
    opts="${APP_OPTS[$app]//\{graph_path\}/$path}"
    opts="${opts//\{bfs_src\}/$bfs_src}"

    # Pass 1: baselines (LRU, SRRIP, GRASP, POPT, ECG_DBG, ECG_PRIMARY)
    out_b="${OUT_ROOT}/${short}-${app}/baselines"
    csv_b="${out_b}/roi_matrix.csv"
    if cell_complete "$csv_b"; then
      date +"%T SKIP  ${short}/${app} baselines" | tee -a "$LOG"
      skip_ct=$((skip_ct+1))
    else
      mkdir -p "$out_b"
      date +"%T BEGIN ${short}/${app} BASELINES" | tee -a "$LOG"
      if ECG_CONTAINER_BITS=64 timeout 7200 python3 scripts/experiments/ecg/roi_matrix.py \
          --suite cache-sim --no-build \
          --benchmark "$app" --options "$opts" \
          --policies LRU SRRIP GRASP POPT ECG_DBG_PRIMARY ECG_DBG_ONLY \
          --l1d-size 32kB --l1d-ways 8 \
          --l2-size 256kB --l2-ways 8 \
          --l3-sizes 1MB --l3-ways 16 --line-size 64 \
          --timeout-cache 3600 --out-dir "$out_b" >> "$LOG" 2>&1; then
        date +"%T OK    ${short}/${app} BASELINES" | tee -a "$LOG"
        run_ct=$((run_ct+1))
      else
        date +"%T FAIL  ${short}/${app} BASELINES" | tee -a "$LOG"
        fail_ct=$((fail_ct+1))
      fi
    fi

    # Pass 2: ECG_PFX combined (ECG_DBG eviction + ECG_PFX prefetch)
    out_pfx="${OUT_ROOT}/${short}-${app}/pfx_combined"
    csv_pfx="${out_pfx}/roi_matrix.csv"
    if cell_complete "$csv_pfx"; then
      date +"%T SKIP  ${short}/${app} pfx_combined" | tee -a "$LOG"
      skip_ct=$((skip_ct+1))
    else
      mkdir -p "$out_pfx"
      date +"%T BEGIN ${short}/${app} ECG_PFX combined" | tee -a "$LOG"
      if ECG_CONTAINER_BITS=64 ECG_PREFETCH_LOOKAHEAD=8 ECG_PREFETCH_MODE=2 \
          timeout 7200 python3 scripts/experiments/ecg/roi_matrix.py \
          --suite cache-sim --no-build \
          --benchmark "$app" --options "$opts" \
          --policies ECG_DBG \
          --prefetcher ECG_PFX --prefetcher-level l2 \
          --l1d-size 32kB --l1d-ways 8 \
          --l2-size 256kB --l2-ways 8 \
          --l3-sizes 1MB --l3-ways 16 --line-size 64 \
          --timeout-cache 3600 --out-dir "$out_pfx" >> "$LOG" 2>&1; then
        date +"%T OK    ${short}/${app} ECG_PFX combined" | tee -a "$LOG"
        run_ct=$((run_ct+1))
      else
        date +"%T FAIL  ${short}/${app} ECG_PFX combined" | tee -a "$LOG"
        fail_ct=$((fail_ct+1))
      fi
    fi

    # Pass 3: DROPLET combined (ECG_DBG eviction + sequential lookahead = DROPLET equivalent)
    out_drp="${OUT_ROOT}/${short}-${app}/droplet_combined"
    csv_drp="${out_drp}/roi_matrix.csv"
    if cell_complete "$csv_drp"; then
      date +"%T SKIP  ${short}/${app} droplet_combined" | tee -a "$LOG"
      skip_ct=$((skip_ct+1))
    else
      mkdir -p "$out_drp"
      date +"%T BEGIN ${short}/${app} DROPLET combined" | tee -a "$LOG"
      if ECG_CONTAINER_BITS=64 ECG_PREFETCH_LOOKAHEAD=8 ECG_PREFETCH_MODE=3 \
          timeout 7200 python3 scripts/experiments/ecg/roi_matrix.py \
          --suite cache-sim --no-build \
          --benchmark "$app" --options "$opts" \
          --policies ECG_DBG \
          --prefetcher DROPLET --prefetcher-level l2 \
          --l1d-size 32kB --l1d-ways 8 \
          --l2-size 256kB --l2-ways 8 \
          --l3-sizes 1MB --l3-ways 16 --line-size 64 \
          --timeout-cache 3600 --out-dir "$out_drp" >> "$LOG" 2>&1; then
        date +"%T OK    ${short}/${app} DROPLET combined" | tee -a "$LOG"
        run_ct=$((run_ct+1))
      else
        date +"%T FAIL  ${short}/${app} DROPLET combined" | tee -a "$LOG"
        fail_ct=$((fail_ct+1))
      fi
    fi
  done
done

date +"%Y-%m-%d %T --- finished kron sweep: ran=${run_ct} skip=${skip_ct} fail=${fail_ct} ---" | tee -a "$LOG"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$LOG"
