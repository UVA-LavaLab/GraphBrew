#!/usr/bin/env bash
# pfx_cache_sim_scale_sweep.sh — populate ECG_DBG + ECG_PFX at scale
# across the literature graph corpus at L3=1MB.
#
# Companion to sprint 6c findings (docs/findings/ecg_pfx_recovery_2026-06-01.md):
# ECG_PFX delivers 10-15 pp L3 miss reduction when paired with ECG eviction
# and ECG_CONTAINER_BITS=64. This sweep emits a CSV per (graph, app) that
# carries:
#   - LRU baseline (control)
#   - GRASP baseline (literature comparator)
#   - POPT baseline (oracle comparator)
#   - ECG:DBG_ONLY (ECG eviction, no prefetch)
#   - ECG:DBG_ONLY + ECG_PFX lookahead=8 (combined claim)
#
# Each policy row is written to the canonical CSV; downstream gates
# compute deltas (ECG_combined vs LRU, vs GRASP, vs POPT).
#
# Output: /tmp/graphbrew-ecg-pfx-cache_sim-scale/<graph>-<app>/lit/roi_matrix.csv
#
# Time: cache_sim is fast (~30s-3min/policy on large graphs), so full sweep
# is ~30-60 min wall, not the multi-hour Sniper budget.
#
# Usage:
#   bash scripts/experiments/ecg/sweeps/pfx_cache_sim_scale_sweep.sh
# Or filter:
#   GRAPH_FILTER=cit-Patents APP_FILTER=pr bash scripts/.../pfx_cache_sim_scale_sweep.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="${OUT_ROOT:-/tmp/graphbrew-ecg-pfx-cache_sim-scale}"
LOG="${OUT_ROOT}/runner_pfx_cache_sim_scale.log"
mkdir -p "$OUT_ROOT"
date +"%Y-%m-%d %T --- start ECG_PFX cache_sim scale sweep ---" | tee -a "$LOG"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$LOG"
echo "ECG_CONTAINER_BITS=64 ECG_PREFETCH_LOOKAHEAD=8 ECG_PREFETCH_MODE=2 (popt)" | tee -a "$LOG"

declare -A GRAPH_PATHS=(
  [email-Eu-core]=results/graphs/email-Eu-core/email-Eu-core.sg
  [web-Google]=results/graphs/web-Google/web-Google.sg
  [soc-pokec]=results/graphs/soc-pokec/soc-pokec.sg
  [cit-Patents]=results/graphs/cit-Patents/cit-Patents.sg
  [soc-LiveJournal1]=results/graphs/soc-LiveJournal1/soc-LiveJournal1.sg
  [com-orkut]=results/graphs/com-orkut/com-orkut.sg
)

declare -A GRAPH_APPS=(
  [email-Eu-core]="pr"
  [web-Google]="pr bfs bc"
  [soc-pokec]="pr sssp bc"
  [cit-Patents]="pr bfs sssp bc"
  [soc-LiveJournal1]="pr bfs sssp bc"
  [com-orkut]="pr"
)

declare -A APP_OPTS=(
  [pr]="-f {graph_path} -s -o 5 -n 1 -i 2"
  [bfs]="-f {graph_path} -s -o 5 -n 1 -r {bfs_src}"
  [sssp]="-f {graph_path} -s -o 5 -n 1 -r {sssp_src}"
  [bc]="-f {graph_path} -s -o 5 -n 1 -r {bc_src} -i 1"
)

declare -A BFS_SRC=(
  [email-Eu-core]=0 [web-Google]=0 [soc-pokec]=800000
  [cit-Patents]=1500000 [soc-LiveJournal1]=2000000 [com-orkut]=1500000
)
declare -A SSSP_SRC=(
  [email-Eu-core]=0 [web-Google]=0 [soc-pokec]=800000
  [cit-Patents]=1500000 [soc-LiveJournal1]=2000000 [com-orkut]=1500000
)
declare -A BC_SRC=(
  [email-Eu-core]=0 [web-Google]=0 [soc-pokec]=800000
  [cit-Patents]=1500000 [soc-LiveJournal1]=2000000 [com-orkut]=1500000
)

# Baseline policies (no prefetch) — these feed the "vs LRU" / "vs GRASP" deltas
BASELINES=(LRU SRRIP GRASP POPT "ECG:DBG_PRIMARY" "ECG:DBG_ONLY")
# PFX-augmented arm: ECG_DBG_ONLY eviction + ECG_PFX prefetcher
# (separate sub-dir since --prefetcher is a different invocation)
PFX_POLICY="ECG:DBG_ONLY"
PFX_LABEL_SUFFIX="_with_PFX"

cell_is_complete() {
  # Skip if baseline CSV exists with all baselines (>=5 policy_labels),
  # PFX-augmented CSV exists, AND DROPLET-augmented CSV exists.
  local cell_dir="$1"
  local base_csv="${cell_dir}/baselines/roi_matrix.csv"
  local pfx_csv="${cell_dir}/pfx_combined/roi_matrix.csv"
  local drop_csv="${cell_dir}/droplet_combined/roi_matrix.csv"
  [ -f "$base_csv" ] || return 1
  [ -f "$pfx_csv" ] || return 1
  [ -f "$drop_csv" ] || return 1
  local actual
  actual="$(awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="policy_label")p=i;next} p{print $p}' "$base_csv" | sort -u | wc -l)"
  [ "$actual" -ge 5 ] || return 1
  local pfx_status drop_status
  pfx_status="$(awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="status")s=i;next} s{print $s; exit}' "$pfx_csv")"
  [ "$pfx_status" = "ok" ] || return 1
  drop_status="$(awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="status")s=i;next} s{print $s; exit}' "$drop_csv")"
  [ "$drop_status" = "ok" ] || return 1
  return 0
}

cell_count=0
skip_count=0
fail_count=0
for short in "${!GRAPH_PATHS[@]}"; do
  if [ -n "${GRAPH_FILTER:-}" ] && [ "$short" != "${GRAPH_FILTER}" ]; then
    continue
  fi
  path="${GRAPH_PATHS[$short]}"
  if [ ! -f "$path" ]; then
    date +"%T MISS  ${short} (sg missing: $path)" | tee -a "$LOG"
    continue
  fi
  for app in ${GRAPH_APPS[$short]}; do
    if [ -n "${APP_FILTER:-}" ] && [ "$app" != "${APP_FILTER}" ]; then
      continue
    fi
    cell_dir="${OUT_ROOT}/${short}-${app}"
    if cell_is_complete "$cell_dir"; then
      date +"%T SKIP  ${short}/${app} (baselines + PFX already complete)" | tee -a "$LOG"
      skip_count=$((skip_count + 1))
      continue
    fi
    mkdir -p "$cell_dir/baselines" "$cell_dir/pfx_combined" "$cell_dir/droplet_combined"
    bfs_src="${BFS_SRC[$short]:-0}"
    sssp_src="${SSSP_SRC[$short]:-0}"
    bc_src="${BC_SRC[$short]:-0}"
    opts="${APP_OPTS[$app]//\{graph_path\}/$path}"
    opts="${opts//\{bfs_src\}/$bfs_src}"
    opts="${opts//\{sssp_src\}/$sssp_src}"
    opts="${opts//\{bc_src\}/$bc_src}"

    # --- Pass 1: baselines (no prefetcher) ---
    if [ -f "$cell_dir/baselines/roi_matrix.csv" ]; then
      base_status="$(awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="status")s=i;next} s{print $s; exit}' "$cell_dir/baselines/roi_matrix.csv" 2>/dev/null || echo '')"
      base_polcount="$(awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="policy_label")p=i;next} p{print $p}' "$cell_dir/baselines/roi_matrix.csv" 2>/dev/null | sort -u | wc -l)"
    else
      base_status=''
      base_polcount=0
    fi
    if [ "$base_status" = "ok" ] && [ "$base_polcount" -ge 5 ]; then
      date +"%T SKIP  ${short}/${app} baselines (cached)" | tee -a "$LOG"
    else
    date +"%T BEGIN ${short}/${app} BASELINES" | tee -a "$LOG"
    if ECG_CONTAINER_BITS=64 timeout 3600 python3 scripts/experiments/ecg/roi_matrix.py \
        --suite cache-sim --no-build \
        --benchmark "$app" --options "$opts" \
        --policies "${BASELINES[@]}" \
        --l1d-size 32kB --l1d-ways 8 \
        --l2-size 256kB --l2-ways 8 \
        --l3-sizes 1MB --l3-ways 16 \
        --line-size 64 --timeout-cache 1800 \
        --out-dir "$cell_dir/baselines" >> "$LOG" 2>&1; then
      date +"%T OK    ${short}/${app} baselines" | tee -a "$LOG"
    else
      date +"%T FAIL  ${short}/${app} baselines" | tee -a "$LOG"
      fail_count=$((fail_count + 1))
      continue
    fi
    fi

    # --- Pass 2: ECG_DBG eviction + ECG_PFX prefetcher (combined arm) ---
    if [ -f "$cell_dir/pfx_combined/roi_matrix.csv" ]; then
      pfx_status="$(awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="status")s=i;next} s{print $s; exit}' "$cell_dir/pfx_combined/roi_matrix.csv" 2>/dev/null || echo '')"
    else
      pfx_status=''
    fi
    if [ "$pfx_status" = "ok" ]; then
      date +"%T SKIP  ${short}/${app} pfx_combined (cached)" | tee -a "$LOG"
    else
    date +"%T BEGIN ${short}/${app} PFX_COMBINED (ECG:DBG_ONLY + ECG_PFX)" | tee -a "$LOG"
    if ECG_CONTAINER_BITS=64 timeout 3600 python3 scripts/experiments/ecg/roi_matrix.py \
        --suite cache-sim --no-build \
        --benchmark "$app" --options "$opts" \
        --policies "$PFX_POLICY" \
        --prefetcher ECG_PFX --ecg-pfx-mode popt --ecg-pfx-lookahead 8 \
        --l1d-size 32kB --l1d-ways 8 \
        --l2-size 256kB --l2-ways 8 \
        --l3-sizes 1MB --l3-ways 16 \
        --line-size 64 --timeout-cache 1800 \
        --out-dir "$cell_dir/pfx_combined" >> "$LOG" 2>&1; then
      date +"%T OK    ${short}/${app} pfx_combined" | tee -a "$LOG"
      cell_count=$((cell_count + 1))
    else
      date +"%T FAIL  ${short}/${app} pfx_combined" | tee -a "$LOG"
      fail_count=$((fail_count + 1))
    fi
    fi

    # --- Pass 3: ECG_DBG eviction + DROPLET prefetcher (comparator) ---
    if [ -f "$cell_dir/droplet_combined/roi_matrix.csv" ]; then
      drop_status="$(awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="status")s=i;next} s{print $s; exit}' "$cell_dir/droplet_combined/roi_matrix.csv" 2>/dev/null || echo '')"
    else
      drop_status=''
    fi
    if [ "$drop_status" = "ok" ]; then
      date +"%T SKIP  ${short}/${app} droplet_combined (cached)" | tee -a "$LOG"
    else
    date +"%T BEGIN ${short}/${app} DROPLET_COMBINED (ECG:DBG_ONLY + DROPLET)" | tee -a "$LOG"
    if ECG_CONTAINER_BITS=64 timeout 3600 python3 scripts/experiments/ecg/roi_matrix.py \
        --suite cache-sim --no-build \
        --benchmark "$app" --options "$opts" \
        --policies "$PFX_POLICY" \
        --prefetcher DROPLET --ecg-pfx-lookahead 8 \
        --l1d-size 32kB --l1d-ways 8 \
        --l2-size 256kB --l2-ways 8 \
        --l3-sizes 1MB --l3-ways 16 \
        --line-size 64 --timeout-cache 1800 \
        --out-dir "$cell_dir/droplet_combined" >> "$LOG" 2>&1; then
      date +"%T OK    ${short}/${app} droplet_combined" | tee -a "$LOG"
    else
      date +"%T FAIL  ${short}/${app} droplet_combined" | tee -a "$LOG"
      fail_count=$((fail_count + 1))
    fi
    fi
  done
done

date +"%Y-%m-%d %T --- finished ECG_PFX cache_sim scale sweep: ran=${cell_count} skip=${skip_count} fail=${fail_count} ---" | tee -a "$LOG"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$LOG"
