#!/usr/bin/env bash
# pfx_sniper_validation_sweep.sh — Sniper cycle-accurate validation of the
# ECG_PFX vs DROPLET efficiency claim from sprint 6e.
#
# Runs the 3-arm matched-proof sweep (none/DROPLET/ECG_PFX) on graphs that
# fit the workstation Sniper budget (~10-30 min/cell each):
#   email-Eu-core   (1k vertices, anchor)
#   delaunay_n19    (524k vertices, mesh)
#   roadNet-CA      (1.9M vertices, road network)
#   web-Google      (875k vertices, web)
# × pr/bfs/sssp = 12 cells × 3 arms = 36 observations.
#
# Uses the patched sg_kernel binary (commit 1247565) with frontier-head
# lookahead for BFS/SSSP and env-tunable lookahead for PR, so ECG_PFX
# generates meaningful prefetch activity (not the lookahead-1 sprint 6b
# limitation).
#
# Output: /tmp/graphbrew-pfx-sniper-validation/<graph>-<app>/<arm>/roi_matrix.csv
#
# Usage:
#   bash scripts/experiments/ecg/sweeps/pfx_sniper_validation_sweep.sh
#
# Filter while debugging:
#   GRAPH_FILTER=delaunay_n19 APP_FILTER=pr bash scripts/.../pfx_sniper_validation_sweep.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="${OUT_ROOT:-/tmp/graphbrew-pfx-sniper-validation}"
LOG="${OUT_ROOT}/runner_pfx_sniper_validation.log"
mkdir -p "$OUT_ROOT"
date +"%Y-%m-%d %T --- start Sniper PFX validation sweep ---" | tee -a "$LOG"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$LOG"

declare -A GRAPH_PATHS=(
  [email-Eu-core]=results/graphs/email-Eu-core/email-Eu-core.sg
  [delaunay_n19]=results/graphs/delaunay_n19/delaunay_n19.sg
  [roadNet-CA]=results/graphs/roadNet-CA/roadNet-CA.sg
  [web-Google]=results/graphs/web-Google/web-Google.sg
)

declare -A GRAPH_APPS=(
  [email-Eu-core]="pr bfs sssp"
  [delaunay_n19]="pr bfs sssp"
  [roadNet-CA]="pr bfs sssp"
  [web-Google]="pr bfs sssp"
)

# Per-graph BFS/SSSP source vertices. delaunay_n19 is uniformly connected;
# roadNet-CA needs middle vertex; web-Google works from 0.
declare -A BFS_SRC=(
  [email-Eu-core]=0 [delaunay_n19]=100000 [roadNet-CA]=500000 [web-Google]=0
)
declare -A SSSP_SRC=(
  [email-Eu-core]=0 [delaunay_n19]=100000 [roadNet-CA]=500000 [web-Google]=0
)

declare -A APP_OPTS=(
  [pr]="-f {graph_path} -s -o 5 -n 1 -i 2"
  [bfs]="-f {graph_path} -s -o 5 -n 1 -r {bfs_src}"
  [sssp]="-f {graph_path} -s -o 5 -n 1 -r {sssp_src}"
)

ARMS=(none DROPLET ECG_PFX)

cell_arm_complete() {
  local csv="$1"
  [ -f "$csv" ] || return 1
  local status
  status="$(awk -F, '
    NR==1 { for (i=1;i<=NF;i++) if($i=="status") si=i; next }
    si { print $si; exit }
  ' "$csv" 2>/dev/null || echo '')"
  # Accept ok, active_no_fill, inactive — reject error
  case "$status" in
    ok|active_no_fill|inactive) return 0 ;;
    *) return 1 ;;
  esac
}

run_count=0
skip_count=0
fail_count=0
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
    bfs_src="${BFS_SRC[$short]:-0}"
    sssp_src="${SSSP_SRC[$short]:-0}"
    opts="${APP_OPTS[$app]//\{graph_path\}/$path}"
    opts="${opts//\{bfs_src\}/$bfs_src}"
    opts="${opts//\{sssp_src\}/$sssp_src}"
    for arm in "${ARMS[@]}"; do
      if [ -n "${ARM_FILTER:-}" ] && [ "$arm" != "${ARM_FILTER}" ]; then
        continue
      fi
      outdir="${OUT_ROOT}/${short}-${app}/${arm}"
      csv="${outdir}/roi_matrix.csv"
      if cell_arm_complete "$csv"; then
        date +"%T SKIP  ${short}/${app}/${arm}" | tee -a "$LOG"
        skip_count=$((skip_count + 1))
        continue
      fi
      mkdir -p "$outdir"
      pfx_arg=()
      if [ "$arm" != "none" ]; then
        pfx_arg=(--prefetcher "$arm" --prefetcher-level l2)
      fi
      # ECG_PFX needs container=64 to give PFX bits room (per sprint 6c)
      date +"%T BEGIN ${short}/${app}/${arm} opts='${opts}'" | tee -a "$LOG"
      if ECG_CONTAINER_BITS=64 timeout 6000 python3 scripts/experiments/ecg/roi_matrix.py \
          --suite sniper --no-build \
          --benchmark "$app" \
          --sniper-workload sg_kernel --allow-sniper-sg-kernel-workload \
          --options "$opts" \
          --policies LRU \
          "${pfx_arg[@]}" \
          --l1d-size 32kB --l1d-ways 8 \
          --l2-size 256kB --l2-ways 8 \
          --l3-sizes 1MB --l3-ways 16 \
          --line-size 64 \
          --timeout-sniper 5400 \
          --out-dir "$outdir" >> "$LOG" 2>&1; then
        date +"%T OK    ${short}/${app}/${arm}" | tee -a "$LOG"
        run_count=$((run_count + 1))
      else
        date +"%T FAIL  ${short}/${app}/${arm}" | tee -a "$LOG"
        fail_count=$((fail_count + 1))
      fi
    done
  done
done

date +"%Y-%m-%d %T --- finished Sniper PFX validation sweep: ran=${run_count} skip=${skip_count} fail=${fail_count} ---" | tee -a "$LOG"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$LOG"
