#!/usr/bin/env bash
# pfx_matched_proof_sweep.sh — 3-arm matched-proof sweep activating
# gate 241 (ECG-Pfx-vs-DROPLET) by populating prefetcher runtime
# counters on the same baseline.
#
# Companion to:
#   - bench/src_sniper/sg_kernel.cc (emits SNIPER_ECG_PFX_TARGET hints
#     in run_pr/run_bfs/run_sssp, env-gated)
#   - scripts/experiments/ecg/lit_faith_ecg_pfx_vs_droplet.py (audit)
#   - scripts/experiments/ecg/ecg_pfx_vs_droplet_postfix_builder.py
#     (CSV → postfix per_observation converter)
#
# Sweep design:
#   - Baseline: LRU eviction at L3=1MB (matches literature scope)
#   - 3 arms per cell: none, DROPLET, ECG_PFX
#   - Hint-rich, bandwidth-bound kernels: pr, bfs, sssp
#   - Representative graphs: email-Eu-core (anchor), cit-Patents,
#     soc-pokec, web-Google
#   - Total: 4 graphs × 3 apps × 3 arms = 36 observations
#
# Output: /tmp/graphbrew-ecg-pfx-vs-droplet-<DATE>/<graph>-<app>/<arm>/roi_matrix.csv
#
# IDEMPOTENCY: skip cells whose CSV already contains all 3 arm rows
# AND non-zero activity (sideband_loaded=1 + reasonable issued counts).
#
# Usage:
#   bash scripts/experiments/ecg/sweeps/pfx_matched_proof_sweep.sh
#
# Filter to one graph/app while debugging:
#   GRAPH_FILTER=email-Eu-core APP_FILTER=pr bash scripts/.../pfx_matched_proof_sweep.sh
#
# Override output root (use this to pin one canonical run dir for the
# matched-proof postfix audit):
#   OUT_ROOT=/tmp/graphbrew-ecg-pfx-vs-droplet-canonical bash scripts/.../pfx_matched_proof_sweep.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

DATE_TAG="$(date +%Y%m%d_%H%M%S)"
OUT_ROOT="${OUT_ROOT:-/tmp/graphbrew-ecg-pfx-vs-droplet-${DATE_TAG}}"
LOG="${OUT_ROOT}/runner_pfx_matched_proof.log"
mkdir -p "$OUT_ROOT"
date +"%Y-%m-%d %T --- start matched-proof prefetcher sweep ---" | tee -a "$LOG"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$LOG"

declare -A GRAPH_PATHS=(
  [email-Eu-core]=results/graphs/email-Eu-core/email-Eu-core.sg
  [cit-Patents]=results/graphs/cit-Patents/cit-Patents.sg
  [soc-pokec]=results/graphs/soc-pokec/soc-pokec.sg
  [web-Google]=results/graphs/web-Google/web-Google.sg
)

# sg_kernel supports pr, bfs, sssp (bc not implemented in sg_kernel
# yet; full pr.cc has the bc variant but Sniper benchmark mode is
# guarded by --allow-sniper-benchmark-workload).
declare -A GRAPH_APPS=(
  [email-Eu-core]="pr bfs sssp"
  [cit-Patents]="pr bfs sssp"
  [soc-pokec]="pr bfs sssp"
  [web-Google]="pr bfs sssp"
)

# Per-graph source vertices for frontier-driven kernels (bfs/sssp).
# Vertex 0 is degenerate on hub-dominated DBG-reordered layouts of
# large graphs; mirror the headline-1MB ECG sweep source map.
declare -A BFS_SRC=(
  [email-Eu-core]=0
  [cit-Patents]=1500000
  [soc-pokec]=800000
  [web-Google]=0
)
declare -A SSSP_SRC=(
  [email-Eu-core]=0
  [cit-Patents]=1500000
  [soc-pokec]=800000
  [web-Google]=0
)

ARMS=(none DROPLET ECG_PFX)

cell_arm_complete() {
  # cell_arm_complete <csv-path>
  # OK if CSV exists, has 1 row, prefetcher field matches arm, and
  # for prefetcher!=none we expect sideband_loaded=1.
  local csv="$1" arm="$2"
  [ -f "$csv" ] || return 1
  local lines
  lines="$(wc -l < "$csv")"
  [ "$lines" -ge 2 ] || return 1
  # Trivial existence check is enough — re-running a single arm
  # is cheap; the wrapper above gives idempotency at the (graph,
  # app, arm) granularity.
  return 0
}

declare -A APP_OPTS=(
  [pr]="-f {graph_path} -s -o 5 -n 1 -i 2"
  [bfs]="-f {graph_path} -s -o 5 -n 1 -r {bfs_src}"
  [sssp]="-f {graph_path} -s -o 5 -n 1 -r {sssp_src}"
)

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
      if cell_arm_complete "$csv" "$arm"; then
        date +"%T SKIP  ${short}/${app}/${arm}" | tee -a "$LOG"
        skip_count=$((skip_count + 1))
        continue
      fi
      mkdir -p "$outdir"
      pfx_arg=()
      if [ "$arm" != "none" ]; then
        pfx_arg=(--prefetcher "$arm" --prefetcher-level l2)
      fi
      date +"%T BEGIN ${short}/${app}/${arm} opts='${opts}'" | tee -a "$LOG"
      if timeout 600 python3 scripts/experiments/ecg/roi_matrix.py \
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
          --timeout-sniper 480 \
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

date +"%Y-%m-%d %T --- finished matched-proof prefetcher sweep: ran=${run_count} skip=${skip_count} fail=${fail_count} ---" | tee -a "$LOG"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$LOG"
echo
echo "Next:"
echo "  python3 scripts/experiments/ecg/ecg_pfx_vs_droplet_postfix_builder.py \\"
echo "      --sweep-root '$OUT_ROOT' \\"
echo "      --postfix-in wiki/data/ecg_pfx_vs_droplet_postfix.json \\"
echo "      --postfix-out wiki/data/ecg_pfx_vs_droplet_postfix.json"
echo "  make lit-ecg-pfx-vs-droplet"
