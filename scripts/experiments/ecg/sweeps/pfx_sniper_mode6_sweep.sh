#!/usr/bin/env bash
# pfx_sniper_mode6_sweep.sh — Sniper cycle-accurate validation of ECG mode 6
# (per-edge mask) added in sprint 6f-6 (commit 1aa1b24b).
#
# Mirrors pfx_sniper_validation_sweep.sh (which exercises mode 2 = POPT
# lookahead, the cache_sim-default for the Sniper validation arm). This
# variant adds the ECG_PFX arm at mode 6 so the cross-sim audit covers
# the paper's headline design.
#
# Output: /tmp/graphbrew-pfx-sniper-mode6/<graph>-<app>/<arm>/roi_matrix.csv
#   (separate from /tmp/graphbrew-pfx-sniper-validation to preserve the
#   existing mode-2 results.)
#
# Cells: 4 graphs × pr × 3 arms = 12 observations. Mode 6 is PR-only
# (cache_sim mode 6 is also PR-only).
#
# Usage:
#   bash scripts/experiments/ecg/sweeps/pfx_sniper_mode6_sweep.sh
#   GRAPH_FILTER=delaunay_n19 bash scripts/.../pfx_sniper_mode6_sweep.sh
#   DEFAULT_FILTER= bash scripts/.../pfx_sniper_mode6_sweep.sh  # all graphs

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="${OUT_ROOT:-/tmp/graphbrew-pfx-sniper-mode6}"
LOG="${OUT_ROOT}/runner_pfx_sniper_mode6.log"
mkdir -p "$OUT_ROOT"
date +"%Y-%m-%d %T --- start Sniper PFX validation sweep (mode 6) ---" | tee -a "$LOG"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$LOG"

DEFAULT_FILTER="${DEFAULT_FILTER:-email-Eu-core delaunay_n19}"

declare -A GRAPH_PATHS=(
  [email-Eu-core]=results/graphs/email-Eu-core/email-Eu-core.sg
  [delaunay_n19]=results/graphs/delaunay_n19/delaunay_n19.sg
  [roadNet-CA]=results/graphs/roadNet-CA/roadNet-CA.sg
  [web-Google]=results/graphs/web-Google/web-Google.sg
)

declare -A GRAPH_APPS=(
  [email-Eu-core]="pr"
  [delaunay_n19]="pr"
  [roadNet-CA]="pr"
  [web-Google]="pr"
)

declare -A APP_OPTS=(
  [pr]="-f {graph_path} -s -o 5 -n 1 -i 2"
)

ARMS=(none DROPLET ECG_PFX)
# Where to attach the prefetcher. Default is l2 (matches the prior
# sweep). Set PREFETCH_LEVEL=l1d for paper-faithful comparison with
# the original DROPLET design (Basak HPCA'19), which prefetches at L1.
PREFETCH_LEVEL="${PREFETCH_LEVEL:-l2}"

cell_arm_complete() {
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

is_in_default_filter() {
  local g="$1"
  for d in $DEFAULT_FILTER; do
    [ "$d" = "$g" ] && return 0
  done
  return 1
}

run_count=0
skip_count=0
fail_count=0
for short in "${!GRAPH_PATHS[@]}"; do
  if [ -n "${GRAPH_FILTER:-}" ]; then
    [ "$short" != "${GRAPH_FILTER}" ] && continue
  elif ! is_in_default_filter "$short"; then
    date +"%T DEFER ${short} (not in DEFAULT_FILTER='$DEFAULT_FILTER')" | tee -a "$LOG"
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
    opts="${APP_OPTS[$app]//\{graph_path\}/$path}"
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
      mode_arg=()
      if [ "$arm" = "DROPLET" ]; then
        pfx_arg=(--prefetcher DROPLET --prefetcher-level "$PREFETCH_LEVEL")
      elif [ "$arm" = "ECG_PFX" ]; then
        pfx_arg=(--prefetcher ECG_PFX --prefetcher-level "$PREFETCH_LEVEL")
        mode_arg=(--ecg-pfx-mode per_edge)
      fi
      date +"%T BEGIN ${short}/${app}/${arm} opts='${opts}'" | tee -a "$LOG"
      if ECG_CONTAINER_BITS=64 timeout 6000 python3 scripts/experiments/ecg/roi_matrix.py \
          --suite sniper --no-build \
          --benchmark "$app" \
          --sniper-workload sg_kernel --allow-sniper-sg-kernel-workload \
          --options "$opts" \
          --policies LRU \
          "${pfx_arg[@]}" "${mode_arg[@]}" \
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

date +"%Y-%m-%d %T --- finished Sniper PFX validation sweep (mode 6): ran=${run_count} skip=${skip_count} fail=${fail_count} ---" | tee -a "$LOG"
echo "OUT_ROOT=$OUT_ROOT" | tee -a "$LOG"
