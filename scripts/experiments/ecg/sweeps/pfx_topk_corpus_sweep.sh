#!/usr/bin/env bash
# pfx_topk_corpus_sweep.sh — sprint 6f-3 corpus validation.
#
# Runs ECG_PFX with Top-K=1 (default), Top-K=4 (matched-bandwidth with
# DROPLET), and DROPLET across the 12 active literature cells from
# sprint 6f. Answers: does the cit-Patents/pr finding (K=4 ties DROPLET,
# K=1 wins per-request) hold corpus-wide?

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="${OUT_ROOT:-/tmp/graphbrew-ecg-pfx-topk-corpus}"
LOG="${OUT_ROOT}/runner.log"
mkdir -p "$OUT_ROOT"
date +"%Y-%m-%d %T --- start ECG_PFX Top-K corpus sweep ---" | tee -a "$LOG"

# 12 active cells (BC cells excluded — kernel emits no hints).
declare -a CELLS=(
  "cit-Patents:pr:0"
  "cit-Patents:bfs:1500000"
  "cit-Patents:sssp:1500000"
  "com-orkut:pr:0"
  "soc-LiveJournal1:pr:0"
  "soc-LiveJournal1:bfs:2000000"
  "soc-LiveJournal1:sssp:2000000"
  "soc-pokec:pr:0"
  "soc-pokec:sssp:800000"
  "web-Google:pr:0"
  "web-Google:bfs:0"
)

cell_complete() {
  [ -f "$1" ] || return 1
  local status
  status="$(awk -F, 'NR==1 { for (i=1;i<=NF;i++) if($i=="status") si=i; next }
    si { print $si; exit }' "$1" 2>/dev/null || echo '')"
  case "$status" in ok|active_no_fill|inactive) return 0 ;; *) return 1 ;; esac
}

run_arm() {
  local graph="$1" app="$2" src="$3" arm="$4" outdir="$5"
  local opts
  case "$app" in
    pr)   opts="-f results/graphs/$graph/$graph.sg -s -o 5 -n 1 -i 2" ;;
    bfs)  opts="-f results/graphs/$graph/$graph.sg -s -o 5 -n 1 -r $src" ;;
    sssp) opts="-f results/graphs/$graph/$graph.sg -s -o 5 -n 1 -r $src" ;;
  esac
  local pfx_args topk mode
  case "$arm" in
    K1)      pfx_args=(--prefetcher ECG_PFX --prefetcher-level l2); topk=1; mode=2 ;;
    K4)      pfx_args=(--prefetcher ECG_PFX --prefetcher-level l2); topk=4; mode=2 ;;
    DROPLET) pfx_args=(--prefetcher DROPLET --prefetcher-level l2); topk=1; mode=3 ;;
  esac
  date +"%T BEGIN ${graph}/${app}/${arm}" | tee -a "$LOG"
  if ECG_CONTAINER_BITS=64 ECG_PREFETCH_LOOKAHEAD=8 \
     ECG_PREFETCH_MODE="$mode" ECG_PREFETCH_TOP_K="$topk" \
     timeout 1800 python3 scripts/experiments/ecg/roi_matrix.py \
     --suite cache-sim --no-build \
     --benchmark "$app" --options "$opts" \
     --policies ECG_DBG \
     "${pfx_args[@]}" \
     --l1d-size 32kB --l1d-ways 8 --l2-size 256kB --l2-ways 8 \
     --l3-sizes 1MB --l3-ways 16 --line-size 64 \
     --timeout-cache 1500 --out-dir "$outdir" >> "$LOG" 2>&1; then
    date +"%T OK    ${graph}/${app}/${arm}" | tee -a "$LOG"
    return 0
  else
    date +"%T FAIL  ${graph}/${app}/${arm}" | tee -a "$LOG"
    return 1
  fi
}

ok=0; skip=0; fail=0
for cell in "${CELLS[@]}"; do
  IFS=':' read -r graph app src <<< "$cell"
  for arm in K1 K4 DROPLET; do
    out="${OUT_ROOT}/${graph}-${app}/${arm}"
    csv="${out}/roi_matrix.csv"
    if cell_complete "$csv"; then
      date +"%T SKIP  ${graph}/${app}/${arm}" | tee -a "$LOG"
      skip=$((skip+1)); continue
    fi
    mkdir -p "$out"
    if run_arm "$graph" "$app" "$src" "$arm" "$out"; then
      ok=$((ok+1))
    else
      fail=$((fail+1))
    fi
  done
done
date +"%Y-%m-%d %T --- finished ECG_PFX Top-K corpus sweep: ok=$ok skip=$skip fail=$fail ---" | tee -a "$LOG"
