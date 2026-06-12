#!/usr/bin/env bash
# Full literature-corpus cache_sim regeneration at the GRASP-faithful
# array-relative hot_fraction (default 0.15, set in graph_cache_context.h).
#
# Reproduces the EXACT committed analysis corpus (wiki/data/oracle_gap.csv):
# 8 graphs x their canonical apps x their canonical L3 sizes x 5 policies
# (LRU SRRIP GRASP POPT ECG:DBG_PRIMARY) = 114 (graph,app,L3) cells / ~456 runs.
#
# Output layout matches what `make lit-claims`/`make lit-faith` read:
#   $OUT_ROOT/<graph>-<app>/lit/roi_matrix.csv
#
# Single-threaded cache_sim (roi_matrix defaults --cache-sim-omp-threads 1) for
# deterministic, reproducible miss counts.
#
# Idempotent + resumable: cells whose CSV already has all 5 policies AND all
# expected L3 sizes (non-degenerate) are skipped.
#
# Usage:
#   nohup bash scripts/experiments/ecg/sweeps/full_corpus_lit_regen.sh \
#       > /tmp/full_corpus_lit_regen.out 2>&1 &
#   GRAPH_FILTER=email-Eu-core APP_FILTER=pr bash scripts/.../full_corpus_lit_regen.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="${OUT_ROOT:-/tmp/graphbrew-lit-baseline}"
LOG="${OUT_ROOT}/runner_full_corpus_lit_regen.log"
mkdir -p "$OUT_ROOT"
date +"%Y-%m-%d %T --- start full-corpus lit regen (array-relative GRASP 0.15) ---" | tee -a "$LOG"

declare -A GRAPH_PATHS=(
  [email-Eu-core]=results/graphs/email-Eu-core/email-Eu-core.sg
  [web-Google]=results/graphs/web-Google/web-Google.sg
  [soc-pokec]=results/graphs/soc-pokec/soc-pokec.sg
  [cit-Patents]=results/graphs/cit-Patents/cit-Patents.sg
  [soc-LiveJournal1]=results/graphs/soc-LiveJournal1/soc-LiveJournal1.sg
  [com-orkut]=results/graphs/com-orkut/com-orkut.sg
  [roadNet-CA]=results/graphs/roadNet-CA/roadNet-CA.sg
  [delaunay_n19]=results/graphs/delaunay_n19/delaunay_n19.sg
)

# Canonical apps per graph (mirrors wiki/data/oracle_gap.csv exactly).
declare -A GRAPH_APPS=(
  [email-Eu-core]="pr bfs bc"
  [web-Google]="pr bfs sssp bc cc"
  [soc-pokec]="pr bfs sssp bc cc"
  [cit-Patents]="pr bfs sssp bc cc"
  [soc-LiveJournal1]="pr bfs sssp bc cc"
  [com-orkut]="pr bfs sssp bc cc"
  [roadNet-CA]="pr bfs sssp bc cc"
  [delaunay_n19]="pr"
)

# Per-graph L3 sweep (mirrors the committed corpus): social/web/citation graphs
# use the 1MB/4MB/8MB literature row; the mesh + road graphs (which fit in cache)
# use the small-cache regime 4kB..1MB.
declare -A GRAPH_L3=(
  [email-Eu-core]="1MB 4MB 8MB"
  [web-Google]="1MB 4MB 8MB"
  [soc-pokec]="1MB 4MB 8MB"
  [cit-Patents]="1MB 4MB 8MB"
  [soc-LiveJournal1]="1MB 4MB 8MB"
  [com-orkut]="1MB 4MB 8MB"
  [roadNet-CA]="4kB 16kB 64kB 256kB 1MB"
  [delaunay_n19]="4kB 16kB 64kB 256kB 1MB"
)

declare -A APP_OPTS=(
  [pr]="-f {graph_path} -s -o 5 -n 1 -i 2"
  [bfs]="-f {graph_path} -s -o 5 -n 1 -r {src}"
  [sssp]="-f {graph_path} -s -o 5 -n 1 -r {src}"
  [bc]="-f {graph_path} -s -o 5 -n 1 -r {src} -i 1"
  [cc]="-f {graph_path} -s -o 5 -n 1"
)

# Frontier source vertices (avoid the degenerate hub vertex 0 on DBG-reordered
# large graphs). Matches headline_1mb_ecg_sweep.sh; roadNet-CA ~ vertices/2.
declare -A SRC=(
  [email-Eu-core]=0
  [web-Google]=0
  [soc-pokec]=800000
  [cit-Patents]=1500000
  [soc-LiveJournal1]=2000000
  [com-orkut]=1500000
  [roadNet-CA]=1000000
)

POLICIES=(LRU SRRIP GRASP POPT "ECG:DBG_PRIMARY" "ECG:POPT_PRIMARY")

cell_is_complete() {
  # <csv-path> <expected-policy-count> <expected-l3-count>
  local csv="$1" exp_pol="$2" exp_l3="$3"
  [ -f "$csv" ] || return 1
  local n_pol n_l3 avg_acc
  n_pol="$(awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="policy_label")p=i;next} p{print $p}' "$csv" | sort -u | wc -l)"
  n_l3="$(awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="l3_size")p=i;next} p{print $p}' "$csv" | sort -u | wc -l)"
  avg_acc="$(awk -F, 'NR==1{for(i=1;i<=NF;i++)if($i=="total_accesses")t=i;next} t{s+=$t;n++} END{if(n>0)print int(s/n);else print 0}' "$csv")"
  [ "$n_pol" -ge "$exp_pol" ] && [ "$n_l3" -ge "$exp_l3" ] && [ "$avg_acc" -ge 100 ]
}

cell_count=0; skip_count=0; fail_count=0
for short in "${!GRAPH_PATHS[@]}"; do
  [ -n "${GRAPH_FILTER:-}" ] && [ "$short" != "${GRAPH_FILTER}" ] && continue
  path="${GRAPH_PATHS[$short]}"
  if [ ! -f "$path" ]; then
    date +"%T MISS  ${short} (sg not on disk: $path)" | tee -a "$LOG"; continue
  fi
  l3s="${GRAPH_L3[$short]}"; read -ra L3_ARR <<< "$l3s"
  for app in ${GRAPH_APPS[$short]}; do
    [ -n "${APP_FILTER:-}" ] && [ "$app" != "${APP_FILTER}" ] && continue
    outdir="${OUT_ROOT}/${short}-${app}/lit"
    csv="${outdir}/roi_matrix.csv"
    if cell_is_complete "$csv" "${#POLICIES[@]}" "${#L3_ARR[@]}"; then
      date +"%T SKIP  ${short}/${app} (CSV complete: ${#POLICIES[@]} pol x ${#L3_ARR[@]} L3)" | tee -a "$LOG"
      skip_count=$((skip_count + 1)); continue
    fi
    mkdir -p "$outdir"
    src="${SRC[$short]:-0}"
    opts="${APP_OPTS[$app]//\{graph_path\}/$path}"
    opts="${opts//\{src\}/$src}"
    date +"%T BEGIN ${short}/${app} L3=${l3s} opts='${opts}'" | tee -a "$LOG"
    if python3 scripts/experiments/ecg/roi_matrix.py \
        --suite cache-sim --benchmark "$app" --options "$opts" \
        --policies "${POLICIES[@]}" \
        --l1d-size 32kB --l1d-ways 8 --l2-size 256kB --l2-ways 8 \
        --l3-sizes "${L3_ARR[@]}" --l3-ways 16 --line-size 64 \
        --timeout-cache 7200 --out-dir "$outdir" --no-build >> "$LOG" 2>&1; then
      date +"%T OK    ${short}/${app}" | tee -a "$LOG"; cell_count=$((cell_count + 1))
    else
      date +"%T FAIL  ${short}/${app}" | tee -a "$LOG"; fail_count=$((fail_count + 1))
    fi
  done
done

date +"%Y-%m-%d %T --- finished full-corpus lit regen: ${cell_count} ran, ${skip_count} skipped, ${fail_count} failed ---" | tee -a "$LOG"
