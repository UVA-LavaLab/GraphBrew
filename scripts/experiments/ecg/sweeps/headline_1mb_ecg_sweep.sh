#!/usr/bin/env bash
# headline_1mb_ecg_sweep.sh — populate cache_sim ECG_DBG_PRIMARY cells
# at the literature 1MB canonical L3 row for every literature graph.
#
# This is the permanent reproducible version of the ad-hoc
# /tmp/gb-lit-sweep-rN.sh scripts that produced the original
# LRU/SRRIP/GRASP/POPT baseline. Adds ECG:DBG_PRIMARY to the policy
# roster on the same cells, so gate 282 (headline_coverage) and
# gate 283 (headline_parity) can score the ECG column.
#
# Sweep cells (matches literature_concrete_cells filter l3=1MB,
# narrowed to apps {pr,bfs,sssp,bc} per ECG-Final-Runs P-OPT scope):
#   cit-Patents      / {pr,bfs,sssp,bc}
#   com-orkut        / pr
#   soc-LiveJournal1 / {pr,bfs,sssp,bc}
#   soc-pokec        / {pr,sssp,bc}
#   web-Google       / {pr,bfs,bc}
#   email-Eu-core    / pr  (sanity anchor)
# Total: 16 cells (15 literature + 1 sanity).
#
# Output: /tmp/graphbrew-lit-baseline/<graph>-<app>/lit/roi_matrix.csv
#
# IDEMPOTENCY MODEL:
#   roi_matrix.py REWRITES roi_matrix.csv on every invocation from the
#   in-memory row set produced by the current --policies argument. We
#   therefore pass the FULL canonical policy roster (LRU + SRRIP + GRASP
#   + POPT + ECG:DBG_PRIMARY) at the FULL canonical L3 set (1MB, 4MB,
#   8MB) on every run. This re-executes the four baselines on each pass,
#   which is acceptable: cache_sim is fast enough that the redundancy
#   is < 2-3 minutes per graph for the LRU/SRRIP/GRASP/POPT triplet on
#   small/medium graphs, and the CSV stays canonical (a partial CSV would
#   silently mislead lit-faith aggregation). The script skips cells
#   whose CSV already contains an ECG_DBG_PRIMARY row.
#
# Cache hierarchy: GRASP HPCA20 canonical (32kB L1d/8w, 256kB L2/8w,
# 1MB L3/16w, 64B lines). Reorder mode: -o 5 (DBG, required for GRASP).
#
# Usage:
#   bash scripts/experiments/ecg/sweeps/headline_1mb_ecg_sweep.sh
# Or one cell at a time:
#   GRAPH_FILTER=email-Eu-core APP_FILTER=pr bash scripts/.../headline_1mb_ecg_sweep.sh

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

OUT_ROOT="${OUT_ROOT:-/tmp/graphbrew-lit-baseline}"
LOG="${OUT_ROOT}/runner_headline_1mb_ecg.log"
mkdir -p "$OUT_ROOT"
date +"%Y-%m-%d %T --- start headline-1MB ECG sweep ---" | tee -a "$LOG"

# (graph_short, graph_path, apps)
# Apps reflect the literature-canonical cells for each graph per
# literature_baselines.{INVARIANT,PER_GRAPH}_CLAIMS at L3=1MB.
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

# Per-graph source vertices for frontier-driven kernels (bfs/sssp/bc).
# Vertex 0 is degenerate on hub-dominated DBG-reordered layouts of large
# graphs (com-orkut, soc-pokec, soc-LiveJournal1, cit-Patents): the BFS
# frontier dies after 1 hop because vertex 0 becomes the dominant hub
# with thousands of out-edges, all of which short-circuit. Use a
# middle-rank vertex (~vertices/2 after DBG ordering) to produce
# meaningful traversal. Matches /tmp/gb-lit-sweep-r6.sh historical fix.
declare -A BFS_SRC=(
  [email-Eu-core]=0
  [web-Google]=0
  [soc-pokec]=800000
  [cit-Patents]=1500000
  [soc-LiveJournal1]=2000000
  [com-orkut]=1500000
)
declare -A SSSP_SRC=(
  [email-Eu-core]=0
  [web-Google]=0
  [soc-pokec]=800000
  [cit-Patents]=1500000
  [soc-LiveJournal1]=2000000
  [com-orkut]=1500000
)
declare -A BC_SRC=(
  [email-Eu-core]=0
  [web-Google]=0
  [soc-pokec]=800000
  [cit-Patents]=1500000
  [soc-LiveJournal1]=2000000
  [com-orkut]=1500000
)

# Canonical policy roster — must always be passed in full because
# roi_matrix.py emits the CSV from the in-memory row set for THIS
# invocation (it does NOT merge with existing per-policy JSONs).
POLICIES=(LRU SRRIP GRASP POPT "ECG:DBG_PRIMARY")
TARGET_LABEL="ECG_DBG_PRIMARY"   # what we test for in cell_has_policy
# Canonical L3 sweep — match the original lit baseline so we don't
# regress from {1MB,4MB,8MB} to just {1MB}.
L3_SIZES=(1MB 4MB 8MB)

cell_is_complete() {
  # cell_is_complete <csv-path> <expected-policy-label-count>
  # Returns 0 (skip) if the CSV has >= expected_count distinct policy_label values.
  local csv="$1" expected="$2"
  [ -f "$csv" ] || return 1
  local actual
  actual="$(awk -F, '
    NR==1 { for (i=1;i<=NF;i++) if($i=="policy_label") pli=i; next }
    pli { print $pli }
  ' "$csv" | sort -u | wc -l)"
  if [ "$actual" -lt "$expected" ]; then
    return 1
  fi
  # Also reject degenerate cells: if avg total_accesses < 100 the
  # frontier source was likely a degenerate hub (see r6 historical fix).
  local avg_acc
  avg_acc="$(awk -F, '
    NR==1 { for (i=1;i<=NF;i++) if($i=="total_accesses") ti=i; next }
    ti { sum+=$ti; n++ } END { if(n>0) print int(sum/n); else print 0 }
  ' "$csv")"
  if [ "$avg_acc" -lt 100 ]; then
    return 1  # degenerate, re-run
  fi
  return 0  # complete
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
    outdir="${OUT_ROOT}/${short}-${app}/lit"
    csv="${outdir}/roi_matrix.csv"
    if cell_is_complete "$csv" "${#POLICIES[@]}"; then
      date +"%T SKIP  ${short}/${app} (CSV has >= ${#POLICIES[@]} policy_labels)" | tee -a "$LOG"
      skip_count=$((skip_count + 1))
      continue
    fi
    mkdir -p "$outdir"
    bfs_src="${BFS_SRC[$short]:-0}"
    sssp_src="${SSSP_SRC[$short]:-0}"
    bc_src="${BC_SRC[$short]:-0}"
    opts="${APP_OPTS[$app]//\{graph_path\}/$path}"
    opts="${opts//\{bfs_src\}/$bfs_src}"
    opts="${opts//\{sssp_src\}/$sssp_src}"
    opts="${opts//\{bc_src\}/$bc_src}"
    date +"%T BEGIN ${short}/${app} L3=${L3_SIZES[*]} policies=${POLICIES[*]} opts='${opts}'" | tee -a "$LOG"
    if python3 scripts/experiments/ecg/roi_matrix.py \
        --suite cache-sim \
        --benchmark "$app" \
        --options "$opts" \
        --policies "${POLICIES[@]}" \
        --l1d-size 32kB --l1d-ways 8 \
        --l2-size 256kB --l2-ways 8 \
        --l3-sizes "${L3_SIZES[@]}" --l3-ways 16 \
        --line-size 64 \
        --timeout-cache 5400 \
        --out-dir "$outdir" \
        --no-build >> "$LOG" 2>&1; then
      date +"%T OK    ${short}/${app}" | tee -a "$LOG"
      cell_count=$((cell_count + 1))
    else
      date +"%T FAIL  ${short}/${app}" | tee -a "$LOG"
    fi
  done
done

date +"%Y-%m-%d %T --- finished headline-1MB ECG sweep: ${cell_count} new cells, ${skip_count} skipped ---" | tee -a "$LOG"
