#!/usr/bin/env bash
set -euo pipefail

# Timeout-safe PR wallclock sweep for GraphBrew variants.
#
# Example:
#   OMP_NUM_THREADS=8 ./scripts/experiments/vldb_pr_wallclock_sweep.sh
#
# Override defaults:
#   TRIALS=5 TIMEOUT_SEC=180 VARIANTS="hrab hlr" \
#   GRAPHS="hollywood-2009 com-Orkut" \
#   ./scripts/experiments/vldb_pr_wallclock_sweep.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

GRAPHS="${GRAPHS:-hollywood-2009 com-Orkut soc-LiveJournal1 soc-pokec}"
VARIANTS="${VARIANTS:-hrab hlr}"
TRIALS="${TRIALS:-3}"
ITERATIONS="${ITERATIONS:-20}"
TIMEOUT_SEC="${TIMEOUT_SEC:-180}"
THREADS="${OMP_NUM_THREADS:-8}"
OUT_DIR="${OUT_DIR:-results/data/raw_2026_05_19_hlr}"

mkdir -p "$OUT_DIR"
STAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$OUT_DIR/pr_wallclock_sweep_${STAMP}.log"
CSV_FILE="$OUT_DIR/pr_wallclock_sweep_${STAMP}.csv"

printf "graph,variant,trial,time_sec,status\n" > "$CSV_FILE"
printf "# threads=%s iterations=%s trials=%s timeout_sec=%s\n" \
  "$THREADS" "$ITERATIONS" "$TRIALS" "$TIMEOUT_SEC" | tee -a "$LOG_FILE"

run_one() {
  local graph="$1"
  local variant="$2"
  local trial="$3"

  printf "%s/%s/T%s: " "$graph" "$variant" "$trial" | tee -a "$LOG_FILE"

  local line
  line=$(timeout "${TIMEOUT_SEC}s" bash -lc \
    "OMP_NUM_THREADS=$THREADS ./bench/bin/pr -f results/graphs/$graph/$graph.sg -n 1 -i $ITERATIONS -o 12:$variant 2>&1 | grep 'Average Time:' | head -1" || true)

  if [[ -n "$line" ]]; then
    local t
    t="$(awk '{print $3}' <<< "$line")"
    printf "%s\n" "$line" | tee -a "$LOG_FILE"
    printf "%s,%s,%s,%s,ok\n" "$graph" "$variant" "$trial" "$t" >> "$CSV_FILE"
  else
    printf "TIMEOUT_OR_NO_OUTPUT\n" | tee -a "$LOG_FILE"
    printf "%s,%s,%s,,timeout_or_no_output\n" "$graph" "$variant" "$trial" >> "$CSV_FILE"
  fi
}

for g in $GRAPHS; do
  for v in $VARIANTS; do
    for t in $(seq 1 "$TRIALS"); do
      run_one "$g" "$v" "$t"
    done
  done
done

printf "\nCSV=%s\nLOG=%s\n" "$CSV_FILE" "$LOG_FILE"

awk -F, '
NR==1 { next }
$5=="ok" {
  key=$1 SUBSEP $2
  sum[key]+=$4
  n[key]++
  graphs[$1]=1
}
END {
  print ""
  printf("%-18s  %10s  %10s  %10s\n", "graph", "hrab_mean", "hlr_mean", "delta_%")
  for (g in graphs) {
    hk=g SUBSEP "hrab"
    lk=g SUBSEP "hlr"
    if (n[hk] > 0 && n[lk] > 0) {
      h=sum[hk]/n[hk]
      l=sum[lk]/n[lk]
      d=(l/h-1.0)*100.0
      printf("%-18s  %10.5f  %10.5f  %10.2f\n", g, h, l, d)
      ratio=l/h
      if (ratio > 0) {
        logsum += log(ratio)
        m++
      }
    }
  }
  if (m > 0) {
    geo=(exp(logsum/m)-1.0)*100.0
    printf("GEOMEAN_DELTA_PCT %.2f\n", geo)
  }
}
' "$CSV_FILE"
