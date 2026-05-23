#!/bin/bash
# ---------------------------------------------------------------------------
# Chained SLURM launcher — one job per stage with proper dependencies.
# Submits 01 -> 02 -> 03 -> [04 cache, optional] -> 05 figures.
#
# Usage:
#   bash scripts/experiments/vldb/stages/slurm/run_all.sh                  # full sweep
#   bash scripts/experiments/vldb/stages/slurm/run_all.sh --preview        # 2 tiny graphs
#   bash scripts/experiments/vldb/stages/slurm/run_all.sh --64gb           # 11 eval graphs
#   GRAPHS="cit-Patents com-Orkut" bash .../run_all.sh                     # specific graphs
#   SKIP_CACHE=1 bash .../run_all.sh                                       # skip stage 04
#   SKIP_PREP=1  bash .../run_all.sh                                       # graphs already on disk
# ---------------------------------------------------------------------------
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
EXTRA="${EXTRA_ARGS:-${*:-}}"

submit() {
    local name="$1"; shift
    sbatch --parsable "$@" "$HERE/$name"
}

J1=""
if [[ -z "${SKIP_PREP:-}" ]]; then
    J1=$(submit 01_prep.sbatch --export=ALL,EXTRA_ARGS="$EXTRA")
    echo "01_prep   -> $J1"
fi
DEP_J1=${J1:+--dependency=afterok:$J1}

J2=$(submit 02_reorder.sbatch $DEP_J1 --export=ALL,EXTRA_ARGS="$EXTRA")
echo "02_reorder -> $J2"

J3=$(submit 03_cpu_perf.sbatch --dependency=afterok:$J2 --export=ALL,EXTRA_ARGS="$EXTRA")
echo "03_cpu    -> $J3"

DEP="afterok:$J3"
if [[ -z "${SKIP_CACHE:-}" ]]; then
    J4=$(submit 04_cache_sim.sbatch --dependency=afterok:$J2 --export=ALL,EXTRA_ARGS="$EXTRA")
    echo "04_cache  -> $J4"
    DEP="afterok:$J3:$J4"
fi

J5=$(submit 05_aggregate.sbatch --dependency=$DEP)
echo "05_aggreg -> $J5"

echo
echo "Track:    squeue -u \$USER"
echo "Final:    cat results/slurm_logs/gbrew-aggregate-$J5.out"
