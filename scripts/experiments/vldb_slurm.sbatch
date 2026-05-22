#!/bin/bash
# =============================================================================
# vldb_slurm.sbatch — SLURM template for GraphBrew VLDB experiments on UVA
#
# Submit one job per (experiment, graph) so that:
#  - A timeout or node failure loses only that single slice
#  - --resume (built into ResultsStore) skips already-completed cells
#  - Per-job stdout/stderr stays small and greppable
#
# USAGE — submit a single experiment on a single graph:
#   sbatch --export=ALL,EXP=8,GRAPH=hollywood-2009 \
#          scripts/experiments/vldb_slurm.sbatch
#
# USAGE — fan out all experiments × all 64GB graphs:
#   for g in cit-Patents soc-pokec hollywood-2009 soc-LiveJournal1 \
#            com-Orkut USA-road-d.USA kron_g500-logn21 indochina-2004 uk-2002; do
#     for exp in 2 3 5 7 8; do
#       sbatch --export=ALL,EXP=$exp,GRAPH=$g \
#              scripts/experiments/vldb_slurm.sbatch
#     done
#   done
#
# USAGE — re-run a job after it timed out (will pick up where it left off):
#   sbatch --export=ALL,EXP=2,GRAPH=twitter7 \
#          scripts/experiments/vldb_slurm.sbatch
# =============================================================================

#SBATCH --job-name=gbrew-vldb
#SBATCH --partition=standard                # adjust for your UVA partition
#SBATCH --account=YOUR_ALLOC                # CHANGE ME
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32                  # max OMP threads we'll sweep
#SBATCH --mem=128G                          # bump to 256G for twitter/webbase
#SBATCH --time=24:00:00                     # ResultsStore makes restarts cheap
#SBATCH --output=results/slurm_logs/%x-%j-exp${EXP}-${GRAPH}.out
#SBATCH --error=results/slurm_logs/%x-%j-exp${EXP}-${GRAPH}.err

set -euo pipefail

# ---- required env vars (passed via --export=ALL,EXP=...,GRAPH=...) ----
: "${EXP:?must set EXP=<1-8>}"
: "${GRAPH:?must set GRAPH=<graph-name>}"

# Optional overrides
GRAPHSET="${GRAPHSET:-64gb}"      # 64gb | local | full
EXTRA_ARGS="${EXTRA_ARGS:-}"      # e.g. "--dry-run"

REPO_ROOT="${REPO_ROOT:-$SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"

mkdir -p results/slurm_logs

# ---- environment ----
module purge 2>/dev/null || true
module load gcc/12 python/3.11 2>/dev/null || true   # adjust to UVA modules

# Use the repo's venv if present; create one if not
if [[ -d .venv ]]; then
  source .venv/bin/activate
fi

# Pin threads for predictable scaling numbers
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

# ---- build (idempotent — skip-setup handles already-built case) ----
if [[ ! -x bench/bin/pr ]]; then
  make -j"$OMP_NUM_THREADS" pr bfs cc sssp bc tc converter
fi

# ---- print provenance to the job log ----
echo "================================================================"
echo "  host:      $(hostname)"
echo "  date:      $(date)"
echo "  jobid:     ${SLURM_JOB_ID:-N/A}"
echo "  partition: ${SLURM_JOB_PARTITION:-N/A}"
echo "  cpus:      ${SLURM_CPUS_PER_TASK:-N/A}"
echo "  mem:       ${SLURM_MEM_PER_NODE:-N/A} MB"
echo "  git:       $(git rev-parse HEAD 2>/dev/null || echo n/a)"
echo "  EXP=$EXP  GRAPH=$GRAPH  GRAPHSET=$GRAPHSET"
echo "================================================================"

# ---- run ----
# vldb_paper_experiments.py with ResultsStore checkpoints after every cell,
# so re-submitting this exact job picks up where it left off automatically.
GRAPHSET_FLAG=""
case "$GRAPHSET" in
  64gb)  GRAPHSET_FLAG="--64gb"  ;;
  local) GRAPHSET_FLAG="--local" ;;
  full)  GRAPHSET_FLAG=""        ;;
esac

# --graphs <name> restricts to that single graph (matches name in EVAL_GRAPHS).
python3 scripts/experiments/vldb_paper_experiments.py \
    --exp "$EXP" \
    --graphs "$GRAPH" \
    $GRAPHSET_FLAG \
    --skip-setup --skip-download --no-figures \
    $EXTRA_ARGS

echo "================================================================"
echo "  done: $(date)"
echo "================================================================"
