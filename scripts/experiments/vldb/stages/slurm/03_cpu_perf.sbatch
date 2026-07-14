#!/bin/bash
#SBATCH --job-name=gbrew-cpuperf
#SBATCH --partition=standard
#SBATCH --time=12:00:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=results/slurm_logs/%x-%j.out
# ---------------------------------------------------------------------------
# Stage 03 — CPU wall-clock benchmark sweep.
# Real-hardware timing. Use a fast, isolated partition for repeatable numbers.
# Requires stage 01 (.sg files) and stage 02 (.lo mappings) completed.
# Override via env: EXP, GRAPHS, EXTRA_ARGS
#   sbatch --export=ALL,EXP=2,GRAPHS="com-Orkut" 03_cpu_perf.sbatch
# For largemem graphs (twitter7, webbase, kron, uk-2002, indochina, wikipedia):
#   sbatch --partition=largemem --mem=512G --export=ALL,EXP=2,GRAPHS=twitter7 03_cpu_perf.sbatch
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p results/slurm_logs
module load gcc miniforge || true
source .venv/bin/activate 2>/dev/null || true

EXP="${EXP:-2}"
GRAPHS="${GRAPHS:-}"
EXTRA="${EXTRA_ARGS:-}"

GRAPH_FLAG=""
if [[ -n "$GRAPHS" ]]; then GRAPH_FLAG="--graphs $GRAPHS"; fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp "$EXP" $GRAPH_FLAG $EXTRA
