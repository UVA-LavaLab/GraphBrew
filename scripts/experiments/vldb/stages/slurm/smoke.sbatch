#!/bin/bash
#SBATCH --job-name=gbrew-smoke
#SBATCH --partition=standard
#SBATCH --time=00:30:00
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=results/slurm_logs/%x-%j.out
# ---------------------------------------------------------------------------
# SLURM smoke test — runs the full stages 02 -> 03 -> 05 pipeline in
# --preview mode on the 2 tiny graphs (email-Eu-core, cit-Patents).
# Use to validate the SLURM environment, module loads, OMP settings, and
# the entire result-writing pipeline (~10 min wall on a 16-core node).
#
# Submit:
#   sbatch scripts/experiments/vldb/stages/slurm/smoke.sbatch
#
# After it returns, expect:
#   results/vldb_mappings/{email-Eu-core,cit-Patents}/   (74 files each)
#   results/vldb_runs/{email-Eu-core,cit-Patents}/       (60 files each)
#   results/vldb_paper/exp2_speedup/speedup_results.json (120 rows)
#   results/INDEX.json
#   results/vldb_paper/figures/fig2_*.png
# ---------------------------------------------------------------------------
set -euo pipefail
mkdir -p results/slurm_logs
module load gcc miniforge || true
source .venv/bin/activate 2>/dev/null || true

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

echo "=== STAGE 02: reorder cache (.lo) ==="
time python3 scripts/experiments/vldb/stages/02_reorder.py --exp 2 --preview

echo "=== STAGE 03: CPU benchmarks ==="
time python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --preview

echo "=== STAGE 05: aggregate + figures + INDEX.json ==="
python3 scripts/experiments/vldb/stages/05_aggregate.py --exp 0

echo
echo "=== Smoke-test result counts ==="
python3 -m scripts.lib.analysis.results_index --print-summary
