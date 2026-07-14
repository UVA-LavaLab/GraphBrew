# exp3 — Reorder Overhead

**What it measures:** Wall-clock cost of the reorder pass itself
(amortizable cost analysis).

**Output:** `results/vldb_paper/exp3_overhead/overhead_results.json`

Uses the `.time` sidecars produced by [stage 02](../stages/02_reorder.py)
plus a small fresh sweep.

## Run

```bash
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 3 --local
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 3 --local
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 3 --local
```

## SLURM

```bash
sbatch --export=ALL,EXP=3 scripts/experiments/vldb/stages/slurm/03_cpu_perf.sbatch
```
