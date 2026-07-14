# exp4 — End-to-End Runtime

**What it measures:** Total time = reorder + kernel. The break-even point
between reorder cost and kernel speedup.

**Output:** `results/vldb_paper/exp4_endtoend/endtoend_results.json`

## Run

```bash
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 4 --local
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 4 --local
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 4 --local
```

## SLURM

```bash
sbatch --export=ALL,EXP=4 scripts/experiments/vldb/stages/slurm/03_cpu_perf.sbatch
```
