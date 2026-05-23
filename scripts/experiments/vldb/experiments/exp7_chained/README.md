# exp7 — Chained Reorderings

**What it measures:** Sequential compositions (e.g. Rabbit → Gorder, Leiden →
RCM, RCM++ chains) compared against single-pass primitives.

**Output:** `results/vldb_paper/exp7_chained/chained_results.json`

## Run

```bash
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 7 --local
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 7 --local
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 7 --local
```

## SLURM

```bash
sbatch --export=ALL,EXP=7 scripts/experiments/vldb/stages/slurm/03_cpu_perf.sbatch
```
