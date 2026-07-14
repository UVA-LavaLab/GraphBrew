# exp8 — Scalability

**What it measures:** Strong scaling: 1, 2, 4, 8, 16, 32 threads at fixed
problem size.

**Output:** `results/vldb_paper/exp8_scalability/scalability_results.json`

## Run

```bash
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 8 --local
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 8 --local
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 8 --local
```

## SLURM (recommend a full node)

```bash
sbatch --cpus-per-task=32 --export=ALL,EXP=8 \
       scripts/experiments/vldb/stages/slurm/03_cpu_perf.sbatch
```
