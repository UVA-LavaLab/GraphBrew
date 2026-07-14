# exp1 — Cache Performance Analysis

**What it measures:** L1/L2/L3 hit-rate + miss-count for every algorithm
(16 baselines + 10 GraphBrew variants) on each evaluation graph, using the
cache simulator (`bench/bin_sim/pr`).

**Output:** `results/vldb_paper/exp1_cache/cache_results.json`

**CPU speed independent** — runs on the simulator, host wall-clock does not
affect numbers.

## Run

```bash
# Quick smoke
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 1 --preview
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 1 --preview
python3 scripts/experiments/vldb/stages/04_cache_sim.py --exp 1 --preview

# Local 6-graph eval
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 1 --local
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 1 --local
python3 scripts/experiments/vldb/stages/04_cache_sim.py --exp 1 --local
```

## SLURM

```bash
sbatch --export=ALL,EXP=1 scripts/experiments/vldb/stages/slurm/04_cache_sim.sbatch
```

This experiment does **not** need stage 03 (`03_cpu_perf.py`). See
[stages/README.md](../stages/README.md) for the full pipeline.
