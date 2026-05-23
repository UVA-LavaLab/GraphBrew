# exp5 — Ablation Study

**What it measures:** Contribution of each axis (SuperGraph × Community ×
Intra × Refinement) by varying one knob at a time.

**Output:** `results/vldb_paper/exp5_ablation/ablation_results.json`

## Run

```bash
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 5 --local
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 5 --local
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 5 --local
```

## SLURM

```bash
sbatch --export=ALL,EXP=5 scripts/experiments/vldb/stages/slurm/03_cpu_perf.sbatch
```
