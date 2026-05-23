# exp6 — Graph Sensitivity

**What it measures:** Same recipes across graph families
(social / road / kron / web) to characterize which workloads benefit.

**Output:** `results/vldb_paper/exp6_sensitivity/sensitivity_results.json`

## Run

```bash
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 6 --local
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 6 --local
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 6 --local
```

## SLURM

```bash
sbatch --export=ALL,EXP=6 scripts/experiments/vldb/stages/slurm/03_cpu_perf.sbatch
```
