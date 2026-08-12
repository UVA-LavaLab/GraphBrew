# exp4 — End-to-End Runtime

**What it measures:** Total time = reorder + kernel. The break-even point
between reorder cost and kernel speedup.

**Output:** `results/vldb_paper/exp4_e2e/e2e_results.json` and
`results/vldb_paper/tables/table_end_to_end.tex`.

## Run

```bash
python3 scripts/experiments/vldb/stages/03_cpu_perf.py \
  --exp 4 \
  --measurement-generation vldb-final-20260808 \
  --skip-build \
  --graph-dir /media/Data/00_GraphDatasets/GraphBrew \
  --artifact-root /media/Data/00_GraphDatasets/GraphBrew/artifacts \
  --threads 16 --cpu-list 0-15
```

## SLURM

```bash
sbatch --export=ALL,EXP=4 scripts/experiments/vldb/stages/slurm/03_cpu_perf.sbatch
```
