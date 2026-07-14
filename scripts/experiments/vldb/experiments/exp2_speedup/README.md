# exp2 — Kernel Speedup

**What it measures:** End-to-end wall-clock per (graph, kernel, algorithm).
Speedup is computed by [stage 05](../stages/05_aggregate.py) against the
identity-order baseline.

**Output:** `results/vldb_paper/exp2_speedup/speedup_results.json`

**Real-hardware experiment.** Use a fast, isolated CPU partition for
reproducible numbers (set `--cpus-per-task` and `OMP_NUM_THREADS`).

## Run

```bash
# Smoke test
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 2 --preview
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 2 --preview
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --preview

# Local 6-graph eval
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 2 --local
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 2 --local
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --local

# Single graph
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --graphs com-Orkut
```

## SLURM (fan-out one job per graph)

```bash
# 1. Stage on login node first
python3 scripts/experiments/vldb/stages/01_prep.py --exp 2 --local

# 2. Mappings (once)
sbatch --export=ALL,EXP=2 scripts/experiments/vldb/stages/slurm/02_reorder.sbatch

# 3. Per-graph CPU sweep
for g in cit-Patents com-Orkut hollywood-2009 soc-pokec; do
  sbatch --job-name="gbrew-exp2-$g" \
         --export=ALL,EXP=2,GRAPHS="$g" \
         scripts/experiments/vldb/stages/slurm/03_cpu_perf.sbatch
done
```

For large graphs (`twitter7`, `webbase-2001`, `kron_g500-logn21`,
`uk-2002`, `indochina-2004`):

```bash
sbatch --partition=largemem --mem=512G --time=24:00:00 \
       --export=ALL,EXP=2,GRAPHS=twitter7 \
       scripts/experiments/vldb/stages/slurm/03_cpu_perf.sbatch
```
