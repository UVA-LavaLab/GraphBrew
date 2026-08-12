# exp8 — Scalability

**What it measures:** Representative mapping-generation thread scaling at
1, 2, 4, 8, and 16 physical cores. Schedule-sensitive algorithms may produce
different but comparable mappings across thread counts; this is not a
fixed-permutation strong-scaling claim.

**Output:** `results/vldb_paper/exp8_scalability/scalability_results.json`

## Core run

```bash
python3 scripts/experiments/vldb/stages/03_cpu_perf.py \
  --exp 8 \
  --measurement-generation vldb-final-20260808 \
  --skip-build --reorder-timeout 3600 \
  --graph-dir /media/Data/00_GraphDatasets/GraphBrew \
  --artifact-root /media/Data/00_GraphDatasets/GraphBrew/artifacts \
  --threads 16 --cpu-list 0-15
```

The Exp8 SSOT default is the three-graph citation/road/social core with seven
algorithms. Absolute 16-thread time is the primary result; self-relative
speedup over one thread is secondary.

## Twitter addendum

```bash
python3 scripts/experiments/vldb/stages/03_cpu_perf.py \
  --exp 8 --graphs twitter7 \
  --algorithms 8:csr 8:boost 12:rabbit \
    12:rabbit:compose:sg_super_rabbit:comm_degree_desc:intra_hubsort \
  --measurement-generation vldb-final-20260808 \
  --skip-build --reorder-timeout 3600 \
  --graph-dir /media/Data/00_GraphDatasets/GraphBrew \
  --artifact-root /media/Data/00_GraphDatasets/GraphBrew/artifacts \
  --threads 16 --cpu-list 0-15
```

The reduced addendum avoids implying that the serial-core Leiden and Gorder
paths were omitted after measurement. On Twitter, the sampled edge-span proxy
changes by about 38--41% from one to 16 threads for standalone RabbitOrder,
8.6% for GraphBrew Rabbit, and 0.2% for the stable-block variant. Treat these
as schedule-sensitive mapping-generation throughput results, not
fixed-permutation scaling.

## SLURM (recommend a full node)

```bash
sbatch --cpus-per-task=16 --export=ALL,EXP=8 \
       scripts/experiments/vldb/stages/slurm/03_cpu_perf.sbatch
```
