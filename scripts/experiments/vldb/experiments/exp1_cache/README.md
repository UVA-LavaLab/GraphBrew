# exp1 — Cache Performance Analysis

**What it measures:** L1/L2/L3 hit-rate + miss-count for the controlled
ten-technique matrix (Shuffled, DBG, both Rabbit implementations, Gorder,
RCM, Leiden, HRAB-BFS, HRAB-RCM, and TQR) on each evaluation graph, using
fixed-work PageRank in `bench/bin_sim/pr`.

**Output:** `results/vldb_paper/exp1_cache/cache_results.json`

**CPU speed independent** — runs on the simulator, host wall-clock does not
affect numbers.

## Final paper policy

- Representative graphs: `cit-Patents`, `com-Orkut`, `hollywood-2009`,
  `USA-road-d.USA`
- Five fixed PR iterations and one cold process per cell
- Every-access `ultrafast` CLOCK-style reference-bit/second-chance,
  non-inclusive simulation
- LLC capacities: 2, 8, 22, 32, and 64 MiB; the 22 MiB point uses 11 ways
- Primary derived metric:
  `H = total_accesses + l1_misses + l2_misses + l3_misses`, reported as
  `H_SHUFFLED / H_ordering`

The model excludes hardware prefetching, coherence, dirty writeback traffic,
and multicore contention. It supports relative capacity/locality claims, not
absolute hardware-cycle or DRAM-traffic claims.

## Run

```bash
# Quick smoke
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 1 --preview
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 1 --preview
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --preview --verify-gate
python3 scripts/experiments/vldb/stages/04_cache_sim.py --exp 1 --preview

# Local 6-graph eval
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 1 --local
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 1 --local
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --local --verify-gate
python3 scripts/experiments/vldb/stages/04_cache_sim.py --exp 1 --local

# Final representative cohort (the graph/capacity/mode matrix is the SSOT default)
python3 scripts/experiments/vldb/stages/04_cache_sim.py --exp 1 \
  --graph-dir /media/Data/00_GraphDatasets/GraphBrew \
  --artifact-root /media/Data/00_GraphDatasets/GraphBrew/artifacts \
  --threads 16 --cpu-list 0-15
```

Use `--graphs`, `--cache-sizes-kib`, or `--cache-mode` only for an explicitly
separate exploratory cohort.

## SLURM

```bash
sbatch --export=ALL,EXP=1 scripts/experiments/vldb/stages/slurm/04_cache_sim.sbatch
```

This experiment does not need the timed Stage-03 sweep, but it does require
the host-local `03_cpu_perf.py --verify-gate` artifact. See
[stages/README.md](../../stages/README.md) for the full pipeline.
