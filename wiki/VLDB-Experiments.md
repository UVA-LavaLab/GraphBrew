# VLDB 2026 Experiment Guide

Reproduces every figure and table in the GraphBrew paper from an empty
`results/` directory. The runner auto-builds binaries, downloads graphs
from SuiteSparse where possible, converts them to `.sg`, runs the eight
experiments, and regenerates the figures and LaTeX tables.

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Prerequisites](#2-prerequisites)
3. [Experiment Overview](#3-experiment-overview)
4. [Running Experiments](#4-running-experiments)
5. [Generated Outputs](#5-generated-outputs)
6. [Configuration Reference](#6-configuration-reference)
7. [Troubleshooting](#7-troubleshooting)
8. [SLURM Runbook — UVA Cluster](#8-slurm-runbook--uva-cluster) — smoke test first, then full paper eval

---

## 1. Quick Start

```bash
# Full paper run (256 GB+ RAM, includes twitter7 + webbase-2001):
python3 scripts/experiments/vldb_paper_experiments.py --all

# 64 GB RAM machine (11 auto-downloadable graphs, no >1B-edge graphs):
python3 scripts/experiments/vldb_paper_experiments.py --all --64gb

# Local-machine smoke test (6 graphs ≤ 117M edges, fits 64 GB easily):
python3 scripts/experiments/vldb_paper_experiments.py --all --local

# Preview (2 small graphs, 1 trial — ~5 min validation):
python3 scripts/experiments/vldb_paper_experiments.py --all --preview

# Dry run (print commands without executing):
python3 scripts/experiments/vldb_paper_experiments.py --all --dry-run

# Regenerate figures from existing results:
python3 scripts/experiments/vldb_paper_experiments.py --figures-only
```

The auto-setup phase will:
1. **Build** standard and cache-simulation binaries via `make`
2. **Download** 9 of 11 evaluation graphs from SuiteSparse (2 require manual download — see below)
3. **Convert** downloaded `.mtx` files to `.sg` format

To skip auto-setup (if binaries and graphs are already in place):

```bash
# Uses results/graphs/ by default:
python3 scripts/experiments/vldb_paper_experiments.py --all --skip-setup

# Or specify a custom graph directory:
python3 scripts/experiments/vldb_paper_experiments.py --all --skip-setup \
    --graph-dir /path/to/graphs
```

---

## 2. Prerequisites

### System Requirements

- Linux x86-64 with GCC ≥ 7 (tested on Ubuntu 22.04 / 24.04)
- ≥ 16 GB RAM for preview; 32–64 GB for `--64gb` graph set; 64 GB+ for full evaluation (webbase-2001, twitter7)
- Python ≥ 3.8

### Automatic Steps (handled by `--all`)

The script calls `make -j$(nproc)` and `make all-sim -j$(nproc)` automatically.
If you prefer to build manually:

```bash
make all RABBIT_ENABLE=1      # standard benchmark binaries
make all-sim                   # cache simulation binaries
pip install matplotlib numpy   # optional: figure generation
```

### Manual-Download Graphs (2 of 11)

Nine evaluation graphs are downloaded automatically from SuiteSparse. Two
require manual preparation:

#### wikipedia\_link\_en

Source: [KONECT — Wikipedia link (en)](http://konect.cc/networks/wikipedia_link_en/)

Download the dataset, extract it, and convert the edge list to a file the
converter can read (tab-separated edge list → `.el`):

```bash
mkdir -p results/graphs/wikipedia_link_en
# Download from KONECT, extract, and rename to .el
# Place the edge-list file at:
#   results/graphs/wikipedia_link_en/wikipedia_link_en.el
```

#### Gong-gplus

Source: [Duke University — Google+ Social Networks](https://people.duke.edu/~zg70/gplus.html)
([Google Drive link](https://drive.google.com/file/d/1HF8Q2N_hxsaQ26MarKYxZEQhqI66qAxV/view))

The dataset contains 4 temporal snapshots. To reconstruct snapshot 4
(28.9M vertices, 463M edges), keep all edges with TimeID 0–3:

```bash
mkdir -p results/graphs/Gong-gplus
# 1. Download from the Google Drive link above
# 2. Extract and keep all directed social links (TimeID 0–3)
# 3. Strip the TimeID column to produce a two-column edge list
# 4. Place as: results/graphs/Gong-gplus/Gong-gplus.el
```

> **Note:** The auto-setup will print clear instructions for any missing
> manual-download graphs and proceed with the available ones.

---

## 3. Experiment Overview

The paper's evaluation consists of 6 subsections, each mapped to specific
experiments in the runner:

| § | Paper Subsection | Experiment | What It Measures |
|---|-----------------|------------|------------------|
| 4.2 | Cache Performance | Exp 1 | Cache miss rates across cache sizes (PR, all reorderings) |
| 4.3 | Kernel Speedup | Exp 2 | Algorithm execution time normalized to Original (7 benchmarks) |
| 4.4 | Overhead & E2E | Exp 3+4 | Reorder preprocessing time + amortization analysis |
| 4.5 | Sensitivity & Composability | Exp 5+6+7 | Graph-type sensitivity, layer ablation, chained orderings |
| 4.6 | Scalability | Exp 8 | Thread scaling of reorder step (1–32 threads) |

### Algorithms Evaluated

**Algorithms Evaluated (13):** Original, Random, SORT, HubSort, HubCluster, DBG,
HubSortDBG, HubClusterDBG, RabbitOrder (CSR), RabbitOrder (Boost), Gorder, RCM, GoGraph

**GraphBrew Variants (10):** Leiden, Rabbit, HubCluster, HRAB, TQR, HCache, Streaming,
Rabbit-DBG, Rabbit-HubCluster, RCM

**Chained Orderings (5):** GB-Leiden→DBG, GB-Leiden→HubCluster,
GB-HRAB→DBG, GB-Leiden→GoGraph, RabbitOrder→DBG

### Benchmark Algorithms (7)

BFS, PR (PageRank), PR-SpMV, SSSP, CC (Afforest), CC-SV, BC

### Evaluation Graphs (11)

| Graph | Vertices (M) | Edges (M) | Type |
|-------|------------:|----------:|------|
| cit-Patents | 6.01 | 16.52 | Citation |
| soc-pokec | 1.63 | 30.62 | Social |
| USA-road-d.USA | 23.95 | 58.33 | Road |
| soc-LiveJournal1 | 4.85 | 68.99 | Social |
| delaunay_n24 | 16.78 | 100.66 | Mesh |
| hollywood-2009 | 1.14 | 113.89 | Collaboration |
| com-Orkut | 3.07 | 117.19 | Social |
| wikipedia_link_en | 12.15 | 378.14 | Content |
| Gong-gplus | 28.94 | 462.99 | Social |
| webbase-2001 | 118.14 | 1,019.90 | Web |
| twitter7 | 61.79 | 1,468.36 | Social |

---

## 4. Running Experiments

### Full Evaluation

```bash
# Run all 8 experiments (auto-setup included):
python3 scripts/experiments/vldb_paper_experiments.py --all

# Run all experiments with graphs in a specific directory:
python3 scripts/experiments/vldb_paper_experiments.py \
    --all --skip-setup --graph-dir /data/graphs

# Run specific experiments (e.g., cache + speedup only):
python3 scripts/experiments/vldb_paper_experiments.py \
    --exp 1 2

# Skip figure generation:
python3 scripts/experiments/vldb_paper_experiments.py \
    --all --no-figures
```

### Preview Mode

For fast validation before the full run:

```bash
python3 scripts/experiments/vldb_paper_experiments.py --all --preview
```

Preview uses: 2 small graphs, 1 trial, 2 benchmarks (PR, BFS), 300s timeout.

### Custom Graph Set

```bash
python3 scripts/experiments/vldb_paper_experiments.py \
    --all --graphs cit-Patents soc-pokec
```

### Figure Generation Only

```bash
# From real experiment data:
python3 scripts/experiments/vldb_paper_experiments.py --figures-only

# With sample/placeholder data (for layout preview):
python3 scripts/experiments/vldb_generate_figures.py --sample-data
```

---

## 5. Generated Outputs

```
results/vldb_paper/
├── MANIFEST.json              # Reproducibility metadata (git hash, config, timing)
├── exp1_cache/                # Cache simulation results (JSON)
│                              #   Per-record fields: timing (average_time, reorder_time, …)
│                              #   + L1/L2/L3 cache metrics (l1_hits, l1_misses, l1_hit_rate, …)
├── exp2_speedup/              # Kernel speedup results (JSON)
├── exp3_overhead/             # Reorder overhead results (JSON, .sg input with .el fallback)
├── exp4_e2e/                  # End-to-end derived data
├── exp5_ablation/             # Ablation study results (JSON)
├── exp6_sensitivity/          # Graph-type sensitivity metadata
├── exp7_chained/              # Chained ordering results (JSON)
├── exp8_scalability/          # Thread scaling results (JSON, .sg input with .el fallback)
├── figures/                   # Generated PNG / PDF figures
│   ├── fig1_cache_performance.png
│   ├── fig2_kernel_speedup.png
│   ├── fig3_reorder_overhead.png
│   └── fig_h2h_pareto.{png,pdf}  # head-to-head vs Gorder + Rabbit Pareto
└── tables/                    # Generated LaTeX table snippets
    ├── table_variants.tex
    ├── table_ablation.tex
    ├── table_sensitivity.tex
    ├── table_chained.tex
    ├── table_h2h_per_graph.tex  # paper headline comparison
    └── table_h2h_summary.tex    # cross-graph geo-mean + wins
```

Figures and tables are also mirrored to the paper's `dataCharts/`
directory so `main.tex` can `\input{dataCharts/tables/...}` and
`\includegraphics{dataCharts/speedup/h2h_pareto}` directly without an
extra copy step. See `comparison_vs_baselines()` in
`scripts/experiments/vldb_generate_figures.py` for the head-to-head
artifact generation.

---

## 6. Configuration Reference

All experiment parameters are defined in
`scripts/experiments/vldb_config.py`:

| Parameter | Full | Preview |
|-----------|------|---------|
| Trials | 3 | 1 |
| Benchmarks | 7 (bfs, pr, pr_spmv, sssp, cc, cc_sv, bc) | 2 (pr, bfs) |
| Graphs | 11 | 2 |
| Timeout (per command) | 3600s | 300s |
| Thread counts (scaling) | 1, 2, 4, 8, 16, 32 | 1, 2, 4, 8, 16, 32 |

### CLI Flags

| Flag | Description |
|------|-------------|
| `--all` | Run all 8 experiments |
| `--exp N [N ...]` | Run specific experiment(s) by number (1-8) |
| `--preview` | 2 small graphs, 1 trial, 2 benchmarks (validation) |
| `--local` | 6 graphs ≤117M edges (cit-Patents → com-Orkut, fits 64 GB) |
| `--64gb` | 11 auto-downloadable graphs (no >1B-edge graphs) |
| `--dry-run` | Print commands without executing |
| `--graph-dir PATH` | Directory containing graph files (default: `results/graphs` with `--skip-setup`) |
| `--graphs NAME [...]` | Override graph list by name |
| `--skip-setup` | Skip the auto-setup phase (build, download, convert) |
| `--skip-download` | Skip graph download but still build + convert |
| `--no-figures` | Skip automatic figure generation |
| `--figures-only` | Generate figures from existing results (no experiments) |

### 64 GB Graph Set

For machines with 32–64 GB RAM, use `--64gb` to select an alternative set of 11
auto-downloadable graphs that avoids twitter7 and webbase-2001 (both >1B edges,
require >64 GB RAM). This set adds as-Skitter, kron_g500-logn21, indochina-2004,
and uk-2002 for type diversity:

```bash
python3 scripts/experiments/vldb_paper_experiments.py --all --64gb
```

---

## 7. Troubleshooting

### Common Issues

**"Binary not found"** — The script builds binaries automatically.
If auto-build fails, run `make all RABBIT_ENABLE=1 && make all-sim` manually.

**"Graph file not found"** — Either let auto-setup download the graphs, or
ensure `--graph-dir` points to a directory with `.sg` files matching the graph
names in the config. Both flat layout (`cit-Patents.sg`) and nested layout
(`cit-Patents/cit-Patents.sg`) are supported. Experiments 3 and 8 try `.sg`
first and fall back to `.el` automatically.

**"Conversion failed" for SuiteSparse graphs** — Some SuiteSparse archives
contain auxiliary `.mtx` files (e.g., `*_nodename.mtx`) alongside the actual
graph matrix. The converter prefers files named exactly `{graph_name}.mtx`.
If conversion fails, check that the correct `.mtx` file exists in the nested
directory (`results/graphs/{name}/{name}/{name}.mtx`).

**Graphs that need manual download** — `wikipedia_link_en` (KONECT) and
`Gong-gplus` (Google Drive) cannot be auto-downloaded. See
[Prerequisites §2](#2-prerequisites) for download instructions. The script will
skip these graphs and proceed with the rest.

**"matplotlib not available"** — Install with `pip install matplotlib numpy`.
Tables will still be generated without matplotlib.

**"Timeout"** — Large graphs (twitter7, webbase-2001) may need longer timeouts.
Edit `TIMEOUT_FULL` in `vldb_config.py`.

### Extending

To add a new graph or algorithm, edit `scripts/experiments/vldb_config.py`:
- `EVAL_GRAPHS` — add graph metadata
- `BASELINE_ALGORITHMS` — add algorithm ID and name
- `GRAPHBREW_VARIANTS` — add variant string
- `CHAINED_ORDERINGS` — add (name, flags) tuple

---

### Result JSON Schema

All experiment JSON files share a common set of timing fields extracted by
`parse_timing()`: `trial_time`, `reorder_time`, `average_time`,
`preprocessing_time`, `total_time`, `topology_analysis_time`, `read_time`,
`relabel_map_time`.

Experiment 1 additionally includes per-cache-level metrics extracted by
`parse_cache_sim()`: `l1_hits`, `l1_misses`, `l1_hit_rate`, `l2_hits`,
`l2_misses`, `l2_hit_rate`, `l3_hits`, `l3_misses`, `l3_hit_rate`,
`total_accesses`, `memory_accesses`, `overall_hit_rate`.

LaTeX tables (`table_ablation.tex`, `table_sensitivity.tex`,
`table_chained.tex`) are populated from the JSON data automatically;
fields that have no data yet show `\emph{TBD}`.

---

## 8. SLURM Runbook — UVA Cluster

Two-phase recipe: (a) a 30-minute **smoke test** that proves the harness,
binaries, and ResultsStore work on the cluster, then (b) the **full
evaluation** parallelised over per-(experiment, graph) jobs.

### 8.1 One-time UVA setup

UVA Research Computing's Slurm reference:
<https://www.rc.virginia.edu/userinfo/hpc/slurm/>

```bash
# Clone + checkout
git clone https://github.com/<you>/GraphBrew.git
cd GraphBrew

# Inspect available partitions and your allocation accounts
qlist                          # partition list (UVA convenience wrapper)
qlimits                        # per-partition core/memory/time caps
sacctmgr -p show user $USER    # accounts you can charge
module avail gcc               # confirm gcc module name on the cluster
module avail miniforge         # confirm python/conda module name

# Edit scripts/experiments/vldb_slurm.sbatch:
#   - --account=YOUR_UVA_ALLOC
#   - --partition=... (standard for single-node threaded jobs is the default)
#   - module load gcc miniforge   # change names if `module avail` shows different
```

> **Why standard partition?** UVA's `standard` is the single-node
> serial/threaded queue, which is exactly what our 32-core OpenMP runs
> need. Use `parallel` only for true MPI multi-node work.

> **Data safety reminder:** every job writes per-cell results via
> `ResultsStore` with atomic `tmp + rename`. If a job times out you can
> resubmit it verbatim — already-completed cells are skipped.

> **#SBATCH gotcha (UVA-confirmed):** SLURM directives do **not** expand
> shell variables. Lines like `#SBATCH --output=...-${GRAPH}.out` produce
> filenames with literal `${GRAPH}`. Use only `%x` (job-name) and `%j`
> (jobid) in `--output=`, and pass `--job-name=gbrew-exp${exp}-${g}` on
> the `sbatch` command line so the EXP/GRAPH appear in the log filename
> via `%x`. The template and examples below already do this.

### 8.1.5 Stage graphs on the login node (REQUIRED — compute nodes have no internet)

UVA Rivanna compute nodes do **not** have outbound internet, so SLURM
jobs cannot themselves fetch graphs from SuiteSparse. The
`vldb_slurm.sbatch` template therefore runs with `--skip-setup
--skip-download` and aborts with a clear error if the `.sg` file is
missing. Stage every graph **once** on the login node before submitting:

```bash
# Stage all 64GB graphs at once (builds binaries, downloads, converts to .sg).
# Each graph is small (~100MB-2GB .sg); total ~10GB; ~20-40 min on the
# login node depending on SuiteSparse mirror speed.
for g in cit-Patents soc-pokec hollywood-2009 soc-LiveJournal1 \
         com-Orkut USA-road-d.USA kron_g500-logn21 \
         indochina-2004 uk-2002; do
  python3 scripts/experiments/vldb_paper_experiments.py \
      --exp 2 --graphs "$g" --64gb --no-figures
done

# Verify all .sg files exist before sbatch:
for g in cit-Patents soc-pokec hollywood-2009 soc-LiveJournal1 \
         com-Orkut USA-road-d.USA kron_g500-logn21 \
         indochina-2004 uk-2002; do
  ls -la "results/graphs/$g/$g.sg" 2>/dev/null || echo "MISSING: $g"
done
```

The login-node `--exp 2` invocation does double duty: it triggers
auto-setup (build + download + .el → .sg conversion) *and* runs the
experiment for that one graph. Because `ResultsStore` saves cells
atomically, those results carry into the later SLURM run for free.

**Big-graph addendum (twitter7, webbase-2001):** these are not on
SuiteSparse and need manual download from KONECT/Google-Drive — see
`VLDB_GRAPH_SOURCES` in [scripts/experiments/vldb_config.py](../scripts/experiments/vldb_config.py).
Place the `.el` under `results/graphs/<name>/<name>.el` on the login
node and the SLURM job's converter step will pick it up.

**Escape hatch:** if your cluster *does* allow outbound HTTPS from
compute nodes, set `AUTO_SETUP=1` in `--export` to let the SLURM job
download itself (not recommended on UVA standard partition).

### 8.2 Phase A — SLURM smoke test (30 min, one graph, one experiment)

The goal here is to validate environment / modules / scratch I/O / SLURM
account *before* spending real allocation on the full sweep.

```bash
# Submit ONE job: smallest experiment × smallest graph.
# Pass --job-name with EXP/GRAPH baked in so the log filename is descriptive.
sbatch --time=00:30:00 \
       --job-name=gbrew-exp2-cit-Patents \
       --export=ALL,EXP=2,GRAPH=cit-Patents,GRAPHSET=local \
       scripts/experiments/vldb_slurm.sbatch

# Watch it land
squeue -u $USER
tail -f results/slurm_logs/gbrew-exp2-cit-Patents-*.out
```

**Success criteria** — check after job completes:

```bash
# 1. Did it write the JSON?
ls -la results/vldb_paper/exp2_speedup/speedup_results.json

# 2. Are all cells valid (60 rows expected for --preview-ish single-graph)?
python3 -c "
import json
d = json.load(open('results/vldb_paper/exp2_speedup/speedup_results.json'))
valid = [r for r in d if r.get('average_time') is not None]
compose = [r for r in d if 'compose' in str(r.get('algo_id') or '')]
print(f'rows={len(d)} valid={len(valid)}/{len(d)} compose={len(compose)}')
assert len(valid) == len(d), 'some cells have no timing — check logs'
assert len(compose) > 0, 'compose configs did not run — parser failure?'
print('SMOKE TEST PASSED')
"

# 3. Test resume — resubmit; should finish in <1 min thanks to ResultsStore
sbatch --time=00:10:00 \
       --job-name=gbrew-exp2-cit-Patents-resume \
       --export=ALL,EXP=2,GRAPH=cit-Patents,GRAPHSET=local \
       scripts/experiments/vldb_slurm.sbatch
# Look for "Resume: loaded N existing results" in the new log.
```

UVA-specific health checks (the canonical commands from
<https://www.rc.virginia.edu/userinfo/hpc/slurm/#displaying-job-status>):

```bash
squeue -u $USER                              # is it queued / running?
scontrol show job <jobid>                    # detailed state
seff <jobid>                                 # CPU + memory efficiency after completion
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed
```

If any of the three checks fails, **stop and fix before Phase B**.
Common gotchas:

| Symptom | Cause | Fix |
|---|---|---|
| `module: command not found` | wrong module env on partition | check `module avail` and edit `vldb_slurm.sbatch` |
| `gcc/12: Unable to locate` | module name differs on cluster | run `module avail gcc` and update the `module load` line |
| `bench/bin/converter: not found` | build failed silently | run `make -j$SLURM_CPUS_PER_TASK pr bfs cc sssp bc tc converter` manually first |
| `Permission denied` on `results/slurm_logs/` | log dir doesn't exist | `mkdir -p results/slurm_logs` before sbatch |
| `Invalid account` | wrong `--account=` | `sacctmgr -p show user $USER` to list yours |
| Log file literally named `*-exp${EXP}-${GRAPH}.out` | shell vars don't expand in `#SBATCH` | use `--job-name=gbrew-exp${exp}-${g}` on the sbatch command line; the template's `--output=%x-%j.out` then bakes EXP/GRAPH in via `%x` |
| All cells valid timing but 0 compose rows | old `vldb_config.py` deployed | `git pull` on the cluster |

### 8.3 Phase B — Full paper evaluation (parallel fan-out)

After smoke test passes, fan out the **priority A** experiments
(exp2 kernel speedup, exp3 reorder amortisation, exp8 thread
scalability) across 9 graphs from the 64-GB set. That's 27 jobs,
each runs independently, each ≤ 4h wall.

```bash
# Skip the smallest graphs you already smoked + the manual-download ones
GRAPHS_64GB=(
  cit-Patents soc-pokec USA-road-d.USA soc-LiveJournal1
  delaunay_n24 hollywood-2009 com-Orkut
  kron_g500-logn21 indochina-2004 uk-2002
)

# Three priority experiments — these together produce the paper's
# headline table (kernel speedup), amortisation column, and scalability
# figure. Each iteration pre-sets --job-name so the EXP/GRAPH show up
# in squeue and in the log filename via the %x token.
for g in "${GRAPHS_64GB[@]}"; do
  for exp in 2 3 8; do
    sbatch --time=04:00:00 \
           --job-name=gbrew-exp${exp}-${g} \
           --export=ALL,EXP=$exp,GRAPH=$g,GRAPHSET=64gb \
           scripts/experiments/vldb_slurm.sbatch
  done
done

# Check submission count (should be 30 jobs)
squeue -u $USER -h | wc -l
```

> **Alternative — Job Arrays (UVA-recommended for large fan-outs).**
> SLURM job arrays (`--array=1-N`) submit hundreds of tasks under one
> jobid, and cancel/requeue is per-task. They require an `options.txt`
> with one `(EXP,GRAPH)` per line and a small wrapper around the
> template. See
> <https://www.rc.virginia.edu/userinfo/hpc/slurm/#using-files-with-job-arrays>.
> For 30 jobs the simple `for` loop above is fine; switch to arrays if
> you ever scale to hundreds of cells.

**Re-submit timeouts**. SLURM returns exit code 124 for `timeout`;
just rerun the exact same `sbatch` line — `ResultsStore` picks up
where it left off. Find timeouts with:

```bash
sacct -u $USER --format=JobID,JobName,State,ExitCode,Elapsed --state=TIMEOUT
```

### 8.4 What NOT to run (or run only if budget allows)

| Experiment | Why skip | If you have time |
|---|---|---|
| **exp1** cache-sim | 3+ days on 64gb (cycle-accurate sim per cell). v5 §17 already gives the cache-mechanism story. | Run only on 3 representative graphs: `cit-Patents`, `hollywood-2009`, `com-Orkut`. |
| **exp4** end-to-end | Derivable from exp2 + exp3 JSON by `vldb_generate_figures.py` — no new measurement needed. | (already auto-computed) |
| **exp5** ablation | Mostly redundant with v5 §15 / §18 / §19 ablations done locally. | Run on `cit-Patents` + `hollywood-2009` only. |
| **exp6** sensitivity | Already covered by exp2's per-graph breakdown. | (skip) |
| **exp7** chained | Small (210 cells). Adds the chained-ordering comparison. | Run if reviewers may ask about chains. |

### 8.5 Big-graph addendum (256 GB nodes)

Twitter7 (1.5B edges) and webbase-2001 (1B edges) are the most
impactful generalization checks but only fit on 256-GB partitions.

```bash
# Submit to a high-memory partition with extra wall time.
# Check `qlist` for the exact high-mem partition name on your cluster
# (commonly `largemem` on UVA Rivanna).
sbatch --partition=largemem --mem=256G --time=24:00:00 \
       --job-name=gbrew-exp2-twitter7 \
       --export=ALL,EXP=2,GRAPH=twitter7,GRAPHSET=full \
       scripts/experiments/vldb_slurm.sbatch

sbatch --partition=largemem --mem=256G --time=24:00:00 \
       --job-name=gbrew-exp2-webbase-2001 \
       --export=ALL,EXP=2,GRAPH=webbase-2001,GRAPHSET=full \
       scripts/experiments/vldb_slurm.sbatch
```

These two graphs require **manual download** (KONECT / Google Drive
links in `VLDB_GRAPH_SOURCES`). Stage them under
`results/graphs/<name>/<name>.el` before submitting, then add
`--skip-download` so the harness doesn't try to fetch.

### 8.6 Aggregation & figure generation

Once all jobs finish (or even mid-run), pull JSONs locally and
generate figures. Because every job writes to the same
`results/vldb_paper/exp{N}_*/...json` paths, the cluster filesystem
already has the merged dataset.

```bash
# On the cluster (or rsync to local)
python3 scripts/experiments/vldb_paper_experiments.py --figures-only --64gb

# Outputs:
ls results/vldb_paper/figures/
ls results/vldb_paper/tables/
```

For multi-machine merges (some jobs on UVA, others elsewhere), each
ResultsStore JSON is a flat list of result dicts — concat them with
`jq -s '.[0]+.[1]'` or a 3-line Python script before running
`--figures-only`.

### 8.7 Time budget at a glance

With 32-core nodes, 1 trial, all eight COMPOSE configs added to exp2/exp8:

| Phase | Cells | Parallel jobs | Wall (worst-job) | Total alloc time |
|---|---|---|---|---|
| 8.2 Smoke (1 graph × exp2) | 60 | 1 | ~30 min | 30 min |
| 8.3 Priority A (10 graphs × exp 2,3,8) | ~5,000 | 30 | ≤ 4 h | ~120 CPU-hr |
| 8.5 Big graphs (twitter, webbase × exp2) | ~420 | 2 | ≤ 24 h | ~48 CPU-hr |
| 8.4 (optional) exp1 cache-sim on 3 graphs | ~840 | 3 | ≤ 24 h | ~72 CPU-hr |

**Total wall ≤ 1 day** thanks to parallelism. **Total alloc ≈ 270 CPU-h**
if you include the optional cache-sim.

---

*See also: [[GraphBrewOrder]], [[Running-Benchmarks]], [[Command-Line-Reference]], [[Cache-Simulation]], [[Python-Scripts]]*
