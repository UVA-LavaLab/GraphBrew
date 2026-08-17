[<p align="center"><img src="./docs/figures/logo.svg" width="180"></p>](#graphbrew)

# GraphBrew

GraphBrew is a C++17/OpenMP graph-reordering framework built on the
[GAP Benchmark Suite](https://github.com/sbeamer/gapbs). It provides canonical
graph kernels, multiple reordering baselines, cache simulation, reproducible
experiment orchestration, and infrastructure for developing new
GraphBrew-native orderings.

The primary project surfaces are:

- canonical kernels in `bench/src/`;
- reordering quality and cost in `bench/include/graphbrew/reorder/`;
- reproducible experiment policy in `scripts/graphbrew_experiment.py`;
- reusable Python infrastructure in `scripts/lib/`.

RabbitOrder, Gorder, and Leiden are comparison baselines and diagnostic
anchors. GraphBrew keeps their implementations and costs explicit so new
orderings can be evaluated fairly.

See the [wiki](https://github.com/UVA-LavaLab/GraphBrew/wiki) for detailed
algorithm, CLI, cache-simulation, and architecture documentation.

## Build and Run

RabbitOrder is enabled by default and requires Boost, libnuma, and
google-perftools.

```bash
# Check dependencies through the orchestrator
python3 scripts/graphbrew_experiment.py --check-deps

# Full build
make -j"$(nproc)" all

# Reduced-dependency build
RABBIT_ENABLE=0 make -j"$(nproc)" all

# Small PageRank smoke test
./bench/bin/pr -f scripts/test/data/tiny.el -s -o 0 -n 1
```

Run one ordering with a canonical kernel:

```bash
./bench/bin/pr -f graph.sg -s -o 8:csr -n 3
./bench/bin/bfs -f graph.sg -s -o 12:leiden:flat -n 3
```

Repeated `-o` flags form an ordered composition:

```bash
./bench/bin/pr -f graph.sg -s -o 2 -o 8:csr -n 3
```

## Experiment Harness

Use `scripts/graphbrew_experiment.py` as the public Python entry point. Do not
create one-off experiment runners or duplicate algorithm, graph, benchmark, or
cache registries.

Large graphs and generated mappings belong outside the repository filesystem.
The canonical graph root is `/media/Data/00_GraphDatasets/GraphBrew`, with
large artifacts under its `artifacts/` directory.

### Rapid controlled check

```bash
python3 scripts/graphbrew_experiment.py --vldb 2 --paper-preview \
  --paper-graph-dir /media/Data/00_GraphDatasets/GraphBrew \
  --paper-artifact-root /media/Data/00_GraphDatasets/GraphBrew/artifacts \
  --paper-threads 4 --paper-cpu-list 24-27
```

Rapid runs are for smoke testing, bug detection, and candidate narrowing. They
must not replace full evaluation results.

### Full evaluation

```bash
python3 scripts/graphbrew_experiment.py --vldb \
  --paper-graph-dir /media/Data/00_GraphDatasets/GraphBrew \
  --paper-artifact-root /media/Data/00_GraphDatasets/GraphBrew/artifacts \
  --paper-threads 16 --paper-cpu-list 0-15
```

The full path uses frozen manifests, fixed thread/affinity policy,
pre-generated mappings, repeated trials, verification gates, and complete
provenance. Independent stage runners under
`scripts/experiments/vldb/stages/` are for restartable long runs.

Generic collection remains available through `--full`, `--phase`, `--quick`,
and `--target-graphs`. Use `--dry-run` before broad collection.

## Reordering Algorithms

Use `-o <id[:options]>` to select an ordering.

| ID | Algorithm | Role |
|---:|---|---|
| 0 | ORIGINAL | Input-label baseline |
| 1 | RANDOM | Thread-independent SplitMix64 shuffled control, seed 0 |
| 2 | SORT | Degree sort |
| 3 | HUBSORT | Upstream-style hub sort with non-hub ID preservation |
| 4 | HUBCLUSTER | Upstream-style hub cluster with non-hub ID preservation |
| 5 | DBG | Degree-based grouping |
| 6 | HUBSORTDBG | Compact two-bucket DBG with sorted hubs |
| 7 | HUBCLUSTERDBG | Compact two-bucket DBG with stable hubs |
| 8 | RABBITORDER | Rabbit CSR/Boost baseline |
| 9 | GORDER | Exact `gograph`/`csr` baselines plus deterministic relaxed `fast` |
| 10 | CORDER | Historical 1K baseline; `10:canonical` uses upstream 1 MiB segments |
| 11 | RCM | `11` historical double-pass; `11:mind` single-pass; `11:bnf` CSR-native BNF |
| 12 | GraphBrewOrder | Configurable GraphBrew pipeline |
| 13 | MAP | Load a pre-generated `.lo` mapping |
| 14 | AdaptiveOrder | Load-only offline selector |
| 15 | LeidenOrder | GVE-Leiden communities plus explicit GraphBrew post-layout |
| 16 | GoGraphOrder | M-maximizing core diagnostic; published Rabbit clustering omitted |

Algorithm IDs, option parsing, C++ dispatch, Python canonical names, and
experiment matrices must remain synchronized.

## Canonical Kernels

| Binary | Kernel |
|---|---|
| `pr` | Pull PageRank |
| `pr_spmv` | SpMV PageRank |
| `bfs` | Direction-optimizing BFS |
| `cc` | Afforest connected components |
| `cc_sv` | Shiloach-Vishkin connected components |
| `sssp` | Delta-stepping SSSP |
| `bc` | Betweenness centrality |
| `tc` | Triangle counting |

The default reordering study excludes `tc`; it remains available for explicit
experiments.

## Measurement Contract

New observations are immutable, versioned raw attempts keyed by graph,
algorithm family, exact ordered `-o` specification, benchmark, labeling,
measurement mode, thread policy, mapping
identity, exact permutation fingerprint, and attempt. Failures and timeouts
are retained.

Preprocessing timing is explicit:

- canonical representation build;
- reorder core;
- permutation validation;
- CSR application;
- total preprocessing.

`reorder_time` remains the complete core + validation + application
compatibility metric. Aggregation is performed downstream; raw evidence is
never reduced to a fastest-run record.

The generic Python harness is the official result writer. C++ self-recording is
available only through explicit `-D/--db-dir` or `GRAPHBREW_DB_DIR` use.

The shuffled control is a fixed seeded labeling, not a worst-case claim.

## Adaptive Selection

Benchmark binaries never train models at runtime. They load versioned artifacts
exported offline from measured Tier-0 features.

Legacy non-nested LOGO evaluation is retired and fails closed. Evaluation uses
nested leave-one-topology-out folds with fold-local portfolio selection, model
fitting, and OOD calibration.

## Optional Partitioning

- Compact CSR partitioning and `graph.shard.v1` are separate from normal
  kernels; validate changes with `make check-partition`.

## Project Layout

```text
bench/src/                         canonical kernels
bench/src_sim/                     cache-instrumented kernels
bench/include/graphbrew/reorder/   reordering implementations
bench/include/external/gapbs/      CLI, builder, benchmark lifecycle
scripts/graphbrew_experiment.py    public experiment orchestrator
scripts/lib/                       shared Python policy and pipeline modules
scripts/experiments/               frozen/restartable evaluation runners
scripts/test/                      Python regression suite
wiki/                              detailed documentation
```

Generated binaries, graphs, mappings, and results are ignored. Do not place
large datasets in repository-local `results/graphs/`.

## Verification

```bash
# Authoritative core gate: required binaries, core native tests, include lint,
# and the Python suite
make check

# Reduced-dependency gate
RABBIT_ENABLE=0 make check

# Extended partition/shard integration
make check-partition
```

Edge/GAS validation suites remain explicit and are not part of the primary
core gate.

## Citation and License

GraphBrew integrates ideas and reference implementations from GAPBS,
RabbitOrder, Gorder, Leiden/GVE-Leiden, and related graph-locality work.
Consult the bundled source headers and wiki references for exact attribution.

See [LICENSE](LICENSE) for licensing terms.
