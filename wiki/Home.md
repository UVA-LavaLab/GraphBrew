# GraphBrew Wiki

GraphBrew is a graph-reordering framework with canonical GAP-style
kernels, multiple baseline reorderers, cache simulation, reproducible
experiment orchestration, and infrastructure for developing new orderings.

## Documentation

**Start here**
- [Getting-Started](Getting-Started) — build, run your first benchmark
- [Reordering-Algorithms](Reordering-Algorithms) — every algorithm explained
- [Running-Benchmarks](Running-Benchmarks) — command-line workflow

**Reference**
- [Command-Line-Reference](Command-Line-Reference) — all flags
- [Supported-Graph-Formats](Supported-Graph-Formats) — `.sg`, `.el`, `.wel`
- [Graph-Benchmarks](Graph-Benchmarks) — graph catalog
- [Troubleshooting](Troubleshooting) — common errors and fixes
- [FAQ](FAQ) — short answers to common questions

**Deep dives**
- [GraphBrewOrder](GraphBrewOrder) — the composable pipeline
- [All-Kernel-Low-Reuse-Selector](All-Kernel-Low-Reuse-Selector) — figures, frozen rule, and per-graph evidence
- [Cache-Simulation](Cache-Simulation) — `bench/bin_sim/*` usage
- [Partitioning-and-Shards](Partitioning-and-Shards) — compact CSR packages
- [Code-Architecture](Code-Architecture) — codebase map
- [Reproducible-Experiments](Reproducible-Experiments) — frozen study reproduction

**Developer**
- [Contributing](Contributing) — adding algorithms and benchmarks
- [Python-Scripts](Python-Scripts) — analysis tools

**Selection**
- [AdaptiveOrder-ML](AdaptiveOrder-ML) — offline-model runtime selector
- [All-Kernel-Low-Reuse-Selector](All-Kernel-Low-Reuse-Selector) — validated reuse-1/2 runtime policy

## What GraphBrew gives you

| Pipeline stage | Choices | What it controls |
|---|---|---|
| Partitioner | Leiden, Rabbit Order | vertex groups |
| Block layout | identity, size/degree sort, Rabbit/RCM/tile super-graph | global placement |
| Vertex layout | BFS, RCM, HubCluster, DBG, Gorder | within-block locality |

Variants ship as flags: `-o 12:leiden`, `-o 12:rabbit`, `-o 12:hrab`,
`-o 12:tqr`, `-o 12:hcache`, `-o 12:hrab:bfs_intra`,
`-o 12:hubcluster`, `-o 12:streaming`.

See [Reordering-Algorithms](Reordering-Algorithms) for the full list.

## Repository

- Code: https://github.com/UVA-LavaLab/GraphBrew
- Issues: https://github.com/UVA-LavaLab/GraphBrew/issues
- Reproducibility: use `graphbrew_experiment.py` and the frozen-study guide
