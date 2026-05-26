# GraphBrew Wiki

GraphBrew is a graph reordering framework. It composes lightweight
primitives — Leiden communities, Rabbit Order, RCM, degree bucketing —
into ten variants that match Gorder's cache quality at a fraction of
the reorder cost.

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
- [Cache-Simulation](Cache-Simulation) — `bench/bin_sim/*` usage
- [ECG-Final-Runs](ECG-Final-Runs) — current ECG/gem5 final-run profiles, charged P-OPT, and supported baselines
- [ECG-Sniper-Runs](ECG-Sniper-Runs) — Sniper backend setup, bounded SIFT full-wrapper validation, DROPLET, and current multithread blocker
- [ECG-Slurm-Runs](ECG-Slurm-Runs) — split ECG/gem5 final runs across UVA Slurm, stage final graphs, and aggregate later
- [Code-Architecture](Code-Architecture) — codebase map
- [VLDB-Experiments](VLDB-Experiments) — reproducing the paper

**Developer**
- [Contributing](Contributing) — adding algorithms and benchmarks
- [Python-Scripts](Python-Scripts) — analysis tools

**Research-only (not part of the VLDB submission)**
- [AdaptiveOrder-ML](AdaptiveOrder-ML) — runtime algorithm selector

## What GraphBrew gives you

| Pipeline stage | Choices | What it controls |
|---|---|---|
| Community detection | Leiden, Rabbit Order | spatial locality |
| Intra-community ordering | BFS, RCM, HubCluster, DBG, Gorder | temporal locality |
| Inter-community arrangement | hierarchical sort, Rabbit on super-graph, RCM, tile | global layout |

Variants ship as flags: `-o 12:leiden`, `-o 12:rabbit`, `-o 12:hrab`,
`-o 12:tqr`, `-o 12:hcache`, `-o 12:rcm`, `-o 12:hubcluster`, `-o 12:streaming`.

See [Reordering-Algorithms](Reordering-Algorithms) for the full list.

## Repository

- Code: https://github.com/UVA-LavaLab/GraphBrew
- Issues: https://github.com/UVA-LavaLab/GraphBrew/issues
- Paper: see `paper/main.tex` (VLDB 2026 submission)
