# GraphBrew Research References

> **Purpose:** Authoritative paper references for every ordering algorithm, caching policy, and benchmark kernel in GraphBrew. Agents and contributors must use these files as ground truth to stay aligned with the original papers. Each file includes: verified citations, official GitHub repos, paper performance data, pseudo-code from the papers/implementation, and why faithful implementation matters.

> **This folder is in `.gitignore` and will not be pushed.**

---

## Why This Folder Exists

GraphBrew implements algorithms from **9+ academic papers** and validates them against each other. The ECG paper (Section A) specifically tests that our GRASP and P-OPT implementations match the original papers with **quantitative tolerances** (3%, 5%, 10%). If any baseline deviates from its paper, the validation is scientifically unsound and published comparisons become invalid.

This folder ensures that any agent or contributor working on GraphBrew can:
1. **Verify** implementation correctness against the original paper's algorithm
2. **Check** performance against published numbers from official repos
3. **Understand** why specific parameters (bucket counts, RRPV values, window sizes) must not change
4. **Cross-reference** the official source repos for each method

---

## Official GitHub Repositories

| Method | Repo | License |
|--------|------|---------|
| GRASP | [faldupriyank/grasp](https://github.com/faldupriyank/grasp) | Apache-2.0 |
| DBG / HubSort / HubCluster | [faldupriyank/dbg](https://github.com/faldupriyank/dbg) | Apache-2.0 |
| IISWC'18 Packing Factor | [CMUAbstract/Graph-Reordering-IISWC18](https://github.com/CMUAbstract/Graph-Reordering-IISWC18) | MIT |
| P-OPT | [CMUAbstract/POPT-CacheSim-HPCA21](https://github.com/CMUAbstract/POPT-CacheSim-HPCA21) | MIT |
| RabbitOrder | [araij/rabbit_order](https://github.com/araij/rabbit_order) | Custom |
| GOrder | [datourat/Gorder](https://github.com/datourat/Gorder) | MIT |
| GoGraph | [iDC-NEU/GoGraph](https://github.com/iDC-NEU/GoGraph) | — |
| GVE-Leiden | [puzzlef/leiden-communities-openmp](https://github.com/puzzlef/leiden-communities-openmp) | MIT |
| GAP Benchmark Suite | [sbeamer/gapbs](https://github.com/sbeamer/gapbs) | BSD-3 |

---

## Reordering Algorithms

| Algo ID | Name | Paper | Venue | Year | File |
|---------|------|-------|-------|------|------|
| 0 | ORIGINAL | *(baseline)* | — | — | — |
| 1 | RANDOM | *(baseline)* | — | — | — |
| 2 | SORT | *(baseline)* | — | — | — |
| 3-7 | HUBSORT / HUBCLUSTER / DBG | Faldu et al. | IISWC | 2019 | [hubsort-hubcluster-dbg.md](reordering/hubsort-hubcluster-dbg.md) |
| 8 | RABBITORDER | Arai et al. | IPDPS | 2016 | [rabbitorder.md](reordering/rabbitorder.md) |
| 9 | GORDER | Wei et al. | SIGMOD | 2016 | [gorder.md](reordering/gorder.md) |
| 10 | CORDER | Zhang et al. | IEEE Big Data | 2017 | [corder.md](reordering/corder.md) |
| 11 | RCM | Cuthill-McKee 1969 + RCM++ 2024 | ACM/arXiv | 1969/2024 | [rcm.md](reordering/rcm.md) |
| 12 | GRAPHBREWORDER | Mughrabi et al. | VLDB | 2026 | [graphbreworder.md](reordering/graphbreworder.md) |
| 13 | MAP | *(external mapping)* | — | — | — |
| 14 | ADAPTIVEORDER | Mughrabi et al. | VLDB | 2026 | [adaptiveorder.md](reordering/adaptiveorder.md) |
| 15 | LEIDENORDER | Traag et al. | Sci Rep | 2019 | [leidenorder.md](reordering/leidenorder.md) |
| 16 | GOGRAPHORDER | Zhou et al. | IEEE TPDS | 2024 | [gographorder.md](reordering/gographorder.md) |

## Cache Replacement Policies

| Policy | Paper | Venue | Year | File |
|--------|-------|-------|------|------|
| LRU, FIFO, RANDOM, LFU | *(classic)* | — | — | [baseline-policies.md](caching/baseline-policies.md) |
| SRRIP | Jaleel et al. | ISCA | 2010 | [baseline-policies.md](caching/baseline-policies.md) |
| GRASP | Faldu et al. | HPCA | 2020 | [grasp.md](caching/grasp.md) |
| P-OPT | Balaji et al. | HPCA | 2021 | [popt.md](caching/popt.md) |
| ECG | Mughrabi et al. | GrAPL | 2026 | [ecg.md](caching/ecg.md) |
| DROPLET | Basak et al. | HPCA | 2019 | [droplet.md](caching/droplet.md) |

## Benchmark Kernels

| Kernel | Paper | Venue | Year | File |
|--------|-------|-------|------|------|
| GAP Suite | Beamer et al. | arXiv | 2015 | [gap-suite.md](benchmarks/gap-suite.md) |
| BFS | Beamer et al. | SC | 2012 | [bfs.md](benchmarks/bfs.md) |
| BC | Brandes 2001 + Madduri 2009 | J.Math.Soc / IPDPS | 2001/2009 | [bc.md](benchmarks/bc.md) |
| CC | Sutton et al. (Afforest) | IPDPS | 2018 | [cc.md](benchmarks/cc.md) |
| SSSP | Meyer & Sanders (δ-stepping) | J. Algorithms | 2003 | [sssp.md](benchmarks/sssp.md) |
| PR | Beamer (GAP suite) | — | 2015 | [pr.md](benchmarks/pr.md) |
| TC | Beamer (GAP suite) | — | 2015 | [tc.md](benchmarks/tc.md) |

## ML Features & Metrics

| Reference | File |
|-----------|------|
| IISWC'18, DON-RL, GoGraph FEF, Chen & Chung TPDS 2021, WSR | [features-metrics.md](reordering/features-metrics.md) |

---

## How to Use (for Agents)

1. **Read the corresponding file** before working on any algorithm, cache policy, or benchmark
2. **Do not deviate** from the paper's algorithm without explicit user request
3. **Check the "Why Faithful Implementation Matters" section** to understand consequences of changes
4. **Verify against official repo** when in doubt about implementation details
5. **Cross-reference invariants** from `ecg_config.py` when touching cache simulation code
