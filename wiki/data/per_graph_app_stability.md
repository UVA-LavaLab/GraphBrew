# Per-(graph × app) winner stability across L3

Source: `wiki/data/oracle_gap.json`  •  Scope: 1MB, 4MB, 8MB

Cells: **34** (stable-unique 14, stable-partial 2, regime-change 12, insufficient-L3 6).
Stability fraction (excluding insufficient-L3): **50.0%**.

## Per-graph rollup

| graph | n_apps | stable_unique | partial | regime_change |
|---|---:|---:|---:|---:|
| cit-Patents | 5 | 4 | 0 | 1 |
| com-orkut | 5 | 3 | 0 | 2 |
| delaunay_n19 | 1 | 0 | 0 | 0 |
| email-Eu-core | 3 | 0 | 2 | 1 |
| roadNet-CA | 5 | 0 | 0 | 0 |
| soc-LiveJournal1 | 5 | 4 | 0 | 1 |
| soc-pokec | 5 | 3 | 0 | 2 |
| web-Google | 5 | 0 | 0 | 5 |

## Stable-unique cells (paper-quotable without per-L3 disclaimer)

- cit-Patents/bfs -> GRASP
- cit-Patents/cc -> GRASP
- cit-Patents/pr -> POPT
- cit-Patents/sssp -> GRASP
- com-orkut/bc -> GRASP
- com-orkut/cc -> GRASP
- com-orkut/pr -> POPT
- soc-LiveJournal1/bc -> GRASP
- soc-LiveJournal1/bfs -> GRASP
- soc-LiveJournal1/cc -> GRASP
- soc-LiveJournal1/pr -> POPT
- soc-pokec/bc -> GRASP
- soc-pokec/cc -> GRASP
- soc-pokec/sssp -> GRASP

## Stable-partial cells (tied across L3)

- email-Eu-core/bc -> GRASP,LRU,SRRIP
- email-Eu-core/pr -> GRASP,LRU,POPT,SRRIP

## Regime-change cells (paper MUST break out per L3)

- cit-Patents/bc
- com-orkut/bfs
- com-orkut/sssp
- email-Eu-core/bfs
- soc-LiveJournal1/sssp
- soc-pokec/bfs
- soc-pokec/pr
- web-Google/bc
- web-Google/bfs
- web-Google/cc
- web-Google/pr
- web-Google/sssp

## Insufficient-L3 cells (only one paper L3 size present)

- delaunay_n19/pr
- roadNet-CA/bc
- roadNet-CA/bfs
- roadNet-CA/cc
- roadNet-CA/pr
- roadNet-CA/sssp

## Full per-(graph, app) table

| graph | app | L3 sizes | winners per L3 | intersection | classification |
|---|---|---|---|---|---|
| cit-Patents | bc | 1MB,4MB,8MB | 1MB=SRRIP; 4MB=GRASP; 8MB=GRASP | ∅ | regime_change |
| cit-Patents | bfs | 1MB,4MB,8MB | 1MB=GRASP; 4MB=GRASP; 8MB=GRASP | GRASP | stable_unique |
| cit-Patents | cc | 1MB,4MB,8MB | 1MB=GRASP; 4MB=GRASP; 8MB=GRASP | GRASP | stable_unique |
| cit-Patents | pr | 1MB,4MB,8MB | 1MB=POPT; 4MB=POPT; 8MB=POPT | POPT | stable_unique |
| cit-Patents | sssp | 1MB,4MB,8MB | 1MB=GRASP; 4MB=GRASP; 8MB=GRASP | GRASP | stable_unique |
| com-orkut | bc | 1MB,4MB,8MB | 1MB=GRASP; 4MB=GRASP; 8MB=GRASP | GRASP | stable_unique |
| com-orkut | bfs | 1MB,4MB,8MB | 1MB=GRASP; 4MB=POPT; 8MB=POPT | ∅ | regime_change |
| com-orkut | cc | 1MB,4MB,8MB | 1MB=GRASP; 4MB=GRASP; 8MB=GRASP | GRASP | stable_unique |
| com-orkut | pr | 1MB,4MB,8MB | 1MB=POPT; 4MB=POPT; 8MB=POPT | POPT | stable_unique |
| com-orkut | sssp | 1MB,4MB,8MB | 1MB=POPT; 4MB=POPT; 8MB=SRRIP | ∅ | regime_change |
| delaunay_n19 | pr | 1MB | 1MB=POPT | ∅ | insufficient_l3 |
| email-Eu-core | bc | 1MB,4MB,8MB | 1MB=GRASP,LRU,POPT,SRRIP; 4MB=GRASP,LRU,SRRIP; 8MB=GRASP,LRU,POPT,SRRIP | GRASP,LRU,SRRIP | stable_partial |
| email-Eu-core | bfs | 1MB,4MB,8MB | 1MB=SRRIP; 4MB=GRASP,LRU,POPT,SRRIP; 8MB=GRASP | ∅ | regime_change |
| email-Eu-core | pr | 1MB,4MB,8MB | 1MB=GRASP,LRU,POPT,SRRIP; 4MB=GRASP,LRU,POPT,SRRIP; 8MB=GRASP,LRU,POPT,SRRIP | GRASP,LRU,POPT,SRRIP | stable_partial |
| roadNet-CA | bc | 1MB | 1MB=SRRIP | ∅ | insufficient_l3 |
| roadNet-CA | bfs | 1MB | 1MB=LRU | ∅ | insufficient_l3 |
| roadNet-CA | cc | 1MB | 1MB=LRU | ∅ | insufficient_l3 |
| roadNet-CA | pr | 1MB | 1MB=POPT | ∅ | insufficient_l3 |
| roadNet-CA | sssp | 1MB | 1MB=LRU | ∅ | insufficient_l3 |
| soc-LiveJournal1 | bc | 1MB,4MB,8MB | 1MB=GRASP; 4MB=GRASP; 8MB=GRASP | GRASP | stable_unique |
| soc-LiveJournal1 | bfs | 1MB,4MB,8MB | 1MB=GRASP; 4MB=GRASP; 8MB=GRASP | GRASP | stable_unique |
| soc-LiveJournal1 | cc | 1MB,4MB,8MB | 1MB=GRASP; 4MB=GRASP; 8MB=GRASP | GRASP | stable_unique |
| soc-LiveJournal1 | pr | 1MB,4MB,8MB | 1MB=POPT; 4MB=POPT; 8MB=POPT | POPT | stable_unique |
| soc-LiveJournal1 | sssp | 1MB,4MB,8MB | 1MB=POPT; 4MB=GRASP; 8MB=GRASP | ∅ | regime_change |
| soc-pokec | bc | 1MB,4MB,8MB | 1MB=GRASP; 4MB=GRASP; 8MB=GRASP | GRASP | stable_unique |
| soc-pokec | bfs | 1MB,4MB,8MB | 1MB=GRASP; 4MB=POPT; 8MB=POPT | ∅ | regime_change |
| soc-pokec | cc | 1MB,4MB,8MB | 1MB=GRASP; 4MB=GRASP; 8MB=GRASP | GRASP | stable_unique |
| soc-pokec | pr | 1MB,4MB,8MB | 1MB=GRASP; 4MB=POPT; 8MB=GRASP | ∅ | regime_change |
| soc-pokec | sssp | 1MB,4MB,8MB | 1MB=GRASP; 4MB=GRASP; 8MB=GRASP,POPT | GRASP | stable_unique_with_ties |
| web-Google | bc | 1MB,4MB,8MB | 1MB=SRRIP; 4MB=GRASP; 8MB=GRASP | ∅ | regime_change |
| web-Google | bfs | 1MB,4MB,8MB | 1MB=GRASP; 4MB=POPT; 8MB=POPT | ∅ | regime_change |
| web-Google | cc | 1MB,4MB,8MB | 1MB=GRASP; 4MB=GRASP; 8MB=LRU | ∅ | regime_change |
| web-Google | pr | 1MB,4MB,8MB | 1MB=POPT; 4MB=GRASP; 8MB=POPT | ∅ | regime_change |
| web-Google | sssp | 1MB,4MB,8MB | 1MB=POPT; 4MB=LRU; 8MB=SRRIP | ∅ | regime_change |
