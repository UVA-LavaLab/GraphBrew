# Per-graph oracle-gap cache-sensitivity slope

Source: `wiki/data/oracle_gap.json`  •  paper L3: 1MB, 4MB, 8MB  •  graphs with full trajectory: 6  •  full (graph, app, policy) trajectories: 112

Significant anti-scaling threshold: **+1.0 pp** per octave. A cell that grows its gap by that much (or more) at any single octave is flagged.

## Headline

- Cells with significant anti-scaling: **33 / 112**
- Of those, **7** belong to the oracle-aware policies (GRASP + POPT). The remainder are LRU + SRRIP.

## Per-policy anti-scaling cell count

| policy | n_cells_with_anti_scaling |
|---|---:|
| **GRASP** | 2 |
| **LRU** | 13 |
| **POPT** | 5 |
| **SRRIP** | 13 |

## Per-graph anti-scaling cell count

| graph | family | n_cells_with_anti_scaling |
|---|---|---:|
| cit-Patents | citation | 10 |
| com-orkut | social | 5 |
| email-Eu-core | social | 0 |
| soc-LiveJournal1 | social | 9 |
| soc-pokec | social | 3 |
| web-Google | web | 6 |

## Top anti-scaling cells (largest single-octave gap growth)

| graph | family | app | policy | max octave growth | 1MB → 4MB → 8MB gap |
|---|---|---|---|---:|---|
| web-Google | web | bfs | LRU | +14.70 pp | 3.37 → 18.07 → 7.92 |
| web-Google | web | bfs | SRRIP | +14.67 pp | 3.30 → 17.97 → 7.44 |
| cit-Patents | citation | pr | LRU | +9.92 pp | 11.70 → 21.61 → 13.86 |
| com-orkut | social | cc | LRU | +7.60 pp | 10.52 → 13.35 → 20.94 |
| cit-Patents | citation | pr | GRASP | +7.19 pp | 1.05 → 8.25 → 2.83 |
| cit-Patents | citation | pr | SRRIP | +7.16 pp | 10.27 → 17.43 → 8.94 |
| soc-pokec | social | bfs | SRRIP | +5.57 pp | 1.90 → 6.51 → 12.08 |
| soc-pokec | social | bfs | LRU | +5.53 pp | 2.87 → 6.65 → 12.18 |
| cit-Patents | citation | sssp | LRU | +5.22 pp | 3.45 → 8.67 → 6.90 |
| com-orkut | social | cc | SRRIP | +5.17 pp | 6.73 → 11.90 → 10.00 |
| cit-Patents | citation | bfs | SRRIP | +4.92 pp | 0.60 → 3.40 → 8.31 |
| soc-LiveJournal1 | social | cc | LRU | +4.58 pp | 3.82 → 6.63 → 11.21 |
| cit-Patents | citation | bfs | LRU | +4.54 pp | 1.00 → 4.08 → 8.62 |
| soc-pokec | social | cc | LRU | +3.86 pp | 13.11 → 16.97 → 0.04 |
| soc-LiveJournal1 | social | cc | SRRIP | +3.56 pp | 2.72 → 5.50 → 9.06 |
| cit-Patents | citation | bc | LRU | +3.22 pp | 0.63 → 2.05 → 5.27 |
| soc-LiveJournal1 | social | sssp | LRU | +3.21 pp | 1.47 → 4.67 → 3.58 |
| cit-Patents | citation | sssp | SRRIP | +3.08 pp | 3.13 → 6.21 → 4.41 |
| soc-LiveJournal1 | social | pr | LRU | +2.71 pp | 10.74 → 13.45 → 9.60 |
| soc-LiveJournal1 | social | pr | SRRIP | +2.34 pp | 7.46 → 9.80 → 6.10 |
| com-orkut | social | bc | POPT | +2.19 pp | 2.48 → 4.67 → 4.73 |
| web-Google | web | bfs | GRASP | +2.10 pp | 0.00 → 0.39 → 2.48 |
| cit-Patents | citation | bc | SRRIP | +2.03 pp | 0.00 → 0.85 → 2.88 |
| soc-LiveJournal1 | social | sssp | SRRIP | +1.94 pp | 1.41 → 3.35 → 2.14 |
| com-orkut | social | bfs | LRU | +1.90 pp | 0.23 → 2.04 → 3.94 |

## Interpretation

- If `n_oracle_aware_anti_scaling` is 0, the paper can claim: 'no individual graph shows GRASP or POPT regressing as L3 grows' — strictly stronger than the corpus-averaged finding in gate 52.
- If a small handful of (graph, app) cells flag GRASP/POPT anti-scaling, those should be called out specifically in the paper as known exceptions worth disclosing.
