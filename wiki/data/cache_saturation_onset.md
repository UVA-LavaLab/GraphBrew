# Cache-saturation onset detection

Source: `wiki/data/oracle_gap_auc.json`  •  paper L3: 1MB, 4MB, 8MB  •  saturation threshold: **0.5 pp/octave**

A cell is 'saturated at L3=Y' if every L3 octave at Y or larger shows a shrinkage rate below the threshold (positive deltas, i.e. anti-scaling, also disqualify).

## Per-policy saturation summary

| policy | saturated at 1MB | at 4MB | at 8MB | never | n apps |
|---|---:|---:|---:|---:|---:|
| **GRASP** | 0 | 0 | 0 | 5 | 5 |
| **LRU** | 0 | 0 | 0 | 5 | 5 |
| **POPT** | 0 | 1 | 0 | 4 | 5 |
| **SRRIP** | 0 | 0 | 0 | 5 | 5 |

**Saturation ordering (earliest → latest):** POPT > GRASP > LRU > SRRIP

## Per-(app, policy) onset

| app | policy | onset | final-octave slope | final-octave delta_pp |
|---|---|---|---:|---:|
| bc | GRASP | never | -0.5390 | +0.5390 |
| bc | LRU | never | 0.5300 | -0.5300 |
| bc | POPT | never | -1.6165 | +1.6165 |
| bc | SRRIP | never | 0.5238 | -0.5238 |
| bfs | GRASP | never | -2.5415 | +2.5415 |
| bfs | LRU | never | -0.0141 | +0.0141 |
| bfs | POPT | never | 0.8505 | -0.8505 |
| bfs | SRRIP | never | -0.1267 | +0.1267 |
| cc | GRASP | never | -3.8282 | +3.8282 |
| cc | LRU | never | 2.8372 | -2.8372 |
| cc | POPT | never | 2.5350 | -2.5350 |
| cc | SRRIP | never | 2.7480 | -2.7480 |
| pr | GRASP | never | 1.0385 | -1.0385 |
| pr | LRU | never | 3.4777 | -3.4777 |
| pr | POPT | 4MB | 0.0150 | -0.0150 |
| pr | SRRIP | never | 2.5772 | -2.5772 |
| sssp | GRASP | never | 0.5614 | -0.5614 |
| sssp | LRU | never | 2.2442 | -2.2442 |
| sssp | POPT | never | 1.5244 | -1.5244 |
| sssp | SRRIP | never | 1.3576 | -1.3576 |

## Interpretation

- POPT should saturate earliest — it's near-oracle at every L3 — if not, the oracle-popularity hint is being wasted.
- LRU and SRRIP rarely saturate at paper L3 — additional cache almost always helps them. This is the mechanism story: oracle-aware policies hit diminishing returns sooner because they're already close to ideal.
