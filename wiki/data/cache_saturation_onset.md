# Cache-saturation onset detection

Source: `wiki/data/oracle_gap_auc.json`  •  paper L3: 1MB, 4MB, 8MB  •  saturation threshold: **0.5 pp/octave**

A cell is 'saturated at L3=Y' if every L3 octave at Y or larger shows a shrinkage rate below the threshold (positive deltas, i.e. anti-scaling, also disqualify).

## Per-policy saturation summary

| policy | saturated at 1MB | at 4MB | at 8MB | never | n apps |
|---|---:|---:|---:|---:|---:|
| **GRASP** | 0 | 2 | 0 | 3 | 5 |
| **LRU** | 0 | 1 | 0 | 4 | 5 |
| **POPT** | 0 | 3 | 0 | 2 | 5 |
| **SRRIP** | 0 | 1 | 0 | 4 | 5 |

**Saturation ordering (earliest → latest):** POPT > GRASP > LRU > SRRIP

## Per-(app, policy) onset

| app | policy | onset | final-octave slope | final-octave delta_pp |
|---|---|---|---:|---:|
| bc | GRASP | 4MB | -0.0000 | +0.0000 |
| bc | LRU | 4MB | 0.2970 | -0.2970 |
| bc | POPT | 4MB | 0.0995 | -0.0995 |
| bc | SRRIP | 4MB | 0.3045 | -0.3045 |
| bfs | GRASP | never | -0.4992 | +0.4992 |
| bfs | LRU | never | -0.5847 | +0.5847 |
| bfs | POPT | never | -0.0519 | +0.0519 |
| bfs | SRRIP | never | -0.6110 | +0.6110 |
| cc | GRASP | never | -0.0010 | +0.0010 |
| cc | LRU | never | 1.0552 | -1.0552 |
| cc | POPT | never | 2.0570 | -2.0570 |
| cc | SRRIP | never | 1.8034 | -1.8034 |
| pr | GRASP | never | 2.1618 | -2.1618 |
| pr | LRU | never | 3.8421 | -3.8421 |
| pr | POPT | 4MB | 0.0435 | -0.0435 |
| pr | SRRIP | never | 3.0090 | -3.0090 |
| sssp | GRASP | 4MB | 0.2294 | -0.2294 |
| sssp | LRU | never | 1.5926 | -1.5926 |
| sssp | POPT | 4MB | 0.1242 | -0.1242 |
| sssp | SRRIP | never | 1.0910 | -1.0910 |

## Interpretation

- POPT should saturate earliest — it's near-oracle at every L3 — if not, the oracle-popularity hint is being wasted.
- LRU and SRRIP rarely saturate at paper L3 — additional cache almost always helps them. This is the mechanism story: oracle-aware policies hit diminishing returns sooner because they're already close to ideal.
