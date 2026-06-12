# Cache-saturation onset detection

Source: `wiki/data/oracle_gap_auc.json`  •  paper L3: 1MB, 4MB, 8MB  •  saturation threshold: **0.5 pp/octave**

A cell is 'saturated at L3=Y' if every L3 octave at Y or larger shows a shrinkage rate below the threshold (positive deltas, i.e. anti-scaling, also disqualify).

## Per-policy saturation summary

| policy | saturated at 1MB | at 4MB | at 8MB | never | n apps |
|---|---:|---:|---:|---:|---:|
| **GRASP** | 0 | 1 | 0 | 4 | 5 |
| **LRU** | 0 | 2 | 0 | 3 | 5 |
| **POPT** | 0 | 1 | 0 | 4 | 5 |
| **SRRIP** | 0 | 2 | 0 | 3 | 5 |

**Saturation ordering (earliest → latest):** LRU > SRRIP > GRASP > POPT

## Per-(app, policy) onset

| app | policy | onset | final-octave slope | final-octave delta_pp |
|---|---|---|---:|---:|
| bc | GRASP | never | -0.6236 | +0.6236 |
| bc | LRU | 4MB | 0.4455 | -0.4455 |
| bc | POPT | never | -1.1184 | +1.1184 |
| bc | SRRIP | 4MB | 0.4393 | -0.4393 |
| bfs | GRASP | never | -2.2730 | +2.2730 |
| bfs | LRU | 4MB | 0.2541 | -0.2541 |
| bfs | POPT | never | 0.5118 | -0.5118 |
| bfs | SRRIP | 4MB | 0.1418 | -0.1418 |
| cc | GRASP | 4MB | -0.0000 | +0.0000 |
| cc | LRU | never | 1.3244 | -1.3244 |
| cc | POPT | never | 0.7490 | -0.7490 |
| cc | SRRIP | never | 1.2354 | -1.2354 |
| pr | GRASP | never | 2.2946 | -2.2946 |
| pr | LRU | never | 4.7344 | -4.7344 |
| pr | POPT | never | -0.0283 | +0.0283 |
| pr | SRRIP | never | 3.8335 | -3.8335 |
| sssp | GRASP | never | 0.5074 | -0.5074 |
| sssp | LRU | never | 2.1902 | -2.1902 |
| sssp | POPT | 4MB | 0.1904 | -0.1904 |
| sssp | SRRIP | never | 1.3036 | -1.3036 |

## Interpretation

- POPT should saturate earliest — it's near-oracle at every L3 — if not, the oracle-popularity hint is being wasted.
- LRU and SRRIP rarely saturate at paper L3 — additional cache almost always helps them. This is the mechanism story: oracle-aware policies hit diminishing returns sooner because they're already close to ideal.
