# Cache-saturation onset detection

Source: `wiki/data/oracle_gap_auc.json`  •  paper L3: 1MB, 4MB, 8MB  •  saturation threshold: **0.5 pp/octave**

A cell is 'saturated at L3=Y' if every L3 octave at Y or larger shows a shrinkage rate below the threshold (positive deltas, i.e. anti-scaling, also disqualify).

## Per-policy saturation summary

| policy | saturated at 1MB | at 4MB | at 8MB | never | n apps |
|---|---:|---:|---:|---:|---:|
| **GRASP** | 0 | 2 | 0 | 3 | 5 |
| **LRU** | 0 | 1 | 0 | 4 | 5 |
| **POPT** | 0 | 4 | 0 | 1 | 5 |
| **SRRIP** | 0 | 1 | 0 | 4 | 5 |

**Saturation ordering (earliest → latest):** POPT > GRASP > LRU > SRRIP

## Per-(app, policy) onset

| app | policy | onset | final-octave slope | final-octave delta_pp |
|---|---|---|---:|---:|
| bc | GRASP | 4MB | -0.0000 | +0.0000 |
| bc | LRU | 4MB | 0.3664 | -0.3664 |
| bc | POPT | 4MB | 0.0778 | -0.0778 |
| bc | SRRIP | 4MB | 0.1985 | -0.1985 |
| bfs | GRASP | never | -0.5787 | +0.5787 |
| bfs | LRU | never | -0.2798 | +0.2798 |
| bfs | POPT | 4MB | 0.0335 | -0.0335 |
| bfs | SRRIP | never | -0.3891 | +0.3891 |
| cc | GRASP | never | -0.0010 | +0.0010 |
| cc | LRU | never | 1.0552 | -1.0552 |
| cc | POPT | never | 2.0570 | -2.0570 |
| cc | SRRIP | never | 1.8034 | -1.8034 |
| pr | GRASP | never | 2.2235 | -2.2235 |
| pr | LRU | never | 3.9140 | -3.9140 |
| pr | POPT | 4MB | 0.0407 | -0.0407 |
| pr | SRRIP | never | 3.0988 | -3.0988 |
| sssp | GRASP | 4MB | 0.2296 | -0.2296 |
| sssp | LRU | never | 1.8208 | -1.8208 |
| sssp | POPT | 4MB | 0.2312 | -0.2312 |
| sssp | SRRIP | never | 1.2176 | -1.2176 |

## Interpretation

- POPT should saturate earliest — it's near-oracle at every L3 — if not, the oracle-popularity hint is being wasted.
- LRU and SRRIP rarely saturate at paper L3 — additional cache almost always helps them. This is the mechanism story: oracle-aware policies hit diminishing returns sooner because they're already close to ideal.
