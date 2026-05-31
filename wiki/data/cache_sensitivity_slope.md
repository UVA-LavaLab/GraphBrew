# Per-(app, policy) cache-sensitivity slope across L3 octaves

Source: `wiki/data/oracle_gap_auc.json`  •  apps=5  •  policies=4  •  L3 octaves: 1MB → 4MB → 8MB

Slope units: **gap_pp per L3 octave (log2 MB)**. Positive slope means the oracle gap shrinks as L3 grows (the expected sign for any sensible policy).

## Headline

- ⚠️ 9 (app, policy) cells violate monotonicity. See 'Monotonic violations' below.

## Per-policy slope summary (mean across apps)

| policy | mean slope | stdev slope | min slope | max slope | n_apps |
|---|---:|---:|---:|---:|---:|
| **GRASP** | 2.2621 | 1.5121 | 0.5063 | 4.1486 | 5 |
| **LRU** | 0.1885 | 1.0652 | -1.4441 | 1.8049 | 5 |
| **POPT** | 0.9132 | 0.5936 | -0.031 | 1.663 | 5 |
| **SRRIP** | 0.1795 | 0.9463 | -1.313 | 1.6275 | 5 |

## Per-(app, policy) avg slope (gap_pp shrinkage per L3 octave)

| app | GRASP | LRU | POPT | SRRIP |
|---|---|---|---|---|
| **bc** | 2.003 | -0.091 | 0.912 | -0.030 |
| **bfs** | 3.862 | -1.444 | 1.663 | -1.313 |
| **cc** | 0.506 | -0.033 | 1.391 | 0.111 |
| **pr** | 0.791 | 1.805 | -0.031 | 1.627 |
| **sssp** | 4.149 | 0.706 | 0.631 | 0.502 |

## Monotonic violations

| app | policy | 1MB→4MB Δgap | 4MB→8MB Δgap |
|---|---|---:|---:|
| bc | LRU | 0.6401 | -0.3664 |
| bc | SRRIP | 0.2899 | -0.1985 |
| bfs | GRASP | -12.1638 | 0.5787 |
| bfs | LRU | 4.0526 | 0.2798 |
| bfs | SRRIP | 3.5498 | 0.3891 |
| cc | GRASP | -1.52 | 0.001 |
| cc | LRU | 1.154 | -1.0552 |
| cc | SRRIP | 1.4707 | -1.8034 |
| pr | POPT | 0.1338 | -0.0407 |

## Interpretation

- A high mean slope means the policy benefits a lot from each extra cache octave — *cache-hungry*.
- A near-zero mean slope means extra cache buys little improvement — *cache-saturating*. POPT typically saturates fast because it's already close to the oracle ceiling.
- Per-octave slope drop (1MB→4MB vs 4MB→8MB) reveals where a policy hits diminishing returns. The headline policy choice can vary depending on which L3 octave the design targets.
