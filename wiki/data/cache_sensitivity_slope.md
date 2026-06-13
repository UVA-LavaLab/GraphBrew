# Per-(app, policy) cache-sensitivity slope across L3 octaves

Source: `wiki/data/oracle_gap_auc.json`  •  apps=5  •  policies=4  •  L3 octaves: 1MB → 4MB → 8MB

Slope units: **gap_pp per L3 octave (log2 MB)**. Positive slope means the oracle gap shrinks as L3 grows (the expected sign for any sensible policy).

## Headline

- ⚠️ 11 (app, policy) cells violate monotonicity. See 'Monotonic violations' below.

## Per-policy slope summary (mean across apps)

| policy | mean slope | stdev slope | min slope | max slope | n_apps |
|---|---:|---:|---:|---:|---:|
| **GRASP** | 0.5331 | 1.7508 | -2.4838 | 2.901 | 5 |
| **LRU** | 0.4592 | 1.1468 | -1.2273 | 1.8831 | 5 |
| **POPT** | 0.5218 | 0.9165 | -0.6429 | 1.9474 | 5 |
| **SRRIP** | 0.5293 | 0.9764 | -1.0443 | 1.8057 | 5 |

## Per-(app, policy) avg slope (gap_pp shrinkage per L3 octave)

| app | GRASP | LRU | POPT | SRRIP |
|---|---|---|---|---|
| **bc** | 0.584 | -0.436 | -0.643 | 0.010 |
| **bfs** | 1.299 | -1.227 | 1.947 | -1.044 |
| **cc** | -2.484 | 0.697 | -0.043 | 0.769 |
| **pr** | 0.365 | 1.883 | 0.201 | 1.806 |
| **sssp** | 2.901 | 1.379 | 1.146 | 1.107 |

## Monotonic violations

| app | policy | 1MB→4MB Δgap | 4MB→8MB Δgap |
|---|---|---:|---:|
| bc | GRASP | -2.2914 | 0.539 |
| bc | LRU | 1.8378 | -0.53 |
| bc | POPT | 0.3121 | 1.6165 |
| bc | SRRIP | 0.4948 | -0.5238 |
| bfs | GRASP | -6.4396 | 2.5415 |
| bfs | LRU | 3.6677 | 0.0141 |
| bfs | SRRIP | 3.0062 | 0.1267 |
| cc | GRASP | 3.6232 | 3.8282 |
| cc | LRU | 0.7453 | -2.8372 |
| cc | POPT | 2.6627 | -2.535 |
| cc | SRRIP | 0.4423 | -2.748 |

## Interpretation

- A high mean slope means the policy benefits a lot from each extra cache octave — *cache-hungry*.
- A near-zero mean slope means extra cache buys little improvement — *cache-saturating*. POPT typically saturates fast because it's already close to the oracle ceiling.
- Per-octave slope drop (1MB→4MB vs 4MB→8MB) reveals where a policy hits diminishing returns. The headline policy choice can vary depending on which L3 octave the design targets.
