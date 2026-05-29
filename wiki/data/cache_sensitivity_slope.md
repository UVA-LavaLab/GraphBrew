# Per-(app, policy) cache-sensitivity slope across L3 octaves

Source: `wiki/data/oracle_gap_auc.json`  •  apps=5  •  policies=4  •  L3 octaves: 1MB → 4MB → 8MB

Slope units: **gap_pp per L3 octave (log2 MB)**. Positive slope means the oracle gap shrinks as L3 grows (the expected sign for any sensible policy).

## Headline

- ⚠️ 10 (app, policy) cells violate monotonicity. See 'Monotonic violations' below.

## Per-policy slope summary (mean across apps)

| policy | mean slope | stdev slope | min slope | max slope | n_apps |
|---|---:|---:|---:|---:|---:|
| **GRASP** | 2.2929 | 1.5706 | 0.5063 | 4.3436 | 5 |
| **LRU** | 0.1467 | 1.0754 | -1.5284 | 1.7952 | 5 |
| **POPT** | 0.9131 | 0.5948 | -0.0289 | 1.6829 | 5 |
| **SRRIP** | 0.1685 | 0.9573 | -1.3739 | 1.6087 | 5 |

## Per-(app, policy) avg slope (gap_pp shrinkage per L3 octave)

| app | GRASP | LRU | POPT | SRRIP |
|---|---|---|---|---|
| **bc** | 1.950 | -0.071 | 0.843 | 0.011 |
| **bfs** | 3.884 | -1.528 | 1.683 | -1.374 |
| **cc** | 0.506 | -0.033 | 1.391 | 0.111 |
| **pr** | 0.780 | 1.795 | -0.029 | 1.609 |
| **sssp** | 4.344 | 0.571 | 0.677 | 0.486 |

## Monotonic violations

| app | policy | 1MB→4MB Δgap | 4MB→8MB Δgap |
|---|---|---:|---:|
| bc | LRU | 0.5092 | -0.297 |
| bc | SRRIP | 0.2719 | -0.3045 |
| bfs | GRASP | -12.1509 | 0.4992 |
| bfs | LRU | 4.0006 | 0.5847 |
| bfs | POPT | -5.1007 | 0.0519 |
| bfs | SRRIP | 3.5108 | 0.611 |
| cc | GRASP | -1.52 | 0.001 |
| cc | LRU | 1.154 | -1.0552 |
| cc | SRRIP | 1.4707 | -1.8034 |
| pr | POPT | 0.1302 | -0.0435 |

## Interpretation

- A high mean slope means the policy benefits a lot from each extra cache octave — *cache-hungry*.
- A near-zero mean slope means extra cache buys little improvement — *cache-saturating*. POPT typically saturates fast because it's already close to the oracle ceiling.
- Per-octave slope drop (1MB→4MB vs 4MB→8MB) reveals where a policy hits diminishing returns. The headline policy choice can vary depending on which L3 octave the design targets.
