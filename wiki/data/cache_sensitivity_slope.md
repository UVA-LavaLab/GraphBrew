# Per-(app, policy) cache-sensitivity slope across L3 octaves

Source: `wiki/data/oracle_gap_auc.json`  •  apps=5  •  policies=4  •  L3 octaves: 1MB → 4MB → 8MB

Slope units: **gap_pp per L3 octave (log2 MB)**. Positive slope means the oracle gap shrinks as L3 grows (the expected sign for any sensible policy).

## Headline

- ⚠️ 10 (app, policy) cells violate monotonicity. See 'Monotonic violations' below.

## Per-policy slope summary (mean across apps)

| policy | mean slope | stdev slope | min slope | max slope | n_apps |
|---|---:|---:|---:|---:|---:|
| **GRASP** | 1.183 | 0.8924 | 0.5556 | 2.9252 | 5 |
| **LRU** | 0.4737 | 1.3003 | -1.4224 | 2.2558 | 5 |
| **POPT** | 0.4612 | 0.7308 | -0.2064 | 1.7619 | 5 |
| **SRRIP** | 0.5439 | 1.1385 | -1.2394 | 2.1784 | 5 |

## Per-(app, policy) avg slope (gap_pp shrinkage per L3 octave)

| app | GRASP | LRU | POPT | SRRIP |
|---|---|---|---|---|
| **bc** | 0.593 | -0.427 | -0.006 | 0.018 |
| **bfs** | 1.104 | -1.422 | 1.762 | -1.239 |
| **cc** | 0.556 | 0.560 | -0.206 | 0.631 |
| **pr** | 0.737 | 2.256 | -0.009 | 2.178 |
| **sssp** | 2.925 | 1.403 | 0.766 | 1.131 |

## Monotonic violations

| app | policy | 1MB→4MB Δgap | 4MB→8MB Δgap |
|---|---|---:|---:|
| bc | GRASP | -2.4017 | 0.6236 |
| bc | LRU | 1.7274 | -0.4455 |
| bc | POPT | -1.1 | 1.1184 |
| bc | SRRIP | 0.3845 | -0.4393 |
| bfs | GRASP | -5.5856 | 2.273 |
| bfs | LRU | 4.5214 | -0.2541 |
| bfs | SRRIP | 3.8601 | -0.1418 |
| cc | POPT | 1.3682 | -0.749 |
| pr | GRASP | 0.0828 | -2.2946 |
| pr | POPT | 0.0 | 0.0283 |

## Interpretation

- A high mean slope means the policy benefits a lot from each extra cache octave — *cache-hungry*.
- A near-zero mean slope means extra cache buys little improvement — *cache-saturating*. POPT typically saturates fast because it's already close to the oracle ceiling.
- Per-octave slope drop (1MB→4MB vs 4MB→8MB) reveals where a policy hits diminishing returns. The headline policy choice can vary depending on which L3 octave the design targets.
