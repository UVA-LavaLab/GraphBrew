# Literature-faithfulness statistical-sanity audit (LIT-Stat)

Re-derives `delta_pct` from the two miss-rate columns each row compares (LRU-vs-policy, POPT_GE_GRASP, POPT_NEAR_GRASP_IF_BIG_GAP) and checks for rounding drift, sign flips, out-of-bounds rates, NaN/inf, bad status labels, and status-vs-delta inconsistencies.

## Summary

| Metric | Value |
|---|---|
| Total rows | 330 |
| LRU-vs-policy rows | 102 |
| POPT_GE_GRASP rows | 114 |
| POPT_NEAR_GRASP rows | 114 |
| Unknown-kind rows | 0 |
| Missing-pair rows | 0 |
| NaN / inf values | 0 |
| Miss-rate out of [0,1] | 0 |
| Delta mismatch (> 0.001 pp) | 0 |
| Sign mismatch (above 0.01 pp floor) | 0 |
| Signed-delta inconsistencies | 0 |
| Bad status labels | 0 |
| Status-vs-delta inconsistencies | 0 |
| Apps with > 1 pp signal | 5 (bc, bfs, cc, pr, sssp) |
| Apps flat (max abs Δ ≤ 0.05 pp) | 0 |

## Status counts

| Status | Count |
|---|---|
| `insufficient_data` | 7 |
| `known_deviation` | 24 |
| `ok` | 298 |
| `within_tolerance` | 1 |
