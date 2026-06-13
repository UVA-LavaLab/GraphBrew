# Cross-generator gap_pp parity

Tolerance: **0.001 pp**  •  cells checked: 60  •  full triple cells 60  •  mismatches: **0**

## Sources reconciled

- oracle_gap: `wiki/data/oracle_gap.json`
- oracle_gap_auc: `wiki/data/oracle_gap_auc.json`
- cache_sensitivity: `wiki/data/cache_sensitivity_slope.json`

## Headline

✅ All three aggregators report identical gap_pp values (within tolerance) for every (app, policy, L3) triple they share.

## Sample of reconciled cells

| app | policy | L3 | gap_pp (raw) | n graphs | spread |
|---|---|---|---:|---:|---:|
| bc | GRASP | 1MB | 2.6201 | 7 | 0.000000 |
| bc | GRASP | 4MB | 0.3287 | 6 | 0.000000 |
| bc | GRASP | 8MB | 0.8677 | 6 | 0.000000 |
| bc | LRU | 1MB | 1.9817 | 7 | 0.000000 |
| bc | LRU | 4MB | 3.8195 | 6 | 0.000000 |
| bc | LRU | 8MB | 3.2895 | 6 | 0.000000 |
| bc | POPT | 1MB | 3.9859 | 7 | 0.000000 |
| bc | POPT | 4MB | 4.298 | 6 | 0.000000 |
| bc | POPT | 8MB | 5.9145 | 6 | 0.000000 |
| bc | SRRIP | 1MB | 1.431 | 7 | 0.000000 |
| bc | SRRIP | 4MB | 1.9258 | 6 | 0.000000 |
| bc | SRRIP | 8MB | 1.402 | 6 | 0.000000 |
| bfs | GRASP | 1MB | 8.9409 | 7 | 0.000000 |
| bfs | GRASP | 4MB | 2.5013 | 6 | 0.000000 |
| bfs | GRASP | 8MB | 5.0428 | 6 | 0.000000 |
| bfs | LRU | 1MB | 2.047 | 7 | 0.000000 |
| bfs | LRU | 4MB | 5.7147 | 6 | 0.000000 |
| bfs | LRU | 8MB | 5.7288 | 6 | 0.000000 |
| bfs | POPT | 1MB | 6.01 | 7 | 0.000000 |
| bfs | POPT | 4MB | 1.0182 | 6 | 0.000000 |
