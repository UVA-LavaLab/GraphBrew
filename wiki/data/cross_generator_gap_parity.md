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
| bc | GRASP | 1MB | 6.0079 | 7 | 0.000000 |
| bc | GRASP | 4MB | 0.0 | 6 | 0.000000 |
| bc | GRASP | 8MB | 0.0 | 6 | 0.000000 |
| bc | LRU | 1MB | 3.3026 | 7 | 0.000000 |
| bc | LRU | 4MB | 3.9427 | 6 | 0.000000 |
| bc | LRU | 8MB | 3.5763 | 6 | 0.000000 |
| bc | POPT | 1MB | 4.4764 | 7 | 0.000000 |
| bc | POPT | 4MB | 1.8168 | 6 | 0.000000 |
| bc | POPT | 8MB | 1.739 | 6 | 0.000000 |
| bc | SRRIP | 1MB | 1.7656 | 7 | 0.000000 |
| bc | SRRIP | 4MB | 2.0555 | 6 | 0.000000 |
| bc | SRRIP | 8MB | 1.857 | 6 | 0.000000 |
| bfs | GRASP | 1MB | 12.3081 | 7 | 0.000000 |
| bfs | GRASP | 4MB | 0.1443 | 6 | 0.000000 |
| bfs | GRASP | 8MB | 0.723 | 6 | 0.000000 |
| bfs | LRU | 1MB | 2.1866 | 7 | 0.000000 |
| bfs | LRU | 4MB | 6.2392 | 6 | 0.000000 |
| bfs | LRU | 8MB | 6.519 | 6 | 0.000000 |
| bfs | POPT | 1MB | 5.3117 | 7 | 0.000000 |
| bfs | POPT | 4MB | 0.3562 | 6 | 0.000000 |
