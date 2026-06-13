# Per-policy final-octave steepness ranking

**Verdict:** PASS (ranking=GRASP < SRRIP < POPT < LRU)

## Per-policy aggregates (|final-octave slope|, pp/octave)

| policy | n | min | median | mean | max |
|---|---:|---:|---:|---:|---:|
| GRASP | 5 | 0.5390 | 1.0385 | 1.7017 | 3.8282 |
| LRU | 5 | 0.0141 | 2.2442 | 1.8206 | 3.4777 |
| POPT | 5 | 0.0150 | 1.5244 | 1.3083 | 2.5350 |
| SRRIP | 5 | 0.1267 | 1.3576 | 1.4667 | 2.7480 |

## Per-app |final-octave slope| breakdown

| policy | bc | bfs | cc | pr | sssp |
|---|---:|---:|---:|---:|---:|
| GRASP | 0.5390 | 2.5415 | 3.8282 | 1.0385 | 0.5614 |
| LRU | 0.5300 | 0.0141 | 2.8372 | 3.4777 | 2.2442 |
| POPT | 1.6165 | 0.8505 | 2.5350 | 0.0150 | 1.5244 |
| SRRIP | 0.5238 | 0.1267 | 2.7480 | 2.5772 | 1.3576 |

## Checks

| check | ok |
|---|:---:|
| grasp_is_flattest | OK |
| lru_is_steepest | OK |
| grasp_le_lru_median | OK |
| steepness_spread | OK |
| popt_min_saturates | OK |

## Thresholds (locked)

- lru_over_grasp_spread = 1.5
- popt_min_slope_ceiling_pp = 0.2
