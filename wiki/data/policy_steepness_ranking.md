# Per-policy final-octave steepness ranking

**Verdict:** PASS (ranking=POPT < GRASP < SRRIP < LRU)

## Per-policy aggregates (|final-octave slope|, pp/octave)

| policy | n | min | median | mean | max |
|---|---:|---:|---:|---:|---:|
| GRASP | 5 | 0.0000 | 0.6236 | 1.1397 | 2.2946 |
| LRU | 5 | 0.2541 | 1.3244 | 1.7897 | 4.7344 |
| POPT | 5 | 0.0283 | 0.5118 | 0.5196 | 1.1184 |
| SRRIP | 5 | 0.1418 | 1.2354 | 1.3907 | 3.8335 |

## Per-app |final-octave slope| breakdown

| policy | bc | bfs | cc | pr | sssp |
|---|---:|---:|---:|---:|---:|
| GRASP | 0.6236 | 2.2730 | 0.0000 | 2.2946 | 0.5074 |
| LRU | 0.4455 | 0.2541 | 1.3244 | 4.7344 | 2.1902 |
| POPT | 1.1184 | 0.5118 | 0.7490 | 0.0283 | 0.1904 |
| SRRIP | 0.4393 | 0.1418 | 1.2354 | 3.8335 | 1.3036 |

## Checks

| check | ok |
|---|:---:|
| popt_le_grasp_median | OK |
| grasp_le_lru_median | OK |
| popt_lt_srrip_median | OK |
| oracle_aware_ceiling | OK |
| non_oracle_floor | OK |
| oracle_half_of_non_oracle | OK |
| popt_min_saturates | OK |

## Thresholds (locked)

- oracle_aware_ceiling_pp = 0.7
- non_oracle_floor_pp = 0.5
- oracle_aware_half_of_non_oracle = 0.5
- popt_min_slope_ceiling_pp = 0.2
