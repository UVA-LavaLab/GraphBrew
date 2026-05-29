# Per-policy final-octave steepness ranking

**Verdict:** PASS (ranking=POPT < GRASP < LRU < SRRIP)

## Per-policy aggregates (|final-octave slope|, pp/octave)

| policy | n | min | median | mean | max |
|---|---:|---:|---:|---:|---:|
| GRASP | 5 | 0.0000 | 0.2294 | 0.5783 | 2.1618 |
| LRU | 5 | 0.2970 | 1.0552 | 1.4743 | 3.8421 |
| POPT | 5 | 0.0435 | 0.0995 | 0.4752 | 2.0570 |
| SRRIP | 5 | 0.3045 | 1.0910 | 1.3638 | 3.0090 |

## Per-app |final-octave slope| breakdown

| policy | bc | bfs | cc | pr | sssp |
|---|---:|---:|---:|---:|---:|
| GRASP | 0.0000 | 0.4992 | 0.0010 | 2.1618 | 0.2294 |
| LRU | 0.2970 | 0.5847 | 1.0552 | 3.8421 | 1.5926 |
| POPT | 0.0995 | 0.0519 | 2.0570 | 0.0435 | 0.1242 |
| SRRIP | 0.3045 | 0.6110 | 1.8034 | 3.0090 | 1.0910 |

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

- oracle_aware_ceiling_pp = 0.5
- non_oracle_floor_pp = 0.5
- oracle_aware_half_of_non_oracle = 0.5
- popt_min_slope_ceiling_pp = 0.2
