# Per-policy final-octave steepness ranking

**Verdict:** PASS (ranking=POPT < GRASP < LRU < SRRIP)

## Per-policy aggregates (|final-octave slope|, pp/octave)

| policy | n | min | median | mean | max |
|---|---:|---:|---:|---:|---:|
| GRASP | 5 | 0.0000 | 0.2296 | 0.6066 | 2.2235 |
| LRU | 5 | 0.2798 | 1.0552 | 1.4872 | 3.9140 |
| POPT | 5 | 0.0335 | 0.0778 | 0.4880 | 2.0570 |
| SRRIP | 5 | 0.1985 | 1.2176 | 1.3415 | 3.0988 |

## Per-app |final-octave slope| breakdown

| policy | bc | bfs | cc | pr | sssp |
|---|---:|---:|---:|---:|---:|
| GRASP | 0.0000 | 0.5787 | 0.0010 | 2.2235 | 0.2296 |
| LRU | 0.3664 | 0.2798 | 1.0552 | 3.9140 | 1.8208 |
| POPT | 0.0778 | 0.0335 | 2.0570 | 0.0407 | 0.2312 |
| SRRIP | 0.1985 | 0.3891 | 1.8034 | 3.0988 | 1.2176 |

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
