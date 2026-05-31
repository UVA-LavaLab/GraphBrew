# Cliff's delta + Mann-Whitney U on oracle-gap distributions

Source: `wiki/data/oracle_gap.json` (456 rows).

Cliff's delta thresholds (Romano et al. 2006): small ≥ 0.147, medium ≥ 0.33, large ≥ 0.474.

Negative ``cliffs_delta_a_minus_b`` ↔ policy *a* has stochastically smaller gaps (i.e. is the better policy).

## All large-effect dominance pairs (|d| ≥ 0.474, sorted by d asc)

| App | Better (a) | Worse (b) | d (a−b) | MW p | n_a | n_b |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| cc | GRASP | POPT | -0.843 | 5.00e-06 | 20 | 20 |
| pr | POPT | LRU | -0.829 | 0.00e+00 | 28 | 28 |
| pr | POPT | SRRIP | -0.824 | 0.00e+00 | 28 | 28 |
| cc | GRASP | SRRIP | -0.815 | 1.00e-05 | 20 | 20 |
| cc | GRASP | LRU | -0.797 | 1.60e-05 | 20 | 20 |
| bc | GRASP | LRU | -0.618 | 3.28e-04 | 23 | 23 |
| bfs | POPT | SRRIP | -0.618 | 3.28e-04 | 23 | 23 |
| bfs | POPT | LRU | -0.584 | 6.88e-04 | 23 | 23 |
| pr | POPT | GRASP | -0.561 | 3.12e-04 | 28 | 28 |
| bfs | GRASP | SRRIP | -0.516 | 2.71e-03 | 23 | 23 |
| bfs | GRASP | LRU | -0.501 | 3.60e-03 | 23 | 23 |

## Per-app distributions and pairwise tests

### pr

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 28 | 5.228 | 6.818 | 0.000 | 22.195 |
| SRRIP | 28 | 4.758 | 5.004 | 0.000 | 17.918 |
| GRASP | 28 | 1.294 | 2.315 | 0.000 | 8.652 |
| POPT | 28 | 0.000 | 0.095 | 0.000 | 1.063 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.138 | negligible | 3.76e-01 |
| LRU | GRASP | +0.427 | medium | 6.05e-03 |
| LRU | POPT | +0.829 | large | 0.00e+00 |
| SRRIP | LRU | -0.138 | negligible | 3.76e-01 |
| SRRIP | GRASP | +0.341 | medium | 2.87e-02 |
| SRRIP | POPT | +0.824 | large | 0.00e+00 |
| GRASP | LRU | -0.427 | medium | 6.05e-03 |
| GRASP | SRRIP | -0.341 | medium | 2.87e-02 |
| GRASP | POPT | +0.561 | large | 3.12e-04 |
| POPT | LRU | -0.829 | large | 0.00e+00 |
| POPT | SRRIP | -0.824 | large | 0.00e+00 |
| POPT | GRASP | -0.561 | large | 3.12e-04 |

### bc

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 23 | 2.097 | 3.109 | 0.000 | 8.662 |
| SRRIP | 23 | 1.318 | 1.641 | 0.000 | 6.386 |
| GRASP | 23 | 0.000 | 2.172 | 0.000 | 38.128 |
| POPT | 23 | 1.445 | 2.304 | 0.000 | 22.541 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.318 | small | 6.50e-02 |
| LRU | GRASP | +0.618 | large | 3.28e-04 |
| LRU | POPT | +0.287 | small | 9.50e-02 |
| SRRIP | LRU | -0.318 | small | 6.50e-02 |
| SRRIP | GRASP | +0.410 | medium | 1.71e-02 |
| SRRIP | POPT | -0.015 | negligible | 9.30e-01 |
| GRASP | LRU | -0.618 | large | 3.28e-04 |
| GRASP | SRRIP | -0.410 | medium | 1.71e-02 |
| GRASP | POPT | -0.471 | medium | 6.24e-03 |
| POPT | LRU | -0.287 | small | 9.50e-02 |
| POPT | SRRIP | +0.015 | negligible | 9.30e-01 |
| POPT | GRASP | +0.471 | medium | 6.24e-03 |

### cc

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 20 | 6.629 | 6.484 | 0.000 | 20.943 |
| SRRIP | 20 | 4.303 | 4.316 | 0.000 | 11.896 |
| GRASP | 20 | 0.000 | 0.639 | 0.000 | 9.120 |
| POPT | 20 | 1.810 | 3.572 | 0.004 | 11.016 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.168 | small | 3.65e-01 |
| LRU | GRASP | +0.797 | large | 1.60e-05 |
| LRU | POPT | +0.203 | small | 2.73e-01 |
| SRRIP | LRU | -0.168 | small | 3.65e-01 |
| SRRIP | GRASP | +0.815 | large | 1.00e-05 |
| SRRIP | POPT | +0.022 | negligible | 9.03e-01 |
| GRASP | LRU | -0.797 | large | 1.60e-05 |
| GRASP | SRRIP | -0.815 | large | 1.00e-05 |
| GRASP | POPT | -0.843 | large | 5.00e-06 |
| POPT | LRU | -0.203 | small | 2.73e-01 |
| POPT | SRRIP | -0.022 | negligible | 9.03e-01 |
| POPT | GRASP | +0.843 | large | 5.00e-06 |

### bfs

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 23 | 3.418 | 4.598 | 0.000 | 18.153 |
| SRRIP | 23 | 3.740 | 4.221 | 0.000 | 17.920 |
| GRASP | 23 | 0.000 | 4.937 | 0.000 | 68.404 |
| POPT | 23 | 0.044 | 1.801 | 0.000 | 33.391 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.025 | negligible | 8.86e-01 |
| LRU | GRASP | +0.501 | large | 3.60e-03 |
| LRU | POPT | +0.584 | large | 6.88e-04 |
| SRRIP | LRU | -0.025 | negligible | 8.86e-01 |
| SRRIP | GRASP | +0.516 | large | 2.71e-03 |
| SRRIP | POPT | +0.618 | large | 3.28e-04 |
| GRASP | LRU | -0.501 | large | 3.60e-03 |
| GRASP | SRRIP | -0.516 | large | 2.71e-03 |
| GRASP | POPT | -0.068 | negligible | 6.93e-01 |
| POPT | LRU | -0.584 | large | 6.88e-04 |
| POPT | SRRIP | -0.618 | large | 3.28e-04 |
| POPT | GRASP | +0.068 | negligible | 6.93e-01 |

### sssp

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 20 | 3.062 | 3.115 | 0.000 | 10.823 |
| SRRIP | 20 | 2.163 | 2.100 | 0.000 | 7.677 |
| GRASP | 20 | 0.013 | 6.930 | 0.000 | 70.117 |
| POPT | 20 | 0.175 | 1.694 | 0.000 | 13.820 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.138 | negligible | 4.57e-01 |
| LRU | GRASP | +0.355 | medium | 5.48e-02 |
| LRU | POPT | +0.380 | medium | 3.98e-02 |
| SRRIP | LRU | -0.138 | negligible | 4.57e-01 |
| SRRIP | GRASP | +0.385 | medium | 3.73e-02 |
| SRRIP | POPT | +0.415 | medium | 2.48e-02 |
| GRASP | LRU | -0.355 | medium | 5.48e-02 |
| GRASP | SRRIP | -0.385 | medium | 3.73e-02 |
| GRASP | POPT | -0.020 | negligible | 9.14e-01 |
| POPT | LRU | -0.380 | medium | 3.98e-02 |
| POPT | SRRIP | -0.415 | medium | 2.48e-02 |
| POPT | GRASP | +0.020 | negligible | 9.14e-01 |
