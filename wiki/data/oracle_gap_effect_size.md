# Cliff's delta + Mann-Whitney U on oracle-gap distributions

Source: `wiki/data/oracle_gap.json` (456 rows).

Cliff's delta thresholds (Romano et al. 2006): small ≥ 0.147, medium ≥ 0.33, large ≥ 0.474.

Negative ``cliffs_delta_a_minus_b`` ↔ policy *a* has stochastically smaller gaps (i.e. is the better policy).

## All large-effect dominance pairs (|d| ≥ 0.474, sorted by d asc)

| App | Better (a) | Worse (b) | d (a−b) | MW p | n_a | n_b |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| pr | POPT | LRU | -0.833 | 0.00e+00 | 28 | 28 |
| pr | POPT | SRRIP | -0.833 | 0.00e+00 | 28 | 28 |
| cc | GRASP | LRU | -0.777 | 2.60e-05 | 20 | 20 |
| cc | GRASP | SRRIP | -0.777 | 2.60e-05 | 20 | 20 |
| pr | POPT | GRASP | -0.723 | 3.00e-06 | 28 | 28 |
| bfs | POPT | SRRIP | -0.596 | 5.40e-04 | 23 | 23 |
| bfs | POPT | LRU | -0.556 | 1.24e-03 | 23 | 23 |
| cc | GRASP | POPT | -0.555 | 2.68e-03 | 20 | 20 |
| cc | POPT | LRU | -0.492 | 7.71e-03 | 20 | 20 |
| bc | GRASP | LRU | -0.478 | 5.45e-03 | 23 | 23 |

## Per-app distributions and pairwise tests

### pr

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 28 | 6.584 | 7.734 | 0.000 | 24.264 |
| SRRIP | 28 | 4.538 | 5.867 | 0.000 | 19.923 |
| GRASP | 28 | 2.909 | 3.339 | 0.000 | 10.580 |
| POPT | 28 | 0.000 | 0.009 | 0.000 | 0.170 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.131 | negligible | 3.99e-01 |
| LRU | GRASP | +0.311 | small | 4.56e-02 |
| LRU | POPT | +0.833 | large | 0.00e+00 |
| SRRIP | LRU | -0.131 | negligible | 3.99e-01 |
| SRRIP | GRASP | +0.222 | small | 1.54e-01 |
| SRRIP | POPT | +0.833 | large | 0.00e+00 |
| GRASP | LRU | -0.311 | small | 4.56e-02 |
| GRASP | SRRIP | -0.222 | small | 1.54e-01 |
| GRASP | POPT | +0.723 | large | 3.00e-06 |
| POPT | LRU | -0.833 | large | 0.00e+00 |
| POPT | SRRIP | -0.833 | large | 0.00e+00 |
| POPT | GRASP | -0.723 | large | 3.00e-06 |

### bc

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 23 | 3.545 | 2.867 | 0.000 | 7.388 |
| SRRIP | 23 | 1.361 | 1.705 | 0.000 | 7.208 |
| GRASP | 23 | 0.000 | 1.173 | 0.000 | 16.234 |
| POPT | 23 | 0.274 | 2.119 | 0.000 | 16.399 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.257 | small | 1.35e-01 |
| LRU | GRASP | +0.478 | large | 5.45e-03 |
| LRU | POPT | +0.278 | small | 1.06e-01 |
| SRRIP | LRU | -0.257 | small | 1.35e-01 |
| SRRIP | GRASP | +0.452 | medium | 8.66e-03 |
| SRRIP | POPT | +0.149 | small | 3.86e-01 |
| GRASP | LRU | -0.478 | large | 5.45e-03 |
| GRASP | SRRIP | -0.452 | medium | 8.66e-03 |
| GRASP | POPT | -0.214 | small | 2.15e-01 |
| POPT | LRU | -0.278 | small | 1.06e-01 |
| POPT | SRRIP | -0.149 | small | 3.86e-01 |
| POPT | GRASP | +0.214 | small | 2.15e-01 |

### cc

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 20 | 10.166 | 8.786 | 0.000 | 25.689 |
| SRRIP | 20 | 6.820 | 6.590 | 0.000 | 17.260 |
| GRASP | 20 | 0.000 | 0.500 | 0.000 | 4.576 |
| POPT | 20 | 1.849 | 2.844 | 0.000 | 11.717 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.160 | small | 3.87e-01 |
| LRU | GRASP | +0.777 | large | 2.60e-05 |
| LRU | POPT | +0.492 | large | 7.71e-03 |
| SRRIP | LRU | -0.160 | small | 3.87e-01 |
| SRRIP | GRASP | +0.777 | large | 2.60e-05 |
| SRRIP | POPT | +0.415 | medium | 2.48e-02 |
| GRASP | LRU | -0.777 | large | 2.60e-05 |
| GRASP | SRRIP | -0.777 | large | 2.60e-05 |
| GRASP | POPT | -0.555 | large | 2.68e-03 |
| POPT | LRU | -0.492 | large | 7.71e-03 |
| POPT | SRRIP | -0.415 | medium | 2.48e-02 |
| POPT | GRASP | +0.555 | large | 2.68e-03 |

### bfs

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 23 | 4.406 | 4.811 | 0.000 | 18.425 |
| SRRIP | 23 | 3.522 | 4.477 | 0.000 | 18.301 |
| GRASP | 23 | 0.000 | 5.347 | 0.000 | 62.586 |
| POPT | 23 | 0.000 | 1.988 | 0.000 | 34.291 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.059 | negligible | 7.33e-01 |
| LRU | GRASP | +0.338 | medium | 4.93e-02 |
| LRU | POPT | +0.556 | large | 1.24e-03 |
| SRRIP | LRU | -0.059 | negligible | 7.33e-01 |
| SRRIP | GRASP | +0.357 | medium | 3.79e-02 |
| SRRIP | POPT | +0.596 | large | 5.40e-04 |
| GRASP | LRU | -0.338 | medium | 4.93e-02 |
| GRASP | SRRIP | -0.357 | medium | 3.79e-02 |
| GRASP | POPT | +0.104 | negligible | 5.46e-01 |
| POPT | LRU | -0.556 | large | 1.24e-03 |
| POPT | SRRIP | -0.596 | large | 5.40e-04 |
| POPT | GRASP | -0.104 | negligible | 5.46e-01 |

### sssp

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 20 | 2.997 | 3.739 | 0.000 | 11.387 |
| SRRIP | 20 | 1.587 | 2.520 | 0.000 | 9.701 |
| GRASP | 20 | 0.136 | 4.584 | 0.000 | 55.827 |
| POPT | 20 | 0.168 | 1.444 | 0.000 | 8.799 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.182 | small | 3.23e-01 |
| LRU | GRASP | +0.400 | medium | 3.05e-02 |
| LRU | POPT | +0.390 | medium | 3.49e-02 |
| SRRIP | LRU | -0.182 | small | 3.23e-01 |
| SRRIP | GRASP | +0.335 | medium | 6.99e-02 |
| SRRIP | POPT | +0.273 | small | 1.40e-01 |
| GRASP | LRU | -0.400 | medium | 3.05e-02 |
| GRASP | SRRIP | -0.335 | medium | 6.99e-02 |
| GRASP | POPT | -0.100 | negligible | 5.89e-01 |
| POPT | LRU | -0.390 | medium | 3.49e-02 |
| POPT | SRRIP | -0.273 | small | 1.40e-01 |
| POPT | GRASP | +0.100 | negligible | 5.89e-01 |
