# Cliff's delta + Mann-Whitney U on oracle-gap distributions

Source: `wiki/data/oracle_gap.json` (456 rows).

Cliff's delta thresholds (Romano et al. 2006): small ≥ 0.147, medium ≥ 0.33, large ≥ 0.474.

Negative ``cliffs_delta_a_minus_b`` ↔ policy *a* has stochastically smaller gaps (i.e. is the better policy).

## All large-effect dominance pairs (|d| ≥ 0.474, sorted by d asc)

| App | Better (a) | Worse (b) | d (a−b) | MW p | n_a | n_b |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| pr | POPT | LRU | -0.772 | 1.00e-06 | 28 | 28 |
| pr | POPT | SRRIP | -0.717 | 4.00e-06 | 28 | 28 |
| bc | GRASP | POPT | -0.590 | 6.10e-04 | 23 | 23 |
| cc | POPT | LRU | -0.568 | 2.14e-03 | 20 | 20 |
| bc | GRASP | LRU | -0.554 | 1.29e-03 | 23 | 23 |
| pr | POPT | GRASP | -0.551 | 4.01e-04 | 28 | 28 |
| sssp | GRASP | POPT | -0.545 | 3.19e-03 | 20 | 20 |
| bc | GRASP | SRRIP | -0.520 | 2.52e-03 | 23 | 23 |
| bfs | POPT | SRRIP | -0.503 | 3.48e-03 | 23 | 23 |
| cc | POPT | SRRIP | -0.487 | 8.35e-03 | 20 | 20 |
| sssp | GRASP | LRU | -0.480 | 9.41e-03 | 20 | 20 |

## Per-app distributions and pairwise tests

### pr

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 28 | 5.293 | 6.376 | 0.000 | 19.939 |
| SRRIP | 28 | 3.946 | 4.509 | 0.000 | 15.598 |
| GRASP | 28 | 1.898 | 1.980 | 0.000 | 6.445 |
| POPT | 28 | 0.000 | 0.247 | 0.000 | 3.548 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.154 | small | 3.21e-01 |
| LRU | GRASP | +0.421 | medium | 6.85e-03 |
| LRU | POPT | +0.772 | large | 1.00e-06 |
| SRRIP | LRU | -0.154 | small | 3.21e-01 |
| SRRIP | GRASP | +0.323 | small | 3.82e-02 |
| SRRIP | POPT | +0.717 | large | 4.00e-06 |
| GRASP | LRU | -0.421 | medium | 6.85e-03 |
| GRASP | SRRIP | -0.323 | small | 3.82e-02 |
| GRASP | POPT | +0.551 | large | 4.01e-04 |
| POPT | LRU | -0.772 | large | 1.00e-06 |
| POPT | SRRIP | -0.717 | large | 4.00e-06 |
| POPT | GRASP | -0.551 | large | 4.01e-04 |

### bc

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 23 | 3.369 | 2.809 | 0.000 | 7.388 |
| SRRIP | 23 | 1.304 | 1.647 | 0.000 | 7.208 |
| GRASP | 23 | 0.000 | 1.115 | 0.000 | 16.234 |
| POPT | 23 | 3.221 | 4.066 | 0.000 | 17.674 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.253 | small | 1.41e-01 |
| LRU | GRASP | +0.554 | large | 1.29e-03 |
| LRU | POPT | -0.076 | negligible | 6.60e-01 |
| SRRIP | LRU | -0.253 | small | 1.41e-01 |
| SRRIP | GRASP | +0.520 | large | 2.52e-03 |
| SRRIP | POPT | -0.344 | medium | 4.56e-02 |
| GRASP | LRU | -0.554 | large | 1.29e-03 |
| GRASP | SRRIP | -0.520 | large | 2.52e-03 |
| GRASP | POPT | -0.590 | large | 6.10e-04 |
| POPT | LRU | +0.076 | negligible | 6.60e-01 |
| POPT | SRRIP | +0.344 | medium | 4.56e-02 |
| POPT | GRASP | +0.590 | large | 6.10e-04 |

### cc

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 20 | 6.729 | 5.558 | 0.000 | 14.712 |
| SRRIP | 20 | 2.616 | 3.362 | 0.000 | 13.271 |
| GRASP | 20 | 0.321 | 3.905 | 0.000 | 31.015 |
| POPT | 20 | 0.000 | 1.471 | 0.000 | 12.887 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.278 | small | 1.33e-01 |
| LRU | GRASP | +0.378 | medium | 4.11e-02 |
| LRU | POPT | +0.568 | large | 2.14e-03 |
| SRRIP | LRU | -0.278 | small | 1.33e-01 |
| SRRIP | GRASP | +0.307 | small | 9.62e-02 |
| SRRIP | POPT | +0.487 | large | 8.35e-03 |
| GRASP | LRU | -0.378 | medium | 4.11e-02 |
| GRASP | SRRIP | -0.307 | small | 9.62e-02 |
| GRASP | POPT | +0.128 | negligible | 4.90e-01 |
| POPT | LRU | -0.568 | large | 2.14e-03 |
| POPT | SRRIP | -0.487 | large | 8.35e-03 |
| POPT | GRASP | -0.128 | negligible | 4.90e-01 |

### bfs

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 23 | 4.406 | 4.250 | 0.000 | 14.964 |
| SRRIP | 23 | 3.522 | 3.915 | 0.000 | 14.840 |
| GRASP | 23 | 0.000 | 4.785 | 0.000 | 62.586 |
| POPT | 23 | 0.027 | 2.393 | 0.000 | 35.126 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.062 | negligible | 7.17e-01 |
| LRU | GRASP | +0.422 | medium | 1.43e-02 |
| LRU | POPT | +0.458 | medium | 7.85e-03 |
| SRRIP | LRU | -0.062 | negligible | 7.17e-01 |
| SRRIP | GRASP | +0.461 | medium | 7.36e-03 |
| SRRIP | POPT | +0.503 | large | 3.48e-03 |
| GRASP | LRU | -0.422 | medium | 1.43e-02 |
| GRASP | SRRIP | -0.461 | medium | 7.36e-03 |
| GRASP | POPT | -0.119 | negligible | 4.89e-01 |
| POPT | LRU | -0.458 | medium | 7.85e-03 |
| POPT | SRRIP | -0.503 | large | 3.48e-03 |
| POPT | GRASP | +0.119 | negligible | 4.89e-01 |

### sssp

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 20 | 2.684 | 3.638 | 0.000 | 11.387 |
| SRRIP | 20 | 1.326 | 2.418 | 0.000 | 9.701 |
| GRASP | 20 | 0.000 | 4.483 | 0.000 | 55.827 |
| POPT | 20 | 4.040 | 4.030 | 0.000 | 10.818 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.203 | small | 2.73e-01 |
| LRU | GRASP | +0.480 | large | 9.41e-03 |
| LRU | POPT | -0.080 | negligible | 6.65e-01 |
| SRRIP | LRU | -0.203 | small | 2.73e-01 |
| SRRIP | GRASP | +0.415 | medium | 2.48e-02 |
| SRRIP | POPT | -0.307 | small | 9.62e-02 |
| GRASP | LRU | -0.480 | large | 9.41e-03 |
| GRASP | SRRIP | -0.415 | medium | 2.48e-02 |
| GRASP | POPT | -0.545 | large | 3.19e-03 |
| POPT | LRU | +0.080 | negligible | 6.65e-01 |
| POPT | SRRIP | +0.307 | small | 9.62e-02 |
| POPT | GRASP | +0.545 | large | 3.19e-03 |
