# Cliff's delta + Mann-Whitney U on oracle-gap distributions

Source: `wiki/data/oracle_gap.json` (456 rows).

Cliff's delta thresholds (Romano et al. 2006): small ≥ 0.147, medium ≥ 0.33, large ≥ 0.474.

Negative ``cliffs_delta_a_minus_b`` ↔ policy *a* has stochastically smaller gaps (i.e. is the better policy).

## All large-effect dominance pairs (|d| ≥ 0.474, sorted by d asc)

| App | Better (a) | Worse (b) | d (a−b) | MW p | n_a | n_b |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| pr | POPT | LRU | -0.911 | 0.00e+00 | 28 | 28 |
| pr | POPT | SRRIP | -0.890 | 0.00e+00 | 28 | 28 |
| cc | GRASP | POPT | -0.843 | 5.00e-06 | 20 | 20 |
| cc | GRASP | SRRIP | -0.815 | 1.00e-05 | 20 | 20 |
| cc | GRASP | LRU | -0.797 | 1.60e-05 | 20 | 20 |
| bfs | POPT | SRRIP | -0.679 | 8.00e-05 | 23 | 23 |
| bfs | POPT | LRU | -0.656 | 1.38e-04 | 23 | 23 |
| bc | GRASP | LRU | -0.633 | 2.33e-04 | 23 | 23 |
| pr | POPT | GRASP | -0.569 | 2.58e-04 | 28 | 28 |
| bc | GRASP | POPT | -0.516 | 2.71e-03 | 23 | 23 |

## Per-app distributions and pairwise tests

### pr

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 28 | 5.228 | 6.733 | 0.013 | 21.615 |
| SRRIP | 28 | 4.758 | 4.934 | 0.001 | 17.434 |
| GRASP | 28 | 1.052 | 2.251 | 0.000 | 8.246 |
| POPT | 28 | 0.000 | 0.100 | 0.000 | 1.072 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.154 | small | 3.21e-01 |
| LRU | GRASP | +0.467 | medium | 2.71e-03 |
| LRU | POPT | +0.911 | large | 0.00e+00 |
| SRRIP | LRU | -0.154 | small | 3.21e-01 |
| SRRIP | GRASP | +0.371 | medium | 1.71e-02 |
| SRRIP | POPT | +0.890 | large | 0.00e+00 |
| GRASP | LRU | -0.467 | medium | 2.71e-03 |
| GRASP | SRRIP | -0.371 | medium | 1.71e-02 |
| GRASP | POPT | +0.569 | large | 2.58e-04 |
| POPT | LRU | -0.911 | large | 0.00e+00 |
| POPT | SRRIP | -0.890 | large | 0.00e+00 |
| POPT | GRASP | -0.569 | large | 2.58e-04 |

### bc

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 23 | 2.097 | 3.201 | 0.000 | 8.691 |
| SRRIP | 23 | 1.537 | 1.689 | 0.000 | 6.371 |
| GRASP | 23 | 0.000 | 2.124 | 0.000 | 38.128 |
| POPT | 23 | 1.597 | 2.487 | 0.000 | 22.541 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.325 | small | 5.88e-02 |
| LRU | GRASP | +0.633 | large | 2.33e-04 |
| LRU | POPT | +0.268 | small | 1.19e-01 |
| SRRIP | LRU | -0.325 | small | 5.88e-02 |
| SRRIP | GRASP | +0.422 | medium | 1.43e-02 |
| SRRIP | POPT | -0.053 | negligible | 7.58e-01 |
| GRASP | LRU | -0.633 | large | 2.33e-04 |
| GRASP | SRRIP | -0.422 | medium | 1.43e-02 |
| GRASP | POPT | -0.516 | large | 2.71e-03 |
| POPT | LRU | -0.268 | small | 1.19e-01 |
| POPT | SRRIP | +0.053 | negligible | 7.58e-01 |
| POPT | GRASP | +0.516 | large | 2.71e-03 |

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
| LRU | 23 | 2.710 | 4.046 | 0.000 | 18.069 |
| SRRIP | 23 | 2.363 | 3.882 | 0.000 | 17.967 |
| GRASP | 23 | 0.386 | 5.107 | 0.000 | 68.404 |
| POPT | 23 | 0.000 | 1.625 | 0.000 | 33.391 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.006 | negligible | 9.74e-01 |
| LRU | GRASP | +0.406 | medium | 1.82e-02 |
| LRU | POPT | +0.656 | large | 1.38e-04 |
| SRRIP | LRU | -0.006 | negligible | 9.74e-01 |
| SRRIP | GRASP | +0.410 | medium | 1.71e-02 |
| SRRIP | POPT | +0.679 | large | 8.00e-05 |
| GRASP | LRU | -0.406 | medium | 1.82e-02 |
| GRASP | SRRIP | -0.410 | medium | 1.71e-02 |
| GRASP | POPT | +0.248 | small | 1.50e-01 |
| POPT | LRU | -0.656 | large | 1.38e-04 |
| POPT | SRRIP | -0.679 | large | 8.00e-05 |
| POPT | GRASP | -0.248 | small | 1.50e-01 |

### sssp

Distribution (gap_pp):

| Policy | n | median | mean | min | max |
| --- | ---: | ---: | ---: | ---: | ---: |
| LRU | 20 | 2.245 | 2.903 | 0.000 | 11.039 |
| SRRIP | 20 | 1.756 | 2.084 | 0.000 | 7.868 |
| GRASP | 20 | 0.013 | 7.106 | 0.000 | 70.117 |
| POPT | 20 | 0.175 | 1.713 | 0.000 | 13.820 |

Pairwise tests (Cliff's d, MW-U two-sided p):

| a | b | d (a−b) | magnitude | MW p |
| --- | --- | ---: | --- | ---: |
| LRU | SRRIP | +0.083 | negligible | 6.55e-01 |
| LRU | GRASP | +0.290 | small | 1.17e-01 |
| LRU | POPT | +0.360 | medium | 5.15e-02 |
| SRRIP | LRU | -0.083 | negligible | 6.55e-01 |
| SRRIP | GRASP | +0.307 | small | 9.62e-02 |
| SRRIP | POPT | +0.390 | medium | 3.49e-02 |
| GRASP | LRU | -0.290 | small | 1.17e-01 |
| GRASP | SRRIP | -0.307 | small | 9.62e-02 |
| GRASP | POPT | +0.050 | negligible | 7.87e-01 |
| POPT | LRU | -0.360 | medium | 5.15e-02 |
| POPT | SRRIP | -0.390 | medium | 3.49e-02 |
| POPT | GRASP | -0.050 | negligible | 7.87e-01 |

