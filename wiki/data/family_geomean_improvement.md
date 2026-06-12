# Family × App × Policy: geomean improvement vs LRU

Source: `wiki/data/oracle_gap.json`  •  L3 scope: 1MB, 4MB, 8MB
Bootstrap: B=2000, seed=1729, α=0.05 (percentile CI)

Records: **63** (33 CI-strict improvements vs LRU, 2 CI-strict regressions).

## Headline CI-strict improvements (≥10% miss-rate reduction)

| Family | App | Policy | n | Geomean | CI ratio | Improve%  (CI) |
|---|---|---|---|---|---|---|
| citation | pr | POPT | 3 | 0.6409 | [0.548, 0.816] | +35.91% [+18.43, +45.18] |
| social | cc | GRASP | 9 | 0.7436 | [0.636, 0.856] | +25.64% [+14.37, +36.40] |
| citation | cc | GRASP | 3 | 0.7496 | [0.637, 0.838] | +25.04% [+16.16, +36.29] |
| web | pr | POPT | 3 | 0.7666 | [0.642, 0.868] | +23.34% [+13.18, +35.82] |
| social | pr | POPT | 12 | 0.7758 | [0.698, 0.862] | +22.42% [+13.77, +30.22] |
| citation | cc | POPT | 3 | 0.7845 | [0.731, 0.822] | +21.55% [+17.82, +26.95] |
| citation | pr | GRASP | 3 | 0.7851 | [0.689, 0.914] | +21.49% [+8.59, +31.08] |
| social | cc | POPT | 9 | 0.8376 | [0.764, 0.910] | +16.24% [+8.97, +23.58] |
| social | pr | GRASP | 12 | 0.8585 | [0.806, 0.913] | +14.15% [+8.68, +19.42] |
| social | sssp | POPT | 9 | 0.8805 | [0.831, 0.939] | +11.95% [+6.14, +16.93] |
| citation | sssp | GRASP | 3 | 0.8887 | [0.843, 0.951] | +11.13% [+4.92, +15.75] |
| web | bfs | POPT | 3 | 0.8909 | [0.806, 0.975] | +10.91% [+2.53, +19.42] |
| citation | sssp | POPT | 3 | 0.8947 | [0.838, 0.958] | +10.53% [+4.24, +16.25] |
| web | pr | SRRIP | 3 | 0.8998 | [0.839, 0.961] | +10.02% [+3.90, +16.14] |

## CI-strict regressions (geomean ratio > 1.0, CI lo > 1.0)

| Family | App | Policy | n | Geomean | CI ratio | Improve%  (CI) |
|---|---|---|---|---|---|---|
| web | bc | POPT | 3 | 1.1214 | [1.024, 1.309] | -12.14% [-30.86, -2.35] |
| web | bc | GRASP | 3 | 1.0712 | [1.005, 1.192] | -7.12% [-19.21, -0.47] |

## All records (sorted by family, app, policy)

| Family | App | Policy | n | Geomean ratio | CI ratio | Improve% | CI-strict |
|---|---|---|---|---|---|---|---|
| citation | bc | GRASP | 3 | 0.9575 | [0.934, 0.984] | +4.25% | improvement |
| citation | bc | POPT | 3 | 0.9737 | [0.968, 0.983] | +2.63% | improvement |
| citation | bc | SRRIP | 3 | 0.9848 | [0.970, 0.997] | +1.52% | improvement |
| citation | bfs | GRASP | 3 | 0.9635 | [0.946, 0.992] | +3.65% | improvement |
| citation | bfs | POPT | 3 | 0.9543 | [0.912, 0.991] | +4.57% | improvement |
| citation | bfs | SRRIP | 3 | 0.9965 | [0.995, 0.998] | +0.35% | improvement |
| citation | cc | GRASP | 3 | 0.7496 | [0.637, 0.838] | +25.04% | improvement |
| citation | cc | POPT | 3 | 0.7845 | [0.731, 0.822] | +21.55% | improvement |
| citation | cc | SRRIP | 3 | 0.9189 | [0.888, 0.939] | +8.11% | improvement |
| citation | pr | GRASP | 3 | 0.7851 | [0.689, 0.914] | +21.49% | improvement |
| citation | pr | POPT | 3 | 0.6409 | [0.548, 0.816] | +35.91% | improvement |
| citation | pr | SRRIP | 3 | 0.9153 | [0.842, 0.983] | +8.47% | improvement |
| citation | sssp | GRASP | 3 | 0.8887 | [0.843, 0.951] | +11.13% | improvement |
| citation | sssp | POPT | 3 | 0.8947 | [0.838, 0.958] | +10.53% | improvement |
| citation | sssp | SRRIP | 3 | 0.9575 | [0.920, 0.992] | +4.25% | improvement |
| mesh | pr | GRASP | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| mesh | pr | POPT | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| mesh | pr | SRRIP | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | bc | GRASP | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | bc | POPT | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | bc | SRRIP | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | bfs | GRASP | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | bfs | POPT | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | bfs | SRRIP | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | cc | GRASP | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | cc | POPT | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | cc | SRRIP | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | pr | GRASP | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | pr | POPT | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | pr | SRRIP | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | sssp | GRASP | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | sssp | POPT | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| road | sssp | SRRIP | 1 | _skipped_ | _skipped_ | _skipped_ | _insufficient_cells_for_bootstrap_min_2_ |
| social | bc | GRASP | 12 | 0.9289 | [0.898, 0.959] | +7.11% | improvement |
| social | bc | POPT | 12 | 0.9465 | [0.906, 0.981] | +5.35% | improvement |
| social | bc | SRRIP | 12 | 0.9634 | [0.945, 0.980] | +3.66% | improvement |
| social | bfs | GRASP | 12 | 0.9648 | [0.943, 0.984] | +3.52% | improvement |
| social | bfs | POPT | 12 | 0.9484 | [0.915, 0.979] | +5.16% | improvement |
| social | bfs | SRRIP | 12 | 0.9889 | [0.981, 0.996] | +1.11% | improvement |
| social | cc | GRASP | 9 | 0.7436 | [0.636, 0.856] | +25.64% | improvement |
| social | cc | POPT | 9 | 0.8376 | [0.764, 0.910] | +16.24% | improvement |
| social | cc | SRRIP | 9 | 0.9317 | [0.889, 0.970] | +6.83% | improvement |
| social | pr | GRASP | 12 | 0.8585 | [0.806, 0.913] | +14.15% | improvement |
| social | pr | POPT | 12 | 0.7758 | [0.698, 0.862] | +22.42% | improvement |
| social | pr | SRRIP | 12 | 0.9190 | [0.883, 0.954] | +8.10% | improvement |
| social | sssp | GRASP | 9 | 0.9620 | [0.830, 1.222] | +3.80% | — |
| social | sssp | POPT | 9 | 0.8805 | [0.831, 0.939] | +11.95% | improvement |
| social | sssp | SRRIP | 9 | 0.9300 | [0.905, 0.956] | +7.00% | improvement |
| web | bc | GRASP | 3 | 1.0712 | [1.005, 1.192] | -7.12% | regression |
| web | bc | POPT | 3 | 1.1214 | [1.024, 1.309] | -12.14% | regression |
| web | bc | SRRIP | 3 | 0.9759 | [0.957, 1.005] | +2.41% | — |
| web | bfs | GRASP | 3 | 1.0130 | [0.963, 1.114] | -1.30% | — |
| web | bfs | POPT | 3 | 0.8909 | [0.806, 0.975] | +10.91% | improvement |
| web | bfs | SRRIP | 3 | 0.9970 | [0.993, 0.999] | +0.30% | improvement |
| web | cc | GRASP | 3 | 0.9730 | [0.921, 1.000] | +2.70% | — |
| web | cc | POPT | 3 | 0.9464 | [0.848, 1.000] | +5.36% | — |
| web | cc | SRRIP | 3 | 0.9832 | [0.950, 1.000] | +1.68% | — |
| web | pr | GRASP | 3 | 0.9097 | [0.754, 1.100] | +9.03% | — |
| web | pr | POPT | 3 | 0.7666 | [0.642, 0.868] | +23.34% | improvement |
| web | pr | SRRIP | 3 | 0.8998 | [0.839, 0.961] | +10.02% | improvement |
| web | sssp | GRASP | 3 | 1.8937 | [0.965, 3.476] | -89.37% | — |
| web | sssp | POPT | 3 | 0.9889 | [0.967, 1.000] | +1.11% | — |
| web | sssp | SRRIP | 3 | 0.9962 | [0.989, 1.000] | +0.38% | — |
