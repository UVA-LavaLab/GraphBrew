# Family × App × Policy: geomean improvement vs LRU

Source: `wiki/data/oracle_gap.json`  •  L3 scope: 1MB, 4MB, 8MB
Bootstrap: B=2000, seed=1729, α=0.05 (percentile CI)

Records: **63** (28 CI-strict improvements vs LRU, 2 CI-strict regressions).

## Headline CI-strict improvements (≥10% miss-rate reduction)

| Family | App | Policy | n | Geomean | CI ratio | Improve%  (CI) |
|---|---|---|---|---|---|---|
| citation | pr | POPT | 3 | 0.7082 | [0.631, 0.850] | +29.18% [+14.97, +36.90] |
| citation | pr | GRASP | 3 | 0.7851 | [0.689, 0.914] | +21.49% [+8.59, +31.08] |
| social | pr | POPT | 12 | 0.8299 | [0.764, 0.899] | +17.01% [+10.10, +23.61] |
| web | pr | POPT | 3 | 0.8310 | [0.718, 0.902] | +16.91% [+9.77, +28.16] |
| citation | cc | GRASP | 3 | 0.8314 | [0.800, 0.894] | +16.86% [+10.59, +19.97] |
| social | pr | GRASP | 12 | 0.8585 | [0.806, 0.913] | +14.15% [+8.68, +19.42] |
| citation | cc | POPT | 3 | 0.8705 | [0.862, 0.883] | +12.95% [+11.73, +13.81] |
| citation | sssp | GRASP | 3 | 0.8887 | [0.843, 0.951] | +11.13% [+4.92, +15.75] |
| social | cc | POPT | 9 | 0.8908 | [0.844, 0.939] | +10.92% [+6.11, +15.57] |
| web | pr | SRRIP | 3 | 0.8998 | [0.839, 0.961] | +10.02% [+3.90, +16.14] |

## CI-strict regressions (geomean ratio > 1.0, CI lo > 1.0)

| Family | App | Policy | n | Geomean | CI ratio | Improve%  (CI) |
|---|---|---|---|---|---|---|
| web | bc | POPT | 3 | 1.2223 | [1.044, 1.549] | -22.23% [-54.91, -4.38] |
| web | bc | GRASP | 3 | 1.0712 | [1.005, 1.192] | -7.12% [-19.21, -0.47] |

## All records (sorted by family, app, policy)

| Family | App | Policy | n | Geomean ratio | CI ratio | Improve% | CI-strict |
|---|---|---|---|---|---|---|---|
| citation | bc | GRASP | 3 | 0.9575 | [0.934, 0.984] | +4.25% | improvement |
| citation | bc | POPT | 3 | 0.9991 | [0.991, 1.011] | +0.09% | — |
| citation | bc | SRRIP | 3 | 0.9848 | [0.970, 0.997] | +1.52% | improvement |
| citation | bfs | GRASP | 3 | 0.9635 | [0.946, 0.992] | +3.65% | improvement |
| citation | bfs | POPT | 3 | 0.9658 | [0.932, 0.994] | +3.42% | improvement |
| citation | bfs | SRRIP | 3 | 0.9965 | [0.995, 0.998] | +0.35% | improvement |
| citation | cc | GRASP | 3 | 0.8314 | [0.800, 0.894] | +16.86% | improvement |
| citation | cc | POPT | 3 | 0.8705 | [0.862, 0.883] | +12.95% | improvement |
| citation | cc | SRRIP | 3 | 0.9189 | [0.888, 0.939] | +8.11% | improvement |
| citation | pr | GRASP | 3 | 0.7851 | [0.689, 0.914] | +21.49% | improvement |
| citation | pr | POPT | 3 | 0.7082 | [0.631, 0.850] | +29.18% | improvement |
| citation | pr | SRRIP | 3 | 0.9153 | [0.842, 0.983] | +8.47% | improvement |
| citation | sssp | GRASP | 3 | 0.8887 | [0.843, 0.951] | +11.13% | improvement |
| citation | sssp | POPT | 3 | 1.0031 | [0.975, 1.056] | -0.31% | — |
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
| social | bc | POPT | 12 | 0.9952 | [0.973, 1.024] | +0.48% | — |
| social | bc | SRRIP | 12 | 0.9634 | [0.945, 0.980] | +3.66% | improvement |
| social | bfs | GRASP | 12 | 0.9648 | [0.943, 0.984] | +3.52% | improvement |
| social | bfs | POPT | 12 | 0.9591 | [0.927, 0.985] | +4.09% | improvement |
| social | bfs | SRRIP | 12 | 0.9889 | [0.981, 0.996] | +1.11% | improvement |
| social | cc | GRASP | 9 | 1.0508 | [0.813, 1.655] | -5.08% | — |
| social | cc | POPT | 9 | 0.8908 | [0.844, 0.939] | +10.92% | improvement |
| social | cc | SRRIP | 9 | 0.9317 | [0.889, 0.970] | +6.83% | improvement |
| social | pr | GRASP | 12 | 0.8585 | [0.806, 0.913] | +14.15% | improvement |
| social | pr | POPT | 12 | 0.8299 | [0.764, 0.899] | +17.01% | improvement |
| social | pr | SRRIP | 12 | 0.9190 | [0.883, 0.954] | +8.10% | improvement |
| social | sssp | GRASP | 9 | 0.9620 | [0.830, 1.222] | +3.80% | — |
| social | sssp | POPT | 9 | 1.0676 | [0.964, 1.236] | -6.76% | — |
| social | sssp | SRRIP | 9 | 0.9300 | [0.905, 0.956] | +7.00% | improvement |
| web | bc | GRASP | 3 | 1.0712 | [1.005, 1.192] | -7.12% | regression |
| web | bc | POPT | 3 | 1.2223 | [1.044, 1.549] | -22.23% | regression |
| web | bc | SRRIP | 3 | 0.9759 | [0.957, 1.005] | +2.41% | — |
| web | bfs | GRASP | 3 | 1.0130 | [0.963, 1.114] | -1.30% | — |
| web | bfs | POPT | 3 | 0.9091 | [0.842, 0.981] | +9.09% | improvement |
| web | bfs | SRRIP | 3 | 0.9970 | [0.993, 0.999] | +0.30% | improvement |
| web | cc | GRASP | 3 | 2.4541 | [0.923, 4.974] | -145.41% | — |
| web | cc | POPT | 3 | 0.9680 | [0.907, 1.000] | +3.20% | — |
| web | cc | SRRIP | 3 | 0.9832 | [0.950, 1.000] | +1.68% | — |
| web | pr | GRASP | 3 | 0.9097 | [0.754, 1.100] | +9.03% | — |
| web | pr | POPT | 3 | 0.8310 | [0.718, 0.902] | +16.91% | improvement |
| web | pr | SRRIP | 3 | 0.8998 | [0.839, 0.961] | +10.02% | improvement |
| web | sssp | GRASP | 3 | 1.8937 | [0.965, 3.476] | -89.37% | — |
| web | sssp | POPT | 3 | 1.0113 | [1.000, 1.034] | -1.13% | — |
| web | sssp | SRRIP | 3 | 0.9962 | [0.989, 1.000] | +0.38% | — |
