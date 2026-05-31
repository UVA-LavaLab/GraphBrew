# Family × App × Policy: geomean improvement vs LRU

Source: `wiki/data/oracle_gap.json`  •  L3 scope: 1MB, 4MB, 8MB
Bootstrap: B=2000, seed=1729, α=0.05 (percentile CI)

Records: **63** (34 CI-strict improvements vs LRU, 0 CI-strict regressions).

## Headline CI-strict improvements (≥10% miss-rate reduction)

| Family | App | Policy | n | Geomean | CI ratio | Improve%  (CI) |
|---|---|---|---|---|---|---|
| citation | pr | POPT | 3 | 0.6710 | [0.562, 0.862] | +32.90% [+13.77, +43.80] |
| citation | cc | GRASP | 3 | 0.7439 | [0.645, 0.818] | +25.61% [+18.18, +35.50] |
| citation | pr | GRASP | 3 | 0.7633 | [0.658, 0.878] | +23.67% [+12.18, +34.24] |
| social | cc | GRASP | 9 | 0.7664 | [0.661, 0.874] | +23.36% [+12.59, +33.90] |
| web | pr | GRASP | 3 | 0.7887 | [0.747, 0.871] | +21.13% [+12.95, +25.27] |
| web | pr | POPT | 3 | 0.7908 | [0.694, 0.867] | +20.92% [+13.31, +30.62] |
| social | pr | POPT | 12 | 0.7961 | [0.722, 0.877] | +20.39% [+12.33, +27.81] |
| citation | cc | POPT | 3 | 0.8539 | [0.729, 0.954] | +14.61% [+4.61, +27.09] |
| social | pr | GRASP | 12 | 0.8554 | [0.792, 0.920] | +14.46% [+8.04, +20.85] |
| citation | sssp | GRASP | 3 | 0.8556 | [0.761, 0.964] | +14.44% [+3.61, +23.93] |
| social | sssp | GRASP | 9 | 0.8608 | [0.795, 0.929] | +13.92% [+7.13, +20.50] |
| social | cc | POPT | 9 | 0.8763 | [0.790, 0.958] | +12.37% [+4.22, +21.01] |
| social | sssp | POPT | 9 | 0.8826 | [0.836, 0.931] | +11.73% [+6.94, +16.41] |
| citation | sssp | POPT | 3 | 0.8855 | [0.816, 0.967] | +11.45% [+3.28, +18.41] |
| web | bfs | POPT | 3 | 0.8917 | [0.806, 0.976] | +10.83% [+2.44, +19.37] |
| web | bfs | GRASP | 3 | 0.8992 | [0.808, 0.965] | +10.08% [+3.52, +19.17] |

## CI-strict regressions (geomean ratio > 1.0, CI lo > 1.0)

_None — no policy CI-strictly regresses on any (family, app)._

## All records (sorted by family, app, policy)

| Family | App | Policy | n | Geomean ratio | CI ratio | Improve% | CI-strict |
|---|---|---|---|---|---|---|---|
| citation | bc | GRASP | 3 | 0.9679 | [0.918, 1.016] | +3.21% | — |
| citation | bc | POPT | 3 | 0.9721 | [0.940, 0.998] | +2.79% | improvement |
| citation | bc | SRRIP | 3 | 0.9842 | [0.970, 0.995] | +1.58% | improvement |
| citation | bfs | GRASP | 3 | 0.9481 | [0.904, 0.988] | +5.19% | improvement |
| citation | bfs | POPT | 3 | 0.9545 | [0.912, 0.989] | +4.55% | improvement |
| citation | bfs | SRRIP | 3 | 0.9959 | [0.995, 0.997] | +0.41% | improvement |
| citation | cc | GRASP | 3 | 0.7439 | [0.645, 0.818] | +25.61% | improvement |
| citation | cc | POPT | 3 | 0.8539 | [0.729, 0.954] | +14.61% | improvement |
| citation | cc | SRRIP | 3 | 0.9144 | [0.882, 0.934] | +8.56% | improvement |
| citation | pr | GRASP | 3 | 0.7633 | [0.658, 0.878] | +23.67% | improvement |
| citation | pr | POPT | 3 | 0.6710 | [0.562, 0.862] | +32.90% | improvement |
| citation | pr | SRRIP | 3 | 0.9169 | [0.845, 0.984] | +8.31% | improvement |
| citation | sssp | GRASP | 3 | 0.8556 | [0.761, 0.964] | +14.44% | improvement |
| citation | sssp | POPT | 3 | 0.8855 | [0.816, 0.967] | +11.45% | improvement |
| citation | sssp | SRRIP | 3 | 0.9538 | [0.912, 0.995] | +4.62% | improvement |
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
| social | bc | GRASP | 12 | 0.9130 | [0.880, 0.946] | +8.70% | improvement |
| social | bc | POPT | 12 | 0.9919 | [0.914, 1.118] | +0.81% | — |
| social | bc | SRRIP | 12 | 0.9581 | [0.941, 0.975] | +4.19% | improvement |
| social | bfs | GRASP | 12 | 1.0773 | [0.917, 1.440] | -7.74% | — |
| social | bfs | POPT | 12 | 0.9480 | [0.916, 0.977] | +5.20% | improvement |
| social | bfs | SRRIP | 12 | 0.9892 | [0.979, 0.998] | +1.08% | improvement |
| social | cc | GRASP | 9 | 0.7664 | [0.661, 0.874] | +23.36% | improvement |
| social | cc | POPT | 9 | 0.8763 | [0.790, 0.958] | +12.37% | improvement |
| social | cc | SRRIP | 9 | 0.9256 | [0.868, 0.975] | +7.44% | improvement |
| social | pr | GRASP | 12 | 0.8554 | [0.792, 0.920] | +14.46% | improvement |
| social | pr | POPT | 12 | 0.7961 | [0.722, 0.877] | +20.39% | improvement |
| social | pr | SRRIP | 12 | 0.9193 | [0.886, 0.953] | +8.07% | improvement |
| social | sssp | GRASP | 9 | 0.8608 | [0.795, 0.929] | +13.92% | improvement |
| social | sssp | POPT | 9 | 0.8826 | [0.836, 0.931] | +11.73% | improvement |
| social | sssp | SRRIP | 9 | 0.9268 | [0.902, 0.954] | +7.32% | improvement |
| web | bc | GRASP | 3 | 0.9493 | [0.910, 1.012] | +5.07% | — |
| web | bc | POPT | 3 | 1.0298 | [1.000, 1.086] | -2.98% | — |
| web | bc | SRRIP | 3 | 0.9693 | [0.961, 0.983] | +3.07% | improvement |
| web | bfs | GRASP | 3 | 0.8992 | [0.808, 0.965] | +10.08% | improvement |
| web | bfs | POPT | 3 | 0.8917 | [0.806, 0.976] | +10.83% | improvement |
| web | bfs | SRRIP | 3 | 0.9958 | [0.990, 0.999] | +0.42% | improvement |
| web | cc | GRASP | 3 | 0.9589 | [0.882, 1.001] | +4.11% | — |
| web | cc | POPT | 3 | 0.9672 | [0.904, 1.001] | +3.28% | — |
| web | cc | SRRIP | 3 | 0.9766 | [0.931, 1.000] | +2.34% | — |
| web | pr | GRASP | 3 | 0.7887 | [0.747, 0.871] | +21.13% | improvement |
| web | pr | POPT | 3 | 0.7908 | [0.694, 0.867] | +20.92% | improvement |
| web | pr | SRRIP | 3 | 0.9017 | [0.841, 0.962] | +9.83% | improvement |
| web | sssp | GRASP | 3 | 0.9999 | [0.998, 1.001] | +0.01% | — |
| web | sssp | POPT | 3 | 0.9947 | [0.985, 1.001] | +0.53% | — |
| web | sssp | SRRIP | 3 | 0.9967 | [0.991, 1.001] | +0.33% | — |
