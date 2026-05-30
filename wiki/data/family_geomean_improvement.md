# Family × App × Policy: geomean improvement vs LRU

Source: `wiki/data/oracle_gap.json`  •  L3 scope: 1MB, 4MB, 8MB
Bootstrap: B=2000, seed=1729, α=0.05 (percentile CI)

Records: **63** (34 CI-strict improvements vs LRU, 0 CI-strict regressions).

## Headline CI-strict improvements (≥10% miss-rate reduction)

| Family | App | Policy | n | Geomean | CI ratio | Improve%  (CI) |
|---|---|---|---|---|---|---|
| citation | pr | POPT | 3 | 0.6804 | [0.573, 0.869] | +31.96% [+13.11, +42.67] |
| citation | cc | GRASP | 3 | 0.7439 | [0.645, 0.818] | +25.61% [+18.18, +35.50] |
| citation | pr | GRASP | 3 | 0.7660 | [0.661, 0.881] | +23.40% [+11.93, +33.95] |
| social | cc | GRASP | 9 | 0.7664 | [0.661, 0.874] | +23.36% [+12.59, +33.90] |
| web | pr | GRASP | 3 | 0.7886 | [0.747, 0.871] | +21.14% [+12.91, +25.33] |
| web | pr | POPT | 3 | 0.7908 | [0.694, 0.867] | +20.92% [+13.31, +30.64] |
| social | pr | POPT | 12 | 0.7929 | [0.723, 0.866] | +20.71% [+13.36, +27.67] |
| citation | sssp | GRASP | 3 | 0.8450 | [0.743, 0.960] | +15.50% [+4.05, +25.67] |
| citation | cc | POPT | 3 | 0.8539 | [0.729, 0.954] | +14.61% [+4.61, +27.09] |
| social | pr | GRASP | 12 | 0.8753 | [0.794, 0.983] | +12.46% [+1.68, +20.56] |
| social | cc | POPT | 9 | 0.8763 | [0.790, 0.958] | +12.37% [+4.22, +21.01] |
| citation | sssp | POPT | 3 | 0.8813 | [0.804, 0.970] | +11.87% [+3.04, +19.60] |
| social | sssp | GRASP | 9 | 0.8831 | [0.813, 0.953] | +11.69% [+4.74, +18.70] |
| web | bfs | POPT | 3 | 0.8925 | [0.807, 0.977] | +10.75% [+2.34, +19.31] |
| social | sssp | POPT | 9 | 0.8958 | [0.851, 0.944] | +10.42% [+5.65, +14.94] |

## CI-strict regressions (geomean ratio > 1.0, CI lo > 1.0)

_None — no policy CI-strictly regresses on any (family, app)._

## All records (sorted by family, app, policy)

| Family | App | Policy | n | Geomean ratio | CI ratio | Improve% | CI-strict |
|---|---|---|---|---|---|---|---|
| citation | bc | GRASP | 3 | 0.9628 | [0.915, 1.004] | +3.72% | — |
| citation | bc | POPT | 3 | 0.9824 | [0.956, 0.998] | +1.76% | improvement |
| citation | bc | SRRIP | 3 | 0.9792 | [0.961, 0.993] | +2.08% | improvement |
| citation | bfs | GRASP | 3 | 0.9479 | [0.901, 0.990] | +5.21% | improvement |
| citation | bfs | POPT | 3 | 0.9524 | [0.909, 0.991] | +4.76% | improvement |
| citation | bfs | SRRIP | 3 | 0.9949 | [0.992, 0.996] | +0.51% | improvement |
| citation | cc | GRASP | 3 | 0.7439 | [0.645, 0.818] | +25.61% | improvement |
| citation | cc | POPT | 3 | 0.8539 | [0.729, 0.954] | +14.61% | improvement |
| citation | cc | SRRIP | 3 | 0.9144 | [0.882, 0.934] | +8.56% | improvement |
| citation | pr | GRASP | 3 | 0.7660 | [0.661, 0.881] | +23.40% | improvement |
| citation | pr | POPT | 3 | 0.6804 | [0.573, 0.869] | +31.96% | improvement |
| citation | pr | SRRIP | 3 | 0.9188 | [0.849, 0.984] | +8.12% | improvement |
| citation | sssp | GRASP | 3 | 0.8450 | [0.743, 0.960] | +15.50% | improvement |
| citation | sssp | POPT | 3 | 0.8813 | [0.804, 0.970] | +11.87% | improvement |
| citation | sssp | SRRIP | 3 | 0.9525 | [0.907, 0.996] | +4.75% | improvement |
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
| social | bc | GRASP | 12 | 0.9125 | [0.880, 0.945] | +8.75% | improvement |
| social | bc | POPT | 12 | 0.9934 | [0.917, 1.119] | +0.66% | — |
| social | bc | SRRIP | 12 | 0.9593 | [0.943, 0.976] | +4.07% | improvement |
| social | bfs | GRASP | 12 | 1.1019 | [0.940, 1.461] | -10.20% | — |
| social | bfs | POPT | 12 | 0.9597 | [0.929, 0.985] | +4.03% | improvement |
| social | bfs | SRRIP | 12 | 0.9955 | [0.992, 0.999] | +0.45% | improvement |
| social | cc | GRASP | 9 | 0.7664 | [0.661, 0.874] | +23.36% | improvement |
| social | cc | POPT | 9 | 0.8763 | [0.790, 0.958] | +12.37% | improvement |
| social | cc | SRRIP | 9 | 0.9256 | [0.868, 0.975] | +7.44% | improvement |
| social | pr | GRASP | 12 | 0.8753 | [0.794, 0.983] | +12.46% | improvement |
| social | pr | POPT | 12 | 0.7929 | [0.723, 0.866] | +20.71% | improvement |
| social | pr | SRRIP | 12 | 0.9129 | [0.884, 0.940] | +8.71% | improvement |
| social | sssp | GRASP | 9 | 0.8831 | [0.813, 0.953] | +11.69% | improvement |
| social | sssp | POPT | 9 | 0.8958 | [0.851, 0.944] | +10.42% | improvement |
| social | sssp | SRRIP | 9 | 0.9373 | [0.909, 0.966] | +6.27% | improvement |
| web | bc | GRASP | 3 | 0.9399 | [0.896, 1.000] | +6.01% | — |
| web | bc | POPT | 3 | 1.0308 | [0.993, 1.097] | -3.08% | — |
| web | bc | SRRIP | 3 | 0.9655 | [0.953, 0.974] | +3.45% | improvement |
| web | bfs | GRASP | 3 | 0.9005 | [0.811, 0.965] | +9.95% | improvement |
| web | bfs | POPT | 3 | 0.8925 | [0.807, 0.977] | +10.75% | improvement |
| web | bfs | SRRIP | 3 | 0.9974 | [0.994, 0.999] | +0.26% | improvement |
| web | cc | GRASP | 3 | 0.9589 | [0.882, 1.001] | +4.11% | — |
| web | cc | POPT | 3 | 0.9672 | [0.904, 1.001] | +3.28% | — |
| web | cc | SRRIP | 3 | 0.9766 | [0.931, 1.000] | +2.34% | — |
| web | pr | GRASP | 3 | 0.7886 | [0.747, 0.871] | +21.14% | improvement |
| web | pr | POPT | 3 | 0.7908 | [0.694, 0.867] | +20.92% | improvement |
| web | pr | SRRIP | 3 | 0.9016 | [0.842, 0.962] | +9.84% | improvement |
| web | sssp | GRASP | 3 | 0.9999 | [0.998, 1.001] | +0.01% | — |
| web | sssp | POPT | 3 | 0.9947 | [0.985, 1.001] | +0.53% | — |
| web | sssp | SRRIP | 3 | 0.9967 | [0.991, 1.001] | +0.33% | — |
