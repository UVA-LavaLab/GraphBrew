# Wilson 95% CIs on policy win-counts

Source: `wiki/data/oracle_gap.json`.
Z = 1.959964 (two-sided 95%).

## Overall (all cells)

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 19 / 114 | 0.167 | [0.109, 0.246] |
| SRRIP | 20 / 114 | 0.175 | [0.117, 0.256] |
| GRASP | 64 / 114 | 0.561 | [0.470, 0.649] |
| POPT | 50 / 114 | 0.439 | [0.351, 0.530] |

## Per-app

### pr

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 3 / 28 | 0.107 | [0.037, 0.272] | minority |
| SRRIP | 4 / 28 | 0.143 | [0.057, 0.315] | minority |
| GRASP | 9 / 28 | 0.321 | [0.179, 0.507] | - |
| POPT | 21 / 28 | 0.750 | [0.566, 0.873] | majority, above-chance |

### bc

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 5 / 23 | 0.217 | [0.097, 0.419] | minority |
| SRRIP | 5 / 23 | 0.217 | [0.097, 0.419] | minority |
| GRASP | 17 / 23 | 0.739 | [0.535, 0.875] | majority, above-chance |
| POPT | 5 / 23 | 0.217 | [0.097, 0.419] | minority |

### cc

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 3 / 20 | 0.150 | [0.052, 0.360] | minority |
| SRRIP | 3 / 20 | 0.150 | [0.052, 0.360] | minority |
| GRASP | 9 / 20 | 0.450 | [0.258, 0.658] | above-chance |
| POPT | 11 / 20 | 0.550 | [0.342, 0.742] | above-chance |

### bfs

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 4 / 23 | 0.174 | [0.070, 0.371] | minority |
| SRRIP | 3 / 23 | 0.130 | [0.045, 0.321] | minority |
| GRASP | 15 / 23 | 0.652 | [0.449, 0.812] | above-chance |
| POPT | 10 / 23 | 0.435 | [0.256, 0.632] | above-chance |

### sssp

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 4 / 20 | 0.200 | [0.081, 0.416] | minority |
| SRRIP | 5 / 20 | 0.250 | [0.112, 0.469] | minority |
| GRASP | 14 / 20 | 0.700 | [0.481, 0.855] | above-chance |
| POPT | 3 / 20 | 0.150 | [0.052, 0.360] | minority |

## Per-family

### citation

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 0 / 15 | 0.000 | [0.000, 0.204] |
| SRRIP | 0 / 15 | 0.000 | [0.000, 0.204] |
| GRASP | 10 / 15 | 0.667 | [0.417, 0.848] |
| POPT | 5 / 15 | 0.333 | [0.152, 0.583] |

### mesh

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 0 / 5 | 0.000 | [0.000, 0.434] |
| SRRIP | 0 / 5 | 0.000 | [0.000, 0.434] |
| GRASP | 1 / 5 | 0.200 | [0.036, 0.625] |
| POPT | 4 / 5 | 0.800 | [0.376, 0.964] |

### road

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 3 / 25 | 0.120 | [0.042, 0.300] |
| SRRIP | 1 / 25 | 0.040 | [0.007, 0.195] |
| GRASP | 12 / 25 | 0.480 | [0.300, 0.665] |
| POPT | 9 / 25 | 0.360 | [0.203, 0.555] |

### social

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 11 / 54 | 0.204 | [0.118, 0.329] |
| SRRIP | 12 / 54 | 0.222 | [0.132, 0.349] |
| GRASP | 39 / 54 | 0.722 | [0.591, 0.824] |
| POPT | 23 / 54 | 0.426 | [0.303, 0.558] |

### web

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 5 / 15 | 0.333 | [0.152, 0.583] |
| SRRIP | 7 / 15 | 0.467 | [0.248, 0.699] |
| GRASP | 2 / 15 | 0.133 | [0.037, 0.379] |
| POPT | 9 / 15 | 0.600 | [0.357, 0.802] |
