# Wilson 95% CIs on policy win-counts

Source: `wiki/data/oracle_gap.json`.
Z = 1.959964 (two-sided 95%).

## Overall (all cells)

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 10 / 114 | 0.088 | [0.048, 0.154] |
| SRRIP | 14 / 114 | 0.123 | [0.075, 0.196] |
| GRASP | 60 / 114 | 0.526 | [0.435, 0.616] |
| POPT | 44 / 114 | 0.386 | [0.302, 0.478] |

## Per-app

### pr

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 3 / 28 | 0.107 | [0.037, 0.272] | minority |
| SRRIP | 3 / 28 | 0.107 | [0.037, 0.272] | minority |
| GRASP | 9 / 28 | 0.321 | [0.179, 0.507] | - |
| POPT | 22 / 28 | 0.786 | [0.605, 0.898] | majority, above-chance |

### bc

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 1 / 23 | 0.043 | [0.008, 0.210] | below-chance, minority |
| SRRIP | 7 / 23 | 0.304 | [0.156, 0.509] | - |
| GRASP | 15 / 23 | 0.652 | [0.449, 0.812] | above-chance |
| POPT | 5 / 23 | 0.217 | [0.097, 0.419] | minority |

### cc

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 2 / 20 | 0.100 | [0.028, 0.301] | minority |
| SRRIP | 1 / 20 | 0.050 | [0.009, 0.236] | below-chance, minority |
| GRASP | 17 / 20 | 0.850 | [0.640, 0.948] | majority, above-chance |
| POPT | 0 / 20 | 0.000 | [0.000, 0.161] | below-chance, minority |

### bfs

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 1 / 23 | 0.043 | [0.008, 0.210] | below-chance, minority |
| SRRIP | 1 / 23 | 0.043 | [0.008, 0.210] | below-chance, minority |
| GRASP | 12 / 23 | 0.522 | [0.330, 0.708] | above-chance |
| POPT | 9 / 23 | 0.391 | [0.222, 0.592] | - |

### sssp

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 3 / 20 | 0.150 | [0.052, 0.360] | minority |
| SRRIP | 2 / 20 | 0.100 | [0.028, 0.301] | minority |
| GRASP | 7 / 20 | 0.350 | [0.181, 0.567] | - |
| POPT | 8 / 20 | 0.400 | [0.219, 0.613] | - |

## Per-family

### citation

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 0 / 15 | 0.000 | [0.000, 0.204] |
| SRRIP | 1 / 15 | 0.067 | [0.012, 0.298] |
| GRASP | 11 / 15 | 0.733 | [0.480, 0.891] |
| POPT | 3 / 15 | 0.200 | [0.070, 0.452] |

### mesh

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 0 / 5 | 0.000 | [0.000, 0.434] |
| SRRIP | 0 / 5 | 0.000 | [0.000, 0.434] |
| GRASP | 2 / 5 | 0.400 | [0.118, 0.769] |
| POPT | 3 / 5 | 0.600 | [0.231, 0.882] |

### road

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 4 / 25 | 0.160 | [0.064, 0.346] |
| SRRIP | 3 / 25 | 0.120 | [0.042, 0.300] |
| GRASP | 6 / 25 | 0.240 | [0.115, 0.434] |
| POPT | 12 / 25 | 0.480 | [0.300, 0.665] |

### social

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 4 / 54 | 0.074 | [0.029, 0.175] |
| SRRIP | 8 / 54 | 0.148 | [0.077, 0.266] |
| GRASP | 35 / 54 | 0.648 | [0.515, 0.762] |
| POPT | 21 / 54 | 0.389 | [0.270, 0.522] |

### web

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 2 / 15 | 0.133 | [0.037, 0.379] |
| SRRIP | 2 / 15 | 0.133 | [0.037, 0.379] |
| GRASP | 6 / 15 | 0.400 | [0.198, 0.642] |
| POPT | 5 / 15 | 0.333 | [0.152, 0.583] |
