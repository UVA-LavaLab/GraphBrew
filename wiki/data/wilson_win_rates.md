# Wilson 95% CIs on policy win-counts

Source: `wiki/data/oracle_gap.json`.
Z = 1.959964 (two-sided 95%).

## Overall (all cells)

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 7 / 114 | 0.061 | [0.030, 0.121] |
| SRRIP | 11 / 114 | 0.097 | [0.055, 0.165] |
| GRASP | 56 / 114 | 0.491 | [0.401, 0.582] |
| POPT | 45 / 114 | 0.395 | [0.310, 0.486] |

## Per-app

### pr

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 0 / 28 | 0.000 | [0.000, 0.121] | below-chance, minority |
| SRRIP | 0 / 28 | 0.000 | [0.000, 0.121] | below-chance, minority |
| GRASP | 8 / 28 | 0.286 | [0.152, 0.471] | minority |
| POPT | 20 / 28 | 0.714 | [0.529, 0.848] | majority, above-chance |

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
| GRASP | 9 / 23 | 0.391 | [0.222, 0.592] | - |
| POPT | 12 / 23 | 0.522 | [0.330, 0.708] | above-chance |

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
| LRU | 1 / 54 | 0.018 | [0.003, 0.098] |
| SRRIP | 5 / 54 | 0.093 | [0.040, 0.199] |
| GRASP | 31 / 54 | 0.574 | [0.442, 0.697] |
| POPT | 22 / 54 | 0.407 | [0.287, 0.540] |

### web

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 2 / 15 | 0.133 | [0.037, 0.379] |
| SRRIP | 2 / 15 | 0.133 | [0.037, 0.379] |
| GRASP | 6 / 15 | 0.400 | [0.198, 0.642] |
| POPT | 5 / 15 | 0.333 | [0.152, 0.583] |
