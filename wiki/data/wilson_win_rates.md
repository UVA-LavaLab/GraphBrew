# Wilson 95% CIs on policy win-counts

Source: `wiki/data/oracle_gap.json`.
Z = 1.959964 (two-sided 95%).

## Overall (all cells)

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 19 / 114 | 0.167 | [0.109, 0.246] |
| SRRIP | 19 / 114 | 0.167 | [0.109, 0.246] |
| GRASP | 58 / 114 | 0.509 | [0.418, 0.599] |
| POPT | 60 / 114 | 0.526 | [0.435, 0.616] |

## Per-app

### pr

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 3 / 28 | 0.107 | [0.037, 0.272] | minority |
| SRRIP | 3 / 28 | 0.107 | [0.037, 0.272] | minority |
| GRASP | 7 / 28 | 0.250 | [0.127, 0.434] | minority |
| POPT | 24 / 28 | 0.857 | [0.685, 0.943] | majority, above-chance |

### bc

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 5 / 23 | 0.217 | [0.097, 0.419] | minority |
| SRRIP | 5 / 23 | 0.217 | [0.097, 0.419] | minority |
| GRASP | 12 / 23 | 0.522 | [0.330, 0.708] | above-chance |
| POPT | 10 / 23 | 0.435 | [0.256, 0.632] | above-chance |

### cc

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 3 / 20 | 0.150 | [0.052, 0.360] | minority |
| SRRIP | 3 / 20 | 0.150 | [0.052, 0.360] | minority |
| GRASP | 17 / 20 | 0.850 | [0.640, 0.948] | majority, above-chance |
| POPT | 6 / 20 | 0.300 | [0.145, 0.519] | - |

### bfs

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 4 / 23 | 0.174 | [0.070, 0.371] | minority |
| SRRIP | 3 / 23 | 0.130 | [0.045, 0.321] | minority |
| GRASP | 12 / 23 | 0.522 | [0.330, 0.708] | above-chance |
| POPT | 13 / 23 | 0.565 | [0.368, 0.744] | above-chance |

### sssp

| Policy | Wins / N | p̂ | 95% Wilson CI | CI-strict |
| --- | ---: | ---: | --- | --- |
| LRU | 4 / 20 | 0.200 | [0.081, 0.416] | minority |
| SRRIP | 5 / 20 | 0.250 | [0.112, 0.469] | minority |
| GRASP | 10 / 20 | 0.500 | [0.299, 0.701] | above-chance |
| POPT | 7 / 20 | 0.350 | [0.181, 0.567] | - |

## Per-family

### citation

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 0 / 15 | 0.000 | [0.000, 0.204] |
| SRRIP | 0 / 15 | 0.000 | [0.000, 0.204] |
| GRASP | 7 / 15 | 0.467 | [0.248, 0.699] |
| POPT | 8 / 15 | 0.533 | [0.301, 0.752] |

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
| GRASP | 14 / 25 | 0.560 | [0.371, 0.733] |
| POPT | 7 / 25 | 0.280 | [0.143, 0.476] |

### social

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 11 / 54 | 0.204 | [0.118, 0.329] |
| SRRIP | 12 / 54 | 0.222 | [0.132, 0.349] |
| GRASP | 32 / 54 | 0.593 | [0.460, 0.713] |
| POPT | 31 / 54 | 0.574 | [0.442, 0.697] |

### web

| Policy | Wins / N | p̂ | 95% Wilson CI |
| --- | ---: | ---: | --- |
| LRU | 5 / 15 | 0.333 | [0.152, 0.583] |
| SRRIP | 6 / 15 | 0.400 | [0.198, 0.642] |
| GRASP | 4 / 15 | 0.267 | [0.109, 0.519] |
| POPT | 10 / 15 | 0.667 | [0.417, 0.848] |
