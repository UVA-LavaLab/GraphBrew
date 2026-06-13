# Cohen's h on policy win-rate gaps

Source: `wiki/data/oracle_gap.json` (456 rows).

Thresholds (Cohen 1988): small ≥ 0.2, medium ≥ 0.5, large ≥ 0.8.

## Largest effect per kernel

| App | Favors | Comparison | p_a | p_b | h | magnitude |
| --- | --- | --- | ---: | ---: | ---: | --- |
| pr | POPT | LRU vs POPT | 0.107 | 0.750 | 1.427 | **large** |
| bc | GRASP | LRU vs GRASP | 0.217 | 0.739 | 1.099 | **large** |
| cc | POPT | LRU vs POPT | 0.150 | 0.550 | 0.876 | **large** |
| bfs | GRASP | SRRIP vs GRASP | 0.130 | 0.652 | 1.141 | **large** |
| sssp | GRASP | GRASP vs POPT | 0.700 | 0.150 | 1.187 | **large** |

## All large-effect dominance pairs (h ≥ 0.8, p_a > p_b)

| App | Winner | Loser | p_winner | p_loser | h |
| --- | --- | --- | ---: | ---: | ---: |
| pr | POPT | LRU | 0.750 | 0.107 | 1.427 |
| pr | POPT | SRRIP | 0.750 | 0.143 | 1.319 |
| sssp | GRASP | POPT | 0.700 | 0.150 | 1.187 |
| bfs | GRASP | SRRIP | 0.652 | 0.130 | 1.141 |
| bc | GRASP | LRU | 0.739 | 0.217 | 1.099 |
| bc | GRASP | SRRIP | 0.739 | 0.217 | 1.099 |
| bc | GRASP | POPT | 0.739 | 0.217 | 1.099 |
| sssp | GRASP | LRU | 0.700 | 0.200 | 1.055 |
| bfs | GRASP | LRU | 0.652 | 0.174 | 1.020 |
| sssp | GRASP | SRRIP | 0.700 | 0.250 | 0.935 |
| pr | POPT | GRASP | 0.750 | 0.321 | 0.889 |
| cc | POPT | LRU | 0.550 | 0.150 | 0.876 |
| cc | POPT | SRRIP | 0.550 | 0.150 | 0.876 |

## Per-app full table

### pr

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 3 / 28 | 0.107 |
| SRRIP | 4 / 28 | 0.143 |
| GRASP | 9 / 28 | 0.321 |
| POPT | 21 / 28 | 0.750 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | -0.036 | 0.108 | negligible |
| LRU | GRASP | -0.214 | 0.539 | medium |
| LRU | POPT | -0.643 | 1.427 | large |
| SRRIP | LRU | +0.036 | 0.108 | negligible |
| SRRIP | GRASP | -0.179 | 0.430 | small |
| SRRIP | POPT | -0.607 | 1.319 | large |
| GRASP | LRU | +0.214 | 0.539 | medium |
| GRASP | SRRIP | +0.179 | 0.430 | small |
| GRASP | POPT | -0.429 | 0.889 | large |
| POPT | LRU | +0.643 | 1.427 | large |
| POPT | SRRIP | +0.607 | 1.319 | large |
| POPT | GRASP | +0.429 | 0.889 | large |

### bc

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 5 / 23 | 0.217 |
| SRRIP | 5 / 23 | 0.217 |
| GRASP | 17 / 23 | 0.739 |
| POPT | 5 / 23 | 0.217 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | +0.000 | 0.000 | negligible |
| LRU | GRASP | -0.522 | 1.099 | large |
| LRU | POPT | +0.000 | 0.000 | negligible |
| SRRIP | LRU | +0.000 | 0.000 | negligible |
| SRRIP | GRASP | -0.522 | 1.099 | large |
| SRRIP | POPT | +0.000 | 0.000 | negligible |
| GRASP | LRU | +0.522 | 1.099 | large |
| GRASP | SRRIP | +0.522 | 1.099 | large |
| GRASP | POPT | +0.522 | 1.099 | large |
| POPT | LRU | +0.000 | 0.000 | negligible |
| POPT | SRRIP | +0.000 | 0.000 | negligible |
| POPT | GRASP | -0.522 | 1.099 | large |

### cc

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 3 / 20 | 0.150 |
| SRRIP | 3 / 20 | 0.150 |
| GRASP | 9 / 20 | 0.450 |
| POPT | 11 / 20 | 0.550 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | +0.000 | 0.000 | negligible |
| LRU | GRASP | -0.300 | 0.675 | medium |
| LRU | POPT | -0.400 | 0.876 | large |
| SRRIP | LRU | +0.000 | 0.000 | negligible |
| SRRIP | GRASP | -0.300 | 0.675 | medium |
| SRRIP | POPT | -0.400 | 0.876 | large |
| GRASP | LRU | +0.300 | 0.675 | medium |
| GRASP | SRRIP | +0.300 | 0.675 | medium |
| GRASP | POPT | -0.100 | 0.200 | small |
| POPT | LRU | +0.400 | 0.876 | large |
| POPT | SRRIP | +0.400 | 0.876 | large |
| POPT | GRASP | +0.100 | 0.200 | small |

### bfs

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 4 / 23 | 0.174 |
| SRRIP | 3 / 23 | 0.130 |
| GRASP | 15 / 23 | 0.652 |
| POPT | 10 / 23 | 0.435 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | +0.043 | 0.121 | negligible |
| LRU | GRASP | -0.478 | 1.020 | large |
| LRU | POPT | -0.261 | 0.580 | medium |
| SRRIP | LRU | -0.043 | 0.121 | negligible |
| SRRIP | GRASP | -0.522 | 1.141 | large |
| SRRIP | POPT | -0.304 | 0.701 | medium |
| GRASP | LRU | +0.478 | 1.020 | large |
| GRASP | SRRIP | +0.522 | 1.141 | large |
| GRASP | POPT | +0.217 | 0.440 | small |
| POPT | LRU | +0.261 | 0.580 | medium |
| POPT | SRRIP | +0.304 | 0.701 | medium |
| POPT | GRASP | -0.217 | 0.440 | small |

### sssp

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 4 / 20 | 0.200 |
| SRRIP | 5 / 20 | 0.250 |
| GRASP | 14 / 20 | 0.700 |
| POPT | 3 / 20 | 0.150 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | -0.050 | 0.120 | negligible |
| LRU | GRASP | -0.500 | 1.055 | large |
| LRU | POPT | +0.050 | 0.132 | negligible |
| SRRIP | LRU | +0.050 | 0.120 | negligible |
| SRRIP | GRASP | -0.450 | 0.935 | large |
| SRRIP | POPT | +0.100 | 0.252 | small |
| GRASP | LRU | +0.500 | 1.055 | large |
| GRASP | SRRIP | +0.450 | 0.935 | large |
| GRASP | POPT | +0.550 | 1.187 | large |
| POPT | LRU | -0.050 | 0.132 | negligible |
| POPT | SRRIP | -0.100 | 0.252 | small |
| POPT | GRASP | -0.550 | 1.187 | large |
