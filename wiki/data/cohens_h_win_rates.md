# Cohen's h on policy win-rate gaps

Source: `wiki/data/oracle_gap.json` (456 rows).

Thresholds (Cohen 1988): small ≥ 0.2, medium ≥ 0.5, large ≥ 0.8.

## Largest effect per kernel

| App | Favors | Comparison | p_a | p_b | h | magnitude |
| --- | --- | --- | ---: | ---: | ---: | --- |
| pr | POPT | LRU vs POPT | 0.107 | 0.857 | 1.700 | **large** |
| bc | GRASP | LRU vs GRASP | 0.217 | 0.522 | 0.644 | **medium** |
| cc | GRASP | LRU vs GRASP | 0.150 | 0.850 | 1.551 | **large** |
| bfs | POPT | SRRIP vs POPT | 0.130 | 0.565 | 0.963 | **large** |
| sssp | GRASP | LRU vs GRASP | 0.200 | 0.500 | 0.643 | **medium** |

## All large-effect dominance pairs (h ≥ 0.8, p_a > p_b)

| App | Winner | Loser | p_winner | p_loser | h |
| --- | --- | --- | ---: | ---: | ---: |
| pr | POPT | LRU | 0.857 | 0.107 | 1.700 |
| pr | POPT | SRRIP | 0.857 | 0.107 | 1.700 |
| cc | GRASP | LRU | 0.850 | 0.150 | 1.551 |
| cc | GRASP | SRRIP | 0.850 | 0.150 | 1.551 |
| pr | POPT | GRASP | 0.857 | 0.250 | 1.319 |
| cc | GRASP | POPT | 0.850 | 0.300 | 1.187 |
| bfs | POPT | SRRIP | 0.565 | 0.130 | 0.963 |
| bfs | GRASP | SRRIP | 0.522 | 0.130 | 0.875 |
| bfs | POPT | LRU | 0.565 | 0.174 | 0.841 |

## Per-app full table

### pr

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 3 / 28 | 0.107 |
| SRRIP | 3 / 28 | 0.107 |
| GRASP | 7 / 28 | 0.250 |
| POPT | 24 / 28 | 0.857 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | +0.000 | 0.000 | negligible |
| LRU | GRASP | -0.143 | 0.380 | small |
| LRU | POPT | -0.750 | 1.700 | large |
| SRRIP | LRU | +0.000 | 0.000 | negligible |
| SRRIP | GRASP | -0.143 | 0.380 | small |
| SRRIP | POPT | -0.750 | 1.700 | large |
| GRASP | LRU | +0.143 | 0.380 | small |
| GRASP | SRRIP | +0.143 | 0.380 | small |
| GRASP | POPT | -0.607 | 1.319 | large |
| POPT | LRU | +0.750 | 1.700 | large |
| POPT | SRRIP | +0.750 | 1.700 | large |
| POPT | GRASP | +0.607 | 1.319 | large |

### bc

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 5 / 23 | 0.217 |
| SRRIP | 5 / 23 | 0.217 |
| GRASP | 12 / 23 | 0.522 |
| POPT | 10 / 23 | 0.435 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | +0.000 | 0.000 | negligible |
| LRU | GRASP | -0.304 | 0.644 | medium |
| LRU | POPT | -0.217 | 0.470 | small |
| SRRIP | LRU | +0.000 | 0.000 | negligible |
| SRRIP | GRASP | -0.304 | 0.644 | medium |
| SRRIP | POPT | -0.217 | 0.470 | small |
| GRASP | LRU | +0.304 | 0.644 | medium |
| GRASP | SRRIP | +0.304 | 0.644 | medium |
| GRASP | POPT | +0.087 | 0.174 | negligible |
| POPT | LRU | +0.217 | 0.470 | small |
| POPT | SRRIP | +0.217 | 0.470 | small |
| POPT | GRASP | -0.087 | 0.174 | negligible |

### cc

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 3 / 20 | 0.150 |
| SRRIP | 3 / 20 | 0.150 |
| GRASP | 17 / 20 | 0.850 |
| POPT | 6 / 20 | 0.300 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | +0.000 | 0.000 | negligible |
| LRU | GRASP | -0.700 | 1.551 | large |
| LRU | POPT | -0.150 | 0.364 | small |
| SRRIP | LRU | +0.000 | 0.000 | negligible |
| SRRIP | GRASP | -0.700 | 1.551 | large |
| SRRIP | POPT | -0.150 | 0.364 | small |
| GRASP | LRU | +0.700 | 1.551 | large |
| GRASP | SRRIP | +0.700 | 1.551 | large |
| GRASP | POPT | +0.550 | 1.187 | large |
| POPT | LRU | +0.150 | 0.364 | small |
| POPT | SRRIP | +0.150 | 0.364 | small |
| POPT | GRASP | -0.550 | 1.187 | large |

### bfs

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 4 / 23 | 0.174 |
| SRRIP | 3 / 23 | 0.130 |
| GRASP | 12 / 23 | 0.522 |
| POPT | 13 / 23 | 0.565 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | +0.043 | 0.121 | negligible |
| LRU | GRASP | -0.348 | 0.754 | medium |
| LRU | POPT | -0.391 | 0.841 | large |
| SRRIP | LRU | -0.043 | 0.121 | negligible |
| SRRIP | GRASP | -0.391 | 0.875 | large |
| SRRIP | POPT | -0.435 | 0.963 | large |
| GRASP | LRU | +0.348 | 0.754 | medium |
| GRASP | SRRIP | +0.391 | 0.875 | large |
| GRASP | POPT | -0.043 | 0.087 | negligible |
| POPT | LRU | +0.391 | 0.841 | large |
| POPT | SRRIP | +0.435 | 0.963 | large |
| POPT | GRASP | +0.043 | 0.087 | negligible |

### sssp

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 4 / 20 | 0.200 |
| SRRIP | 5 / 20 | 0.250 |
| GRASP | 10 / 20 | 0.500 |
| POPT | 7 / 20 | 0.350 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | -0.050 | 0.120 | negligible |
| LRU | GRASP | -0.300 | 0.643 | medium |
| LRU | POPT | -0.150 | 0.339 | small |
| SRRIP | LRU | +0.050 | 0.120 | negligible |
| SRRIP | GRASP | -0.250 | 0.524 | medium |
| SRRIP | POPT | -0.100 | 0.219 | small |
| GRASP | LRU | +0.300 | 0.643 | medium |
| GRASP | SRRIP | +0.250 | 0.524 | medium |
| GRASP | POPT | +0.150 | 0.305 | small |
| POPT | LRU | +0.150 | 0.339 | small |
| POPT | SRRIP | +0.100 | 0.219 | small |
| POPT | GRASP | -0.150 | 0.305 | small |
