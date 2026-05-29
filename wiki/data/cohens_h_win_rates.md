# Cohen's h on policy win-rate gaps

Source: `wiki/data/oracle_gap.json` (456 rows).

Thresholds (Cohen 1988): small ≥ 0.2, medium ≥ 0.5, large ≥ 0.8.

## Largest effect per kernel

| App | Favors | Comparison | p_a | p_b | h | magnitude |
| --- | --- | --- | ---: | ---: | ---: | --- |
| pr | POPT | LRU vs POPT | 0.000 | 0.714 | 2.014 | **large** |
| bc | GRASP | LRU vs GRASP | 0.043 | 0.652 | 1.460 | **large** |
| cc | GRASP | GRASP vs POPT | 0.850 | 0.000 | 2.346 | **large** |
| bfs | POPT | LRU vs POPT | 0.043 | 0.522 | 1.194 | **large** |
| sssp | POPT | SRRIP vs POPT | 0.100 | 0.400 | 0.726 | **medium** |

## All large-effect dominance pairs (h ≥ 0.8, p_a > p_b)

| App | Winner | Loser | p_winner | p_loser | h |
| --- | --- | --- | ---: | ---: | ---: |
| cc | GRASP | POPT | 0.850 | 0.000 | 2.346 |
| pr | POPT | LRU | 0.714 | 0.000 | 2.014 |
| pr | POPT | SRRIP | 0.714 | 0.000 | 2.014 |
| cc | GRASP | SRRIP | 0.850 | 0.050 | 1.895 |
| cc | GRASP | LRU | 0.850 | 0.100 | 1.703 |
| bc | GRASP | LRU | 0.652 | 0.043 | 1.460 |
| bfs | POPT | LRU | 0.522 | 0.043 | 1.194 |
| bfs | POPT | SRRIP | 0.522 | 0.043 | 1.194 |
| pr | GRASP | LRU | 0.286 | 0.000 | 1.128 |
| pr | GRASP | SRRIP | 0.286 | 0.000 | 1.128 |
| bfs | GRASP | LRU | 0.391 | 0.043 | 0.931 |
| bfs | GRASP | SRRIP | 0.391 | 0.043 | 0.931 |
| bc | GRASP | POPT | 0.652 | 0.217 | 0.910 |
| pr | POPT | GRASP | 0.714 | 0.286 | 0.886 |

## Per-app full table

### pr

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 0 / 28 | 0.000 |
| SRRIP | 0 / 28 | 0.000 |
| GRASP | 8 / 28 | 0.286 |
| POPT | 20 / 28 | 0.714 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | +0.000 | 0.000 | negligible |
| LRU | GRASP | -0.286 | 1.128 | large |
| LRU | POPT | -0.714 | 2.014 | large |
| SRRIP | LRU | +0.000 | 0.000 | negligible |
| SRRIP | GRASP | -0.286 | 1.128 | large |
| SRRIP | POPT | -0.714 | 2.014 | large |
| GRASP | LRU | +0.286 | 1.128 | large |
| GRASP | SRRIP | +0.286 | 1.128 | large |
| GRASP | POPT | -0.429 | 0.886 | large |
| POPT | LRU | +0.714 | 2.014 | large |
| POPT | SRRIP | +0.714 | 2.014 | large |
| POPT | GRASP | +0.429 | 0.886 | large |

### bc

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 1 / 23 | 0.043 |
| SRRIP | 7 / 23 | 0.304 |
| GRASP | 15 / 23 | 0.652 |
| POPT | 5 / 23 | 0.217 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | -0.261 | 0.749 | medium |
| LRU | GRASP | -0.609 | 1.460 | large |
| LRU | POPT | -0.174 | 0.550 | medium |
| SRRIP | LRU | +0.261 | 0.749 | medium |
| SRRIP | GRASP | -0.348 | 0.711 | medium |
| SRRIP | POPT | +0.087 | 0.199 | negligible |
| GRASP | LRU | +0.609 | 1.460 | large |
| GRASP | SRRIP | +0.348 | 0.711 | medium |
| GRASP | POPT | +0.435 | 0.910 | large |
| POPT | LRU | +0.174 | 0.550 | medium |
| POPT | SRRIP | -0.087 | 0.199 | negligible |
| POPT | GRASP | -0.435 | 0.910 | large |

### cc

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 2 / 20 | 0.100 |
| SRRIP | 1 / 20 | 0.050 |
| GRASP | 17 / 20 | 0.850 |
| POPT | 0 / 20 | 0.000 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | +0.050 | 0.193 | negligible |
| LRU | GRASP | -0.750 | 1.703 | large |
| LRU | POPT | +0.100 | 0.643 | medium |
| SRRIP | LRU | -0.050 | 0.193 | negligible |
| SRRIP | GRASP | -0.800 | 1.895 | large |
| SRRIP | POPT | +0.050 | 0.451 | small |
| GRASP | LRU | +0.750 | 1.703 | large |
| GRASP | SRRIP | +0.800 | 1.895 | large |
| GRASP | POPT | +0.850 | 2.346 | large |
| POPT | LRU | -0.100 | 0.643 | medium |
| POPT | SRRIP | -0.050 | 0.451 | small |
| POPT | GRASP | -0.850 | 2.346 | large |

### bfs

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 1 / 23 | 0.043 |
| SRRIP | 1 / 23 | 0.043 |
| GRASP | 9 / 23 | 0.391 |
| POPT | 12 / 23 | 0.522 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | +0.000 | 0.000 | negligible |
| LRU | GRASP | -0.348 | 0.931 | large |
| LRU | POPT | -0.478 | 1.194 | large |
| SRRIP | LRU | +0.000 | 0.000 | negligible |
| SRRIP | GRASP | -0.348 | 0.931 | large |
| SRRIP | POPT | -0.478 | 1.194 | large |
| GRASP | LRU | +0.348 | 0.931 | large |
| GRASP | SRRIP | +0.348 | 0.931 | large |
| GRASP | POPT | -0.130 | 0.263 | small |
| POPT | LRU | +0.478 | 1.194 | large |
| POPT | SRRIP | +0.478 | 1.194 | large |
| POPT | GRASP | +0.130 | 0.263 | small |

### sssp

Win rates:

| Policy | Wins / N | p̂ |
| --- | ---: | ---: |
| LRU | 3 / 20 | 0.150 |
| SRRIP | 2 / 20 | 0.100 |
| GRASP | 7 / 20 | 0.350 |
| POPT | 8 / 20 | 0.400 |

Comparisons (h on ordered pairs):

| a | b | Δp | h | magnitude |
| --- | --- | ---: | ---: | --- |
| LRU | SRRIP | +0.050 | 0.152 | negligible |
| LRU | GRASP | -0.200 | 0.471 | small |
| LRU | POPT | -0.250 | 0.574 | medium |
| SRRIP | LRU | -0.050 | 0.152 | negligible |
| SRRIP | GRASP | -0.250 | 0.623 | medium |
| SRRIP | POPT | -0.300 | 0.726 | medium |
| GRASP | LRU | +0.200 | 0.471 | small |
| GRASP | SRRIP | +0.250 | 0.623 | medium |
| GRASP | POPT | -0.050 | 0.103 | negligible |
| POPT | LRU | +0.250 | 0.574 | medium |
| POPT | SRRIP | +0.300 | 0.726 | medium |
| POPT | GRASP | +0.050 | 0.103 | negligible |

