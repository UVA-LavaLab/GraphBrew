# Per-L3-size policy stability

Source: `wiki/data/oracle_gap.json` (456 rows).

Paper-grade L3 sizes: 1MB, 4MB, 8MB.

## Stability summary across paper L3 sizes (1MB, 4MB, 8MB)

| App | Tops (1MB / 4MB / 8MB) | Unique tops | Stable? | Regime change? |
| --- | --- | --- | --- | --- |
| pr | POPT / POPT / POPT | 1 | YES | no |
| bc | GRASP / GRASP / GRASP | 1 | YES | no |
| cc | POPT / GRASP / POPT | 2 | no | YES |
| bfs | GRASP / GRASP / POPT | 2 | no | YES |
| sssp | GRASP / GRASP / SRRIP | 2 | no | YES |

## Per-app per-L3 winner tables

### pr

| L3 | n_cells | Top | wins | share | Runner-up | margin | Unique? |
| --- | ---: | --- | ---: | ---: | --- | ---: | --- |
| 4kB | 2 | **GRASP** | 2 | 1.00 | LRU | 2 | yes |
| 16kB | 2 | **GRASP** | 1 | 0.50 | POPT | 0 | tie |
| 64kB | 2 | **POPT** | 2 | 1.00 | LRU | 2 | yes |
| 256kB | 2 | **POPT** | 2 | 1.00 | LRU | 2 | yes |
| 1MB | 8 | **POPT** | 6 | 0.75 | GRASP | 3 | yes |
| 4MB | 6 | **POPT** | 5 | 0.83 | SRRIP | 3 | yes |
| 8MB | 6 | **POPT** | 5 | 0.83 | GRASP | 3 | yes |

### bc

| L3 | n_cells | Top | wins | share | Runner-up | margin | Unique? |
| --- | ---: | --- | ---: | ---: | --- | ---: | --- |
| 4kB | 1 | **POPT** | 1 | 1.00 | LRU | 1 | yes |
| 16kB | 1 | **POPT** | 1 | 1.00 | LRU | 1 | yes |
| 64kB | 1 | **GRASP** | 1 | 1.00 | LRU | 1 | yes |
| 256kB | 1 | **GRASP** | 1 | 1.00 | LRU | 1 | yes |
| 1MB | 7 | **GRASP** | 5 | 0.71 | LRU | 2 | yes |
| 4MB | 6 | **GRASP** | 5 | 0.83 | SRRIP | 3 | yes |
| 8MB | 6 | **GRASP** | 5 | 0.83 | SRRIP | 3 | yes |

### cc

| L3 | n_cells | Top | wins | share | Runner-up | margin | Unique? |
| --- | ---: | --- | ---: | ---: | --- | ---: | --- |
| 4kB | 1 | **POPT** | 1 | 1.00 | LRU | 1 | yes |
| 16kB | 1 | **GRASP** | 1 | 1.00 | LRU | 1 | yes |
| 64kB | 1 | **GRASP** | 1 | 1.00 | LRU | 1 | yes |
| 256kB | 1 | **POPT** | 1 | 1.00 | LRU | 1 | yes |
| 1MB | 6 | **POPT** | 4 | 0.67 | GRASP | 2 | yes |
| 4MB | 5 | **GRASP** | 3 | 0.60 | POPT | 1 | yes |
| 8MB | 5 | **POPT** | 3 | 0.60 | LRU | 1 | yes |

### bfs

| L3 | n_cells | Top | wins | share | Runner-up | margin | Unique? |
| --- | ---: | --- | ---: | ---: | --- | ---: | --- |
| 4kB | 1 | **GRASP** | 1 | 1.00 | LRU | 1 | yes |
| 16kB | 1 | **GRASP** | 1 | 1.00 | LRU | 1 | yes |
| 64kB | 1 | **GRASP** | 1 | 1.00 | LRU | 1 | yes |
| 256kB | 1 | **POPT** | 1 | 1.00 | LRU | 1 | yes |
| 1MB | 7 | **GRASP** | 6 | 0.86 | LRU | 4 | yes |
| 4MB | 6 | **GRASP** | 4 | 0.67 | POPT | 1 | yes |
| 8MB | 6 | **POPT** | 5 | 0.83 | GRASP | 3 | yes |

### sssp

| L3 | n_cells | Top | wins | share | Runner-up | margin | Unique? |
| --- | ---: | --- | ---: | ---: | --- | ---: | --- |
| 4kB | 1 | **GRASP** | 1 | 1.00 | LRU | 1 | yes |
| 16kB | 1 | **GRASP** | 1 | 1.00 | LRU | 1 | yes |
| 64kB | 1 | **GRASP** | 1 | 1.00 | LRU | 1 | yes |
| 256kB | 1 | **SRRIP** | 1 | 1.00 | LRU | 1 | yes |
| 1MB | 6 | **GRASP** | 5 | 0.83 | LRU | 4 | yes |
| 4MB | 5 | **GRASP** | 4 | 0.80 | LRU | 3 | yes |
| 8MB | 5 | **SRRIP** | 3 | 0.60 | LRU | 1 | yes |
