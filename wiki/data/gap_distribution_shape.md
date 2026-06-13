# Per-cell gap-distribution shape envelope (gate 56)

Per-(app, L3, policy) skew and excess-kurtosis envelope for the paper L3 grid ['1MB', '4MB', '8MB']. Extends gate 46 to the cell level so a single bad cell cannot hide behind pooled marginals.

- source: `wiki/data/oracle_gap.json`
- rows in scope: 360
- cells: 60 (5 apps × 3 L3 × 4 policies)
- envelope: |skew| < 2.0, |excess kurt| < 7.0 (Hesterberg 2015 (Am. Statistician); Efron & Tibshirani 1993 (An Introduction to the Bootstrap).)
- worst |skew| any cell: **2.6458** at bfs/1MB/GRASP
- worst |excess kurt| any cell: **7.0** at bfs/1MB/GRASP
- cells outside envelope: **15** / pinned set: **13** / max allowed: **13**
- NEW offenders vs pin: ['bfs/8MB/POPT', 'cc/1MB/POPT', 'cc/4MB/GRASP', 'pr/4MB/POPT']
- cells that exited the offending set: ['sssp/4MB/POPT', 'sssp/8MB/POPT']
- verdict: **FAIL**

## Per-L3 worst

| L3 | cells | worst \|skew\| | worst \|kurt\| |
| --- | ---: | ---: | ---: |
| 1MB | 20 | 2.6458 | 7.0 |
| 4MB | 20 | 2.4495 | 6.0 |
| 8MB | 20 | 2.4495 | 6.0 |

## Worst 10 cells by |skew|

| app | L3 | policy | n | mean gap pp | skew | excess kurt |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| bfs | 1MB | GRASP | 7 | 8.9409 | 2.6458 | 7.0 |
| bfs | 1MB | POPT | 7 | 6.01 | 2.6042 | 6.8316 |
| bc | 1MB | GRASP | 7 | 2.6201 | 2.5583 | 6.613 |
| bc | 4MB | GRASP | 6 | 0.3287 | 2.4495 | 6.0 |
| bc | 8MB | GRASP | 6 | 0.8677 | 2.4495 | 6.0 |
| bfs | 8MB | POPT | 6 | 0.1677 | 2.4495 | 6.0 |
| pr | 4MB | POPT | 6 | 0.1102 | 2.4495 | 6.0 |
| pr | 8MB | POPT | 6 | 0.0952 | 2.4495 | 6.0 |
| sssp | 1MB | GRASP | 6 | 9.3045 | 2.4495 | 6.0 |
| bc | 1MB | POPT | 7 | 3.9859 | 2.439 | 6.1709 |
