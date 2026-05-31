# Per-cell gap-distribution shape envelope (gate 56)

Per-(app, L3, policy) skew and excess-kurtosis envelope for the paper L3 grid ['1MB', '4MB', '8MB']. Extends gate 46 to the cell level so a single bad cell cannot hide behind pooled marginals.

- source: `wiki/data/oracle_gap.json`
- rows in scope: 360
- cells: 60 (5 apps × 3 L3 × 4 policies)
- envelope: |skew| < 2.0, |excess kurt| < 7.0 (Hesterberg 2015 (Am. Statistician); Efron & Tibshirani 1993 (An Introduction to the Bootstrap).)
- worst |skew| any cell: **2.8284** at pr/1MB/POPT
- worst |excess kurt| any cell: **8.0** at pr/1MB/POPT
- cells outside envelope: **12** / pinned set: **12** / max allowed: **12**
- verdict: **PASS**

## Per-L3 worst

| L3 | cells | worst \|skew\| | worst \|kurt\| |
| --- | ---: | ---: | ---: |
| 1MB | 20 | 2.8284 | 8.0 |
| 4MB | 20 | 2.4495 | 6.0 |
| 8MB | 20 | 2.4495 | 6.0 |

## Worst 10 cells by |skew|

| app | L3 | policy | n | mean gap pp | skew | excess kurt |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| pr | 1MB | POPT | 8 | 0.0434 | 2.8284 | 8.0 |
| bfs | 1MB | POPT | 7 | 5.3117 | 2.6363 | 6.9618 |
| bc | 1MB | GRASP | 7 | 6.0079 | 2.6224 | 6.904 |
| bc | 1MB | POPT | 7 | 4.4764 | 2.5474 | 6.606 |
| cc | 1MB | GRASP | 6 | 1.52 | 2.4495 | 6.0 |
| pr | 4MB | POPT | 6 | 0.1772 | 2.4495 | 6.0 |
| pr | 8MB | POPT | 6 | 0.1365 | 2.4495 | 6.0 |
| sssp | 1MB | GRASP | 6 | 12.4487 | 2.4357 | 5.9456 |
| bfs | 1MB | GRASP | 7 | 12.3081 | 2.3261 | 5.4753 |
| cc | 8MB | GRASP | 5 | 0.001 | 2.2361 | 5.0 |
