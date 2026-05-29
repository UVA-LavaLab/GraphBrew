# Per-cell gap-distribution shape envelope (gate 56)

Per-(app, L3, policy) skew and excess-kurtosis envelope for the paper L3 grid ['1MB', '4MB', '8MB']. Extends gate 46 to the cell level so a single bad cell cannot hide behind pooled marginals.

- source: `wiki/data/oracle_gap.json`
- rows in scope: 360
- cells: 60 (5 apps × 3 L3 × 4 policies)
- envelope: |skew| < 2.0, |excess kurt| < 7.0 (Hesterberg 2015 (Am. Statistician); Efron & Tibshirani 1993 (An Introduction to the Bootstrap).)
- worst |skew| any cell: **2.8284** at pr/1MB/POPT
- worst |excess kurt| any cell: **8.0** at pr/1MB/POPT
- cells outside envelope: **14** / pinned set: **14** / max allowed: **14**
- verdict: **PASS**

## Per-L3 worst

| L3 | cells | worst \|skew\| | worst \|kurt\| |
| --- | ---: | ---: | ---: |
| 1MB | 20 | 2.8284 | 8.0 |
| 4MB | 20 | 2.4495 | 6.0 |
| 8MB | 20 | 2.4494 | 5.9998 |

## Worst 10 cells by |skew|

| app | L3 | policy | n | mean gap pp | skew | excess kurt |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| pr | 1MB | POPT | 8 | 0.0526 | 2.8284 | 8.0 |
| bfs | 1MB | POPT | 7 | 5.164 | 2.6361 | 6.9607 |
| bc | 1MB | GRASP | 7 | 5.8506 | 2.632 | 6.9433 |
| bc | 1MB | POPT | 7 | 4.559 | 2.5454 | 6.6003 |
| bfs | 4MB | POPT | 6 | 0.0633 | 2.4495 | 6.0 |
| cc | 1MB | GRASP | 6 | 1.52 | 2.4495 | 6.0 |
| bfs | 8MB | POPT | 6 | 0.1152 | 2.4494 | 5.9998 |
| pr | 4MB | POPT | 6 | 0.1828 | 2.4466 | 5.9888 |
| pr | 8MB | POPT | 6 | 0.1393 | 2.4456 | 5.985 |
| sssp | 1MB | GRASP | 6 | 13.0337 | 2.4302 | 5.926 |

