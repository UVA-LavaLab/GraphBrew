# Per-cell gap-distribution shape envelope (gate 56)

Per-(app, L3, policy) skew and excess-kurtosis envelope for the paper L3 grid ['1MB', '4MB', '8MB']. Extends gate 46 to the cell level so a single bad cell cannot hide behind pooled marginals.

- source: `wiki/data/oracle_gap.json`
- rows in scope: 360
- cells: 60 (5 apps × 3 L3 × 4 policies)
- envelope: |skew| < 2.0, |excess kurt| < 7.0 (Hesterberg 2015 (Am. Statistician); Efron & Tibshirani 1993 (An Introduction to the Bootstrap).)
- worst |skew| any cell: **2.6457** at bfs/1MB/GRASP
- worst |excess kurt| any cell: **6.9999** at bfs/1MB/GRASP
- cells outside envelope: **13** / pinned set: **13** / max allowed: **13**
- verdict: **PASS**

## Per-L3 worst

| L3 | cells | worst \|skew\| | worst \|kurt\| |
| --- | ---: | ---: | ---: |
| 1MB | 20 | 2.6457 | 6.9999 |
| 4MB | 20 | 2.4495 | 6.0 |
| 8MB | 20 | 2.4495 | 6.0 |

## Worst 10 cells by |skew|

| app | L3 | policy | n | mean gap pp | skew | excess kurt |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| bfs | 1MB | GRASP | 7 | 8.9621 | 2.6457 | 6.9999 |
| bfs | 1MB | POPT | 7 | 5.2856 | 2.6349 | 6.9557 |
| bc | 1MB | POPT | 7 | 2.7323 | 2.5727 | 6.6883 |
| bc | 1MB | GRASP | 7 | 2.7304 | 2.5621 | 6.6361 |
| bc | 4MB | GRASP | 6 | 0.3287 | 2.4495 | 6.0 |
| pr | 8MB | POPT | 6 | 0.0283 | 2.4495 | 6.0 |
| sssp | 1MB | GRASP | 6 | 9.4935 | 2.4473 | 5.9916 |
| bc | 8MB | GRASP | 6 | 0.9523 | 2.3998 | 5.7962 |
| sssp | 8MB | POPT | 5 | 0.0404 | 2.2361 | 5.0 |
| sssp | 4MB | GRASP | 5 | 1.2254 | 2.2208 | 4.9429 |
