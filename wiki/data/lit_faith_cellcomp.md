# Literature-faithfulness cell-completeness audit (LIT-CellComp)

Per (graph, app, l3) cell in the per_observation table: canonical roster {LRU, GRASP, POPT} present + LRU baseline + delta arithmetic + L3 sweep coverage + axis parity + miss-rate bounds + uniqueness.

## Summary

| Metric | Value |
|---|---|
| Total per_observation rows | 456 |
| Distinct (graph, app, l3) cells | 114 |
| Distinct graphs | 8 |
| Distinct apps | 5 |
| Distinct policies | 4 |
| Duplicate rows | 0 |
| Min L3 sweep | 3 |
| Delta arithmetic tolerance | 0.001 pp |
| Violations | 0 |

## Per-cell policy roster (first 30)

| graph | app | l3 | n | policies | LRU miss |
|---|---|---|---|---|---|
| cit-Patents | bc | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.884326 |
| cit-Patents | bc | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.731853 |
| cit-Patents | bc | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.6071230000000001 |
| cit-Patents | bfs | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.970808 |
| cit-Patents | bfs | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.926919 |
| cit-Patents | bfs | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.902066 |
| cit-Patents | cc | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.643092 |
| cit-Patents | cc | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.31697 |
| cit-Patents | cc | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.18154199999999998 |
| cit-Patents | pr | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.894438 |
| cit-Patents | pr | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.58934 |
| cit-Patents | pr | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.325581 |
| cit-Patents | sssp | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.848641 |
| cit-Patents | sssp | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.553585 |
| cit-Patents | sssp | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.260312 |
| com-orkut | bc | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.862362 |
| com-orkut | bc | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.582657 |
| com-orkut | bc | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.382097 |
| com-orkut | bfs | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.9972799999999999 |
| com-orkut | bfs | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.994937 |
| com-orkut | bfs | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.994229 |
| com-orkut | cc | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.6889179999999999 |
| com-orkut | cc | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.503285 |
| com-orkut | cc | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.430593 |
| com-orkut | pr | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.704728 |
| com-orkut | pr | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.308477 |
| com-orkut | pr | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.16360799999999998 |
| com-orkut | sssp | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.660328 |
| com-orkut | sssp | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.16971899999999995 |
| com-orkut | sssp | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.02459599999999995 |

_No violations._
