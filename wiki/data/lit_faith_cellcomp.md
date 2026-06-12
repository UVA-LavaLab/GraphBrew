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
| cit-Patents | bc | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.945117 |
| cit-Patents | bc | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.818927 |
| cit-Patents | bc | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.686411 |
| cit-Patents | bfs | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.98146 |
| cit-Patents | bfs | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.946715 |
| cit-Patents | bfs | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.921363 |
| cit-Patents | cc | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.66682 |
| cit-Patents | cc | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.33691400000000005 |
| cit-Patents | cc | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.197967 |
| cit-Patents | pr | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.895106 |
| cit-Patents | pr | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.590052 |
| cit-Patents | pr | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.32620000000000005 |
| cit-Patents | sssp | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.87683 |
| cit-Patents | sssp | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.568128 |
| cit-Patents | sssp | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.270869 |
| com-orkut | bc | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.836912 |
| com-orkut | bc | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.5735250000000001 |
| com-orkut | bc | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.37604499999999996 |
| com-orkut | bfs | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.998892 |
| com-orkut | bfs | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.996484 |
| com-orkut | bfs | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.995799 |
| com-orkut | cc | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.741545 |
| com-orkut | cc | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.5676140000000001 |
| com-orkut | cc | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.506299 |
| com-orkut | pr | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.742564 |
| com-orkut | pr | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.34560900000000006 |
| com-orkut | pr | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.185573 |
| com-orkut | sssp | 1MB | 4 | GRASP,LRU,POPT,SRRIP | 0.686743 |
| com-orkut | sssp | 4MB | 4 | GRASP,LRU,POPT,SRRIP | 0.17303599999999997 |
| com-orkut | sssp | 8MB | 4 | GRASP,LRU,POPT,SRRIP | 0.016185000000000005 |

_No violations._
