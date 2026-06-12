# Literature-faithfulness rationale-grid audit (LIT-RatGrid)

Per (policy, graph, app) cell: rationale text must be unique within the cell. Theorem-class policies (POPT_GE_GRASP, POPT_NEAR_GRASP_IF_BIG_GAP, SRRIP) additionally require constant rationale across graphs within (policy, app). Every rationale must cite the source paper / figure.

## Summary

| Metric | Value |
|---|---|
| Total per_claim rows | 279 |
| (policy, graph, app) cells | 106 |
| Cell-uniqueness violations | 0 |
| Theorem-invariance violations | 0 |
| Min rationale length | 40 chars |
| Total violations | 0 |

## Unique rationales per policy

| policy | distinct rationales |
|---|---|
| GRASP | 19 |
| POPT | 8 |
| POPT_GE_GRASP | 5 |
| POPT_NEAR_GRASP_IF_BIG_GAP | 1 |
| SRRIP | 5 |

## Rationale count per (policy, app)

| policy | app | count |
|---|---|---|
| GRASP | bc | 4 |
| GRASP | bfs | 3 |
| GRASP | pr | 11 |
| GRASP | sssp | 1 |
| POPT | pr | 5 |
| POPT | sssp | 3 |
| POPT_GE_GRASP | bc | 1 |
| POPT_GE_GRASP | bfs | 1 |
| POPT_GE_GRASP | cc | 1 |
| POPT_GE_GRASP | pr | 1 |
| POPT_GE_GRASP | sssp | 1 |
| POPT_NEAR_GRASP_IF_BIG_GAP | bc | 1 |
| POPT_NEAR_GRASP_IF_BIG_GAP | bfs | 1 |
| POPT_NEAR_GRASP_IF_BIG_GAP | cc | 1 |
| POPT_NEAR_GRASP_IF_BIG_GAP | pr | 1 |
| POPT_NEAR_GRASP_IF_BIG_GAP | sssp | 1 |
| SRRIP | bc | 1 |
| SRRIP | bfs | 1 |
| SRRIP | cc | 1 |
| SRRIP | pr | 1 |
| SRRIP | sssp | 1 |

_No violations._
