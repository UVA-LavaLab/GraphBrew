# Literature-faithfulness margin audit

Per-claim distance to the nearest disagree boundary (threshold for *fragile*: < 1.0 pp).

## Summary

- Total claims: **279**
- Claims with a bounded margin: **279**
- Unbounded magnitude-only (`~` with no `max_abs`): **0**
- **Fragile** cells (< 1.0 pp from disagree boundary): **33**
- `ok`-status cells with negative margin (classifier/audit disagreement): **0**

### Margin distribution (pp)

| stat | value |
|---|---|
| min | -10.2166 |
| p10 | 0.8088 |
| p25 | 2.062 |
| median | 5.4997 |
| mean | 5.328 |
| p75 | 7.7395 |
| p90 | 10.1229 |
| max | 19.3177 |

## Per-status margin

| status | count | min | median | mean | max |
|---|---|---|---|---|---|
| insufficient_data | 28 | 1.0 | 7.0 | 5.2857 | 8.0 |
| known_deviation | 17 | -10.2166 | -2.1753 | -3.0393 | -0.3494 |
| ok | 234 | 0.001 | 5.6407 | 5.9409 | 19.3177 |

## Per-family margin

| family | count | min | median | mean | max |
|---|---|---|---|---|---|
| citation | 52 | -0.8097 | 5.5969 | 5.509 | 17.5804 |
| social | 177 | -10.2166 | 5.1515 | 4.6846 | 13.3172 |
| web | 50 | -1.0758 | 7.3703 | 7.4174 | 19.3177 |

## Binding-boundary breakdown

Which side of the claim envelope is currently nearest (the boundary that defines each cell's margin):

| boundary | count |
|---|---|
| magnitude_max_abs | 95 |
| sign_upper_tol | 84 |
| trigger_headroom | 61 |
| near_grasp_upper | 23 |
| sign_upper_min_abs | 16 |

## Fragile cells (margin < 1.0 pp)

| graph | app | L3 | policy | sign | Δ pp | margin pp | status | binding |
|---|---|---|---|---|---|---|---|---|
| com-orkut | cc | 4MB | POPT_GE_GRASP | - | 11.7166 | -10.217 | known_deviation | sign_upper_tol |
| com-orkut | cc | 8MB | POPT_GE_GRASP | - | 8.9804 | -7.48 | known_deviation | sign_upper_tol |
| com-orkut | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 11.7166 | -4.717 | known_deviation | near_grasp_upper |
| soc-pokec | cc | 4MB | POPT_GE_GRASP | - | 6.1477 | -4.648 | known_deviation | sign_upper_tol |
| com-orkut | cc | 1MB | POPT_GE_GRASP | - | 5.943 | -4.443 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | cc | 8MB | POPT_GE_GRASP | - | 5.8083 | -4.308 | known_deviation | sign_upper_tol |
| com-orkut | bc | 8MB | POPT_GE_GRASP | - | 5.7977 | -4.298 | known_deviation | sign_upper_tol |
| soc-pokec | cc | 1MB | POPT_GE_GRASP | - | 3.7238 | -2.224 | known_deviation | sign_upper_tol |
| com-orkut | bc | 4MB | POPT_GE_GRASP | - | 3.6753 | -2.175 | known_deviation | sign_upper_tol |
| com-orkut | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 8.9804 | -1.98 | known_deviation | near_grasp_upper |
| web-Google | bc | 8MB | POPT_GE_GRASP | - | 2.5758 | -1.076 | known_deviation | sign_upper_tol |
| soc-pokec | sssp | 1MB | POPT_GE_GRASP | - | 2.4916 | -0.992 | known_deviation | sign_upper_tol |
| web-Google | bc | 4MB | POPT_GE_GRASP | - | 2.4858 | -0.986 | known_deviation | sign_upper_tol |
| cit-Patents | bc | 8MB | POPT_GE_GRASP | - | 2.3097 | -0.81 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | cc | 4MB | POPT_GE_GRASP | - | 1.9937 | -0.494 | known_deviation | sign_upper_tol |
| com-orkut | sssp | 1MB | POPT_GE_GRASP | - | 1.9731 | -0.473 | known_deviation | sign_upper_tol |
| cit-Patents | cc | 8MB | POPT_GE_GRASP | - | 1.8494 | -0.349 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | bfs | 4MB | POPT_GE_GRASP | - | 1.499 | 0.001 | ok | sign_upper_tol |
| soc-pokec | bfs | 1MB | POPT_GE_GRASP | - | 1.3587 | 0.141 | ok | sign_upper_tol |
| cit-Patents | bc | 4MB | POPT_GE_GRASP | - | 1.2254 | 0.275 | ok | sign_upper_tol |
| web-Google | bfs | 1MB | POPT_GE_GRASP | - | 1.1221 | 0.378 | ok | sign_upper_tol |
| cit-Patents | sssp | 4MB | POPT_GE_GRASP | - | 0.9482 | 0.552 | ok | sign_upper_tol |
| web-Google | pr | 1MB | POPT | - | -21.4332 | 0.567 | ok | magnitude_max_abs |
| com-orkut | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 1.9731 | 0.614 | ok | trigger_headroom |
| com-orkut | bc | 1MB | POPT_GE_GRASP | - | 0.8399 | 0.66 | ok | sign_upper_tol |
| com-orkut | bfs | 4MB | POPT_GE_GRASP | - | 0.7929 | 0.707 | ok | sign_upper_tol |
| cit-Patents | bfs | 4MB | POPT_GE_GRASP | - | 0.7791 | 0.721 | ok | sign_upper_tol |
| soc-LiveJournal1 | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 6.8471 | 0.725 | ok | trigger_headroom |
| soc-pokec | pr | 8MB | POPT_GE_GRASP | - | 0.1702 | 0.83 | ok | sign_upper_tol |
| soc-pokec | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 6.1477 | 0.852 | ok | near_grasp_upper |
| soc-LiveJournal1 | bc | 8MB | POPT_GE_GRASP | - | 0.614 | 0.886 | ok | sign_upper_tol |
| cit-Patents | sssp | 1MB | POPT_GE_GRASP | - | 0.6036 | 0.896 | ok | sign_upper_tol |
| cit-Patents | cc | 4MB | POPT_GE_GRASP | - | 0.5228 | 0.977 | ok | sign_upper_tol |
