# Literature-faithfulness margin audit

Per-claim distance to the nearest disagree boundary (threshold for *fragile*: < 1.0 pp).

## Summary

- Total claims: **330**
- Claims with a bounded margin: **330**
- Unbounded magnitude-only (`~` with no `max_abs`): **0**
- **Fragile** cells (< 1.0 pp from disagree boundary): **50**
- `ok`-status cells with negative margin (classifier/audit disagreement): **0**

### Margin distribution (pp)

| stat | value |
|---|---|
| min | -9.5164 |
| p10 | 0.2574 |
| p25 | 1.5227 |
| median | 5.4011 |
| mean | 6.3255 |
| p75 | 8.4183 |
| p90 | 10.7161 |
| max | 80.1168 |

## Per-status margin

| status | count | min | median | mean | max |
|---|---|---|---|---|---|
| insufficient_data | 7 | 1.0 | 4.0 | 4.0 | 7.0 |
| known_deviation | 24 | -9.5164 | -2.2287 | -3.0587 | -0.0277 |
| ok | 298 | 0.0181 | 5.6252 | 7.1552 | 80.1168 |
| within_tolerance | 1 | 0.5799 | 0.5799 | 0.5799 | 0.5799 |

## Per-family margin

| family | count | min | median | mean | max |
|---|---|---|---|---|---|
| citation | 52 | -7.2258 | 4.5759 | 4.7292 | 15.6517 |
| mesh | 10 | 0.6681 | 5.1988 | 5.3739 | 10.8674 |
| road | 50 | 0.9686 | 9.7976 | 15.0213 | 80.1168 |
| social | 168 | -9.5164 | 4.7206 | 4.481 | 27.7515 |
| web | 50 | -1.1456 | 6.6484 | 5.6779 | 12.9997 |

## Binding-boundary breakdown

Which side of the claim envelope is currently nearest (the boundary that defines each cell's margin):

| boundary | count |
|---|---|
| sign_upper_tol | 114 |
| trigger_headroom | 95 |
| magnitude_max_abs | 86 |
| near_grasp_upper | 19 |
| sign_upper_min_abs | 16 |

## Fragile cells (margin < 1.0 pp)

| graph | app | L3 | policy | sign | Δ pp | margin pp | status | binding |
|---|---|---|---|---|---|---|---|---|
| com-orkut | cc | 1MB | POPT_GE_GRASP | - | 11.0164 | -9.516 | known_deviation | sign_upper_tol |
| soc-pokec | cc | 1MB | POPT_GE_GRASP | - | 10.5803 | -9.08 | known_deviation | sign_upper_tol |
| com-orkut | cc | 4MB | POPT_GE_GRASP | - | 10.0547 | -8.555 | known_deviation | sign_upper_tol |
| cit-Patents | cc | 1MB | POPT_GE_GRASP | - | 8.7258 | -7.226 | known_deviation | sign_upper_tol |
| com-orkut | cc | 8MB | POPT_GE_GRASP | - | 6.1871 | -4.687 | known_deviation | sign_upper_tol |
| soc-pokec | cc | 4MB | POPT_GE_GRASP | - | 5.6082 | -4.108 | known_deviation | sign_upper_tol |
| com-orkut | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 11.0164 | -4.016 | known_deviation | near_grasp_upper |
| soc-pokec | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 10.5803 | -3.58 | known_deviation | near_grasp_upper |
| com-orkut | bc | 8MB | POPT_GE_GRASP | - | 4.727 | -3.227 | known_deviation | sign_upper_tol |
| com-orkut | bc | 4MB | POPT_GE_GRASP | - | 4.6678 | -3.168 | known_deviation | sign_upper_tol |
| com-orkut | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 10.0547 | -3.055 | known_deviation | near_grasp_upper |
| soc-pokec | sssp | 1MB | POPT_GE_GRASP | - | 3.8096 | -2.31 | known_deviation | sign_upper_tol |
| cit-Patents | cc | 4MB | POPT_GE_GRASP | - | 3.6479 | -2.148 | known_deviation | sign_upper_tol |
| cit-Patents | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 8.7258 | -1.726 | known_deviation | near_grasp_upper |
| soc-LiveJournal1 | cc | 8MB | POPT_GE_GRASP | - | 3.0832 | -1.583 | known_deviation | sign_upper_tol |
| soc-pokec | bc | 1MB | POPT_GE_GRASP | - | 3.0012 | -1.501 | known_deviation | sign_upper_tol |
| web-Google | bc | 4MB | POPT_GE_GRASP | - | 2.6456 | -1.146 | known_deviation | sign_upper_tol |
| web-Google | bc | 8MB | POPT_GE_GRASP | - | 2.6397 | -1.14 | known_deviation | sign_upper_tol |
| com-orkut | bc | 1MB | POPT_GE_GRASP | - | 2.4747 | -0.975 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | cc | 4MB | POPT_GE_GRASP | - | 1.8098 | -0.31 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | bc | 8MB | POPT_GE_GRASP | - | 1.6476 | -0.148 | known_deviation | sign_upper_tol |
| soc-pokec | bc | 4MB | POPT_GE_GRASP | - | 1.6144 | -0.114 | known_deviation | sign_upper_tol |
| web-Google | pr | 4MB | POPT_GE_GRASP | - | 1.0627 | -0.063 | known_deviation | sign_upper_tol |
| cit-Patents | cc | 8MB | POPT_GE_GRASP | - | 1.5277 | -0.028 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | bc | 1MB | POPT_GE_GRASP | - | 1.4819 | 0.018 | ok | sign_upper_tol |
| soc-LiveJournal1 | bc | 4MB | POPT_GE_GRASP | - | 1.4449 | 0.055 | ok | sign_upper_tol |
| cit-Patents | sssp | 8MB | POPT_GE_GRASP | - | 1.4368 | 0.063 | ok | sign_upper_tol |
| cit-Patents | sssp | 4MB | POPT_GE_GRASP | - | 1.4111 | 0.089 | ok | sign_upper_tol |
| soc-LiveJournal1 | cc | 1MB | POPT_GE_GRASP | - | 1.3976 | 0.102 | ok | sign_upper_tol |
| cit-Patents | bc | 8MB | POPT_GE_GRASP | - | 1.3418 | 0.158 | ok | sign_upper_tol |
| soc-pokec | pr | 8MB | POPT_GE_GRASP | - | 0.8187 | 0.181 | ok | sign_upper_tol |
| web-Google | cc | 1MB | POPT_GE_GRASP | - | 1.2675 | 0.232 | ok | sign_upper_tol |
| soc-pokec | bfs | 1MB | POPT_GE_GRASP | - | 1.2618 | 0.238 | ok | sign_upper_tol |
| soc-LiveJournal1 | bfs | 4MB | POPT_GE_GRASP | - | 1.2405 | 0.26 | ok | sign_upper_tol |
| soc-LiveJournal1 | sssp | 4MB | POPT_GE_GRASP | - | 1.2199 | 0.28 | ok | sign_upper_tol |
| soc-LiveJournal1 | bfs | 8MB | POPT_GE_GRASP | - | 1.2003 | 0.3 | ok | sign_upper_tol |
| soc-LiveJournal1 | bfs | 1MB | POPT_GE_GRASP | - | 1.0899 | 0.41 | ok | sign_upper_tol |
| web-Google | bfs | 1MB | POPT_GE_GRASP | - | 1.0462 | 0.454 | ok | sign_upper_tol |
| cit-Patents | bc | 1MB | GRASP | - | 1.4201 | 0.58 | within_tolerance | sign_upper_min_abs |
| cit-Patents | bfs | 4MB | POPT_GE_GRASP | - | 0.8973 | 0.603 | ok | sign_upper_tol |
| soc-pokec | sssp | 4MB | POPT_GE_GRASP | - | 0.8837 | 0.616 | ok | sign_upper_tol |
| soc-pokec | pr | 1MB | POPT_GE_GRASP | - | 0.3474 | 0.653 | ok | sign_upper_tol |
| delaunay_n19 | pr | 16kB | POPT_GE_GRASP | - | 0.3319 | 0.668 | ok | sign_upper_tol |
| soc-LiveJournal1 | sssp | 8MB | POPT_GE_GRASP | - | 0.7472 | 0.753 | ok | sign_upper_tol |
| cit-Patents | bfs | 8MB | POPT_GE_GRASP | - | 0.7344 | 0.766 | ok | sign_upper_tol |
| com-orkut | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 6.1871 | 0.813 | ok | near_grasp_upper |
| cit-Patents | pr | 8MB | GRASP | - | -11.1489 | 0.851 | ok | magnitude_max_abs |
| delaunay_n19 | pr | 4kB | POPT_GE_GRASP | - | 0.0789 | 0.921 | ok | sign_upper_tol |
| roadNet-CA | pr | 16kB | POPT_GE_GRASP | - | 0.0314 | 0.969 | ok | sign_upper_tol |
| cit-Patents | bc | 4MB | POPT_GE_GRASP | - | 0.5181 | 0.982 | ok | sign_upper_tol |
