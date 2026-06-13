# Literature-faithfulness margin audit

Per-claim distance to the nearest disagree boundary (threshold for *fragile*: < 1.0 pp).

## Summary

- Total claims: **279**
- Claims with a bounded margin: **279**
- Unbounded magnitude-only (`~` with no `max_abs`): **0**
- **Fragile** cells (< 1.0 pp from disagree boundary): **49**
- `ok`-status cells with negative margin (classifier/audit disagreement): **0**

### Margin distribution (pp)

| stat | value |
|---|---|
| min | -11.3873 |
| p10 | -1.0874 |
| p25 | 2.2682 |
| median | 5.3906 |
| mean | 5.2103 |
| p75 | 7.6145 |
| p90 | 10.3382 |
| max | 41.0153 |

## Per-status margin

| status | count | min | median | mean | max |
|---|---|---|---|---|---|
| insufficient_data | 28 | 1.0 | 7.0 | 5.2857 | 8.0 |
| known_deviation | 37 | -11.3873 | -2.5405 | -2.8791 | -0.1349 |
| ok | 214 | 0.0953 | 5.8075 | 6.5991 | 41.0153 |

## Per-family margin

| family | count | min | median | mean | max |
|---|---|---|---|---|---|
| citation | 52 | -4.2817 | 5.5565 | 4.6229 | 13.255 |
| social | 177 | -11.3873 | 4.942 | 4.5623 | 41.0153 |
| web | 50 | -6.3909 | 7.5011 | 8.1153 | 30.6346 |

## Binding-boundary breakdown

Which side of the claim envelope is currently nearest (the boundary that defines each cell's margin):

| boundary | count |
|---|---|
| magnitude_max_abs | 94 |
| sign_upper_tol | 84 |
| trigger_headroom | 67 |
| sign_upper_min_abs | 17 |
| near_grasp_upper | 17 |

## Fragile cells (margin < 1.0 pp)

| graph | app | L3 | policy | sign | Δ pp | margin pp | status | binding |
|---|---|---|---|---|---|---|---|---|
| com-orkut | cc | 4MB | POPT_GE_GRASP | - | 12.8873 | -11.387 | known_deviation | sign_upper_tol |
| com-orkut | bc | 8MB | POPT_GE_GRASP | - | 8.9449 | -7.445 | known_deviation | sign_upper_tol |
| web-Google | bc | 8MB | POPT_GE_GRASP | - | 7.8909 | -6.391 | known_deviation | sign_upper_tol |
| com-orkut | cc | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 12.8873 | -5.887 | known_deviation | near_grasp_upper |
| soc-pokec | sssp | 1MB | POPT_GE_GRASP | - | 7.1349 | -5.635 | known_deviation | sign_upper_tol |
| com-orkut | bc | 4MB | POPT_GE_GRASP | - | 6.5042 | -5.004 | known_deviation | sign_upper_tol |
| web-Google | bc | 4MB | POPT_GE_GRASP | - | 6.4403 | -4.94 | known_deviation | sign_upper_tol |
| com-orkut | sssp | 1MB | POPT_GE_GRASP | - | 6.0772 | -4.577 | known_deviation | sign_upper_tol |
| cit-Patents | sssp | 8MB | POPT_GE_GRASP | - | 5.7817 | -4.282 | known_deviation | sign_upper_tol |
| cit-Patents | sssp | 4MB | POPT_GE_GRASP | - | 5.6255 | -4.125 | known_deviation | sign_upper_tol |
| cit-Patents | bc | 8MB | POPT_GE_GRASP | - | 5.2976 | -3.798 | known_deviation | sign_upper_tol |
| soc-pokec | bc | 4MB | POPT_GE_GRASP | - | 4.6776 | -3.178 | known_deviation | sign_upper_tol |
| web-Google | sssp | 1MB | POPT_GE_GRASP | - | 4.6274 | -3.127 | known_deviation | sign_upper_tol |
| com-orkut | cc | 1MB | POPT_GE_GRASP | - | 4.5093 | -3.009 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | cc | 8MB | POPT_GE_GRASP | - | 4.2758 | -2.776 | known_deviation | sign_upper_tol |
| soc-pokec | bc | 8MB | POPT_GE_GRASP | - | 4.2088 | -2.709 | known_deviation | sign_upper_tol |
| com-orkut | sssp | 4MB | POPT_GE_GRASP | - | 4.1055 | -2.606 | known_deviation | sign_upper_tol |
| com-orkut | pr | 1MB | POPT_GE_GRASP | - | 3.5479 | -2.548 | known_deviation | sign_upper_tol |
| soc-pokec | sssp | 4MB | POPT_GE_GRASP | - | 4.0405 | -2.54 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | sssp | 4MB | POPT_GE_GRASP | - | 3.9749 | -2.475 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | bc | 8MB | POPT_GE_GRASP | - | 3.9382 | -2.438 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | bfs | 4MB | POPT_GE_GRASP | - | 3.3583 | -1.858 | known_deviation | sign_upper_tol |
| cit-Patents | bc | 4MB | POPT_GE_GRASP | - | 3.2212 | -1.721 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | bc | 4MB | POPT_GE_GRASP | - | 2.9732 | -1.473 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | sssp | 8MB | POPT_GE_GRASP | - | 2.9246 | -1.425 | known_deviation | sign_upper_tol |
| cit-Patents | cc | 4MB | POPT_GE_GRASP | - | 2.7776 | -1.278 | known_deviation | sign_upper_tol |
| soc-pokec | bfs | 1MB | POPT_GE_GRASP | - | 2.7638 | -1.264 | known_deviation | sign_upper_tol |
| com-orkut | bc | 1MB | POPT_GE_GRASP | - | 2.6801 | -1.18 | known_deviation | sign_upper_tol |
| cit-Patents | sssp | 1MB | POPT_GE_GRASP | - | 2.5642 | -1.064 | known_deviation | sign_upper_tol |
| soc-pokec | pr | 1MB | POPT_GE_GRASP | - | 2.041 | -1.041 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | cc | 4MB | POPT_GE_GRASP | - | 2.4465 | -0.946 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | bfs | 1MB | POPT_GE_GRASP | - | 2.1902 | -0.69 | known_deviation | sign_upper_tol |
| soc-pokec | bc | 1MB | POPT_GE_GRASP | - | 2.1642 | -0.664 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | sssp | 1MB | POPT_GE_GRASP | - | 1.8991 | -0.399 | known_deviation | sign_upper_tol |
| cit-Patents | bfs | 4MB | POPT_GE_GRASP | - | 1.7623 | -0.262 | known_deviation | sign_upper_tol |
| web-Google | bfs | 1MB | POPT_GE_GRASP | - | 1.7476 | -0.248 | known_deviation | sign_upper_tol |
| soc-pokec | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 7.1349 | -0.135 | known_deviation | near_grasp_upper |
| web-Google | bc | 1MB | POPT_GE_GRASP | - | 1.4047 | 0.095 | ok | sign_upper_tol |
| com-orkut | sssp | 8MB | POPT_GE_GRASP | - | 1.2481 | 0.252 | ok | sign_upper_tol |
| soc-pokec | cc | 1MB | POPT_GE_GRASP | - | 1.2477 | 0.252 | ok | sign_upper_tol |
| cit-Patents | cc | 8MB | POPT_GE_GRASP | - | 1.1599 | 0.34 | ok | sign_upper_tol |
| soc-LiveJournal1 | bc | 1MB | POPT_GE_GRASP | - | 1.1347 | 0.365 | ok | sign_upper_tol |
| soc-pokec | pr | 8MB | POPT_GE_GRASP | - | 0.5709 | 0.429 | ok | sign_upper_tol |
| soc-LiveJournal1 | bfs | 8MB | POPT_GE_GRASP | - | 1.0059 | 0.494 | ok | sign_upper_tol |
| com-orkut | bfs | 4MB | POPT_GE_GRASP | - | 0.9888 | 0.511 | ok | sign_upper_tol |
| com-orkut | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 1.558 | 0.513 | ok | trigger_headroom |
| com-orkut | sssp | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 6.0772 | 0.614 | ok | trigger_headroom |
| soc-LiveJournal1 | pr | 4MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 3.6261 | 0.725 | ok | trigger_headroom |
| cit-Patents | bc | 1MB | POPT_GE_GRASP | - | 0.7356 | 0.764 | ok | sign_upper_tol |
