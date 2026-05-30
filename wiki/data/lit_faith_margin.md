# Literature-faithfulness margin audit

Per-claim distance to the nearest disagree boundary (threshold for *fragile*: < 1.0 pp).

## Summary

- Total claims: **330**
- Claims with a bounded margin: **330**
- Unbounded magnitude-only (`~` with no `max_abs`): **0**
- **Fragile** cells (< 1.0 pp from disagree boundary): **47**
- `ok`-status cells with negative margin (classifier/audit disagreement): **0**

### Margin distribution (pp)

| stat | value |
|---|---|
| min | -9.5164 |
| p10 | 0.2376 |
| p25 | 1.6221 |
| median | 5.5328 |
| mean | 6.4249 |
| p75 | 8.7454 |
| p90 | 10.6526 |
| max | 80.1168 |

## Per-status margin

| status | count | min | median | mean | max |
|---|---|---|---|---|---|
| known_deviation | 30 | -9.5164 | -1.5812 | -2.5638 | -0.0277 |
| ok | 298 | 0.1024 | 6.2995 | 7.3625 | 80.1168 |
| within_tolerance | 2 | 1.466 | 1.5644 | 1.5644 | 1.6629 |

## Per-family margin

| family | count | min | median | mean | max |
|---|---|---|---|---|---|
| citation | 52 | -7.2258 | 4.2815 | 4.5448 | 15.2458 |
| mesh | 10 | 0.6681 | 5.1988 | 5.3739 | 10.8674 |
| road | 50 | 0.9686 | 9.7976 | 15.0213 | 80.1168 |
| social | 168 | -9.5164 | 4.9436 | 4.7372 | 27.7515 |
| web | 50 | -1.5498 | 6.6315 | 5.6648 | 12.9997 |

## Binding-boundary breakdown

Which side of the claim envelope is currently nearest (the boundary that defines each cell's margin):

| boundary | count |
|---|---|
| sign_upper_tol | 114 |
| trigger_headroom | 98 |
| magnitude_max_abs | 86 |
| sign_upper_min_abs | 16 |
| near_grasp_upper | 16 |

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
| soc-pokec | sssp | 1MB | POPT_GE_GRASP | - | 4.0968 | -2.597 | known_deviation | sign_upper_tol |
| cit-Patents | cc | 4MB | POPT_GE_GRASP | - | 3.6479 | -2.148 | known_deviation | sign_upper_tol |
| cit-Patents | cc | 1MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 8.7258 | -1.726 | known_deviation | near_grasp_upper |
| soc-LiveJournal1 | cc | 8MB | POPT_GE_GRASP | - | 3.0832 | -1.583 | known_deviation | sign_upper_tol |
| soc-pokec | bc | 1MB | POPT_GE_GRASP | - | 3.0792 | -1.579 | known_deviation | sign_upper_tol |
| web-Google | bc | 8MB | POPT_GE_GRASP | - | 3.0498 | -1.55 | known_deviation | sign_upper_tol |
| web-Google | bc | 4MB | POPT_GE_GRASP | - | 2.9017 | -1.402 | known_deviation | sign_upper_tol |
| cit-Patents | bc | 8MB | POPT_GE_GRASP | - | 2.5292 | -1.029 | known_deviation | sign_upper_tol |
| com-orkut | bc | 1MB | POPT_GE_GRASP | - | 2.4747 | -0.975 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | bc | 1MB | POPT_GE_GRASP | - | 1.9853 | -0.485 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | bc | 4MB | POPT_GE_GRASP | - | 1.8769 | -0.377 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | cc | 4MB | POPT_GE_GRASP | - | 1.8098 | -0.31 | known_deviation | sign_upper_tol |
| cit-Patents | sssp | 4MB | POPT_GE_GRASP | - | 1.8038 | -0.304 | known_deviation | sign_upper_tol |
| soc-pokec | bc | 4MB | POPT_GE_GRASP | - | 1.7174 | -0.217 | known_deviation | sign_upper_tol |
| cit-Patents | sssp | 8MB | POPT_GE_GRASP | - | 1.633 | -0.133 | known_deviation | sign_upper_tol |
| cit-Patents | bc | 4MB | POPT_GE_GRASP | - | 1.597 | -0.097 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | bc | 8MB | POPT_GE_GRASP | - | 1.5828 | -0.083 | known_deviation | sign_upper_tol |
| web-Google | pr | 4MB | POPT_GE_GRASP | - | 1.0722 | -0.072 | known_deviation | sign_upper_tol |
| cit-Patents | cc | 8MB | POPT_GE_GRASP | - | 1.5277 | -0.028 | known_deviation | sign_upper_tol |
| soc-LiveJournal1 | cc | 1MB | POPT_GE_GRASP | - | 1.3976 | 0.102 | ok | sign_upper_tol |
| soc-pokec | pr | 8MB | POPT_GE_GRASP | - | 0.8143 | 0.186 | ok | sign_upper_tol |
| web-Google | cc | 1MB | POPT_GE_GRASP | - | 1.2675 | 0.232 | ok | sign_upper_tol |
| soc-pokec | bfs | 1MB | POPT_GE_GRASP | - | 1.2618 | 0.238 | ok | sign_upper_tol |
| web-Google | bfs | 1MB | POPT_GE_GRASP | - | 1.0982 | 0.402 | ok | sign_upper_tol |
| soc-pokec | pr | 1MB | POPT_GE_GRASP | - | 0.4214 | 0.579 | ok | sign_upper_tol |
| soc-pokec | sssp | 4MB | POPT_GE_GRASP | - | 0.8724 | 0.628 | ok | sign_upper_tol |
| cit-Patents | sssp | 1MB | POPT_GE_GRASP | - | 0.8634 | 0.637 | ok | sign_upper_tol |
| delaunay_n19 | pr | 16kB | POPT_GE_GRASP | - | 0.3319 | 0.668 | ok | sign_upper_tol |
| cit-Patents | bfs | 8MB | POPT_GE_GRASP | - | 0.6887 | 0.811 | ok | sign_upper_tol |
| com-orkut | cc | 8MB | POPT_NEAR_GRASP_IF_BIG_GAP | ~ | 6.1871 | 0.813 | ok | near_grasp_upper |
| delaunay_n19 | pr | 4kB | POPT_GE_GRASP | - | 0.0789 | 0.921 | ok | sign_upper_tol |
| soc-LiveJournal1 | sssp | 8MB | POPT_GE_GRASP | - | 0.5775 | 0.922 | ok | sign_upper_tol |
| roadNet-CA | pr | 16kB | POPT_GE_GRASP | - | 0.0314 | 0.969 | ok | sign_upper_tol |
| email-Eu-core | pr | 4MB | POPT_GE_GRASP | - | 0.025 | 0.975 | ok | sign_upper_tol |
| cit-Patents | pr | 8MB | GRASP | - | -11.0247 | 0.975 | ok | magnitude_max_abs |
| email-Eu-core | pr | 8MB | POPT_GE_GRASP | - | 0.0218 | 0.978 | ok | sign_upper_tol |
