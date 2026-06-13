# Oracle-gap trajectory curvature / knee (gate 58)

Per-(app, policy) discrete second derivative of the oracle-gap trajectory at the 4MB midpoint, on a log2-MB L3 axis. Positive curvature → trajectory is bending up (diminishing returns / the knee). Negative curvature → trajectory still accelerating its descent (no plateau visible).

- source: `wiki/data/oracle_gap_auc.json`
- knee threshold: curvature ≥ 0.05 pp/octave^2
- cells total: 20
- cells with knee: 7
- knee rank by policy: ['GRASP', 'POPT', 'SRRIP', 'LRU']
- verdict: **PASS**
- cross-gate-55 consistency: lead_agrees=False; gate55 rank=['POPT', 'GRASP', 'LRU', 'SRRIP']; gate58 rank=['GRASP', 'POPT', 'SRRIP', 'LRU']

## Per-policy summary

| policy | n_cells | knee_count | mean curv | median curv |
| --- | ---: | ---: | ---: | ---: |
| GRASP | 5 | 4 | 2.3923 | 2.0166 |
| LRU | 5 | 0 | -2.0337 | -1.8197 |
| POPT | 5 | 3 | -0.2099 | 0.2792 |
| SRRIP | 5 | 0 | -1.33 | -1.1572 |

## Per-(app, policy) detail

| app | policy | gap1 | gap4 | gap8 | s01 | s12 | curv | knee |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| bc | GRASP | 2.6201 | 0.3287 | 0.8677 | -1.1457 | 0.539 | 1.6847 | ✅ |
| bc | LRU | 1.9817 | 3.8195 | 3.2895 | 0.9189 | -0.53 | -1.4489 |   |
| bc | POPT | 3.9859 | 4.298 | 5.9145 | 0.1561 | 1.6165 | 1.4605 | ✅ |
| bc | SRRIP | 1.431 | 1.9258 | 1.402 | 0.2474 | -0.5238 | -0.7712 |   |
| bfs | GRASP | 8.9409 | 2.5013 | 5.0428 | -3.2198 | 2.5415 | 5.7613 | ✅ |
| bfs | LRU | 2.047 | 5.7147 | 5.7288 | 1.8338 | 0.0141 | -1.8197 |   |
| bfs | POPT | 6.01 | 1.0182 | 0.1677 | -2.4959 | -0.8505 | 1.6454 | ✅ |
| bfs | SRRIP | 2.0676 | 5.0738 | 5.2005 | 1.5031 | 0.1267 | -1.3764 |   |
| cc | GRASP | 1.3668 | 4.99 | 8.8182 | 1.8116 | 3.8282 | 2.0166 | ✅ |
| cc | LRU | 7.2665 | 8.0118 | 5.1746 | 0.3726 | -2.8372 | -3.2098 |   |
| cc | POPT | 0.9595 | 3.6222 | 1.0872 | 1.3314 | -2.535 | -3.8664 |   |
| cc | SRRIP | 4.6833 | 5.1256 | 2.3776 | 0.2212 | -2.748 | -2.9692 |   |
| pr | GRASP | 2.5266 | 2.4708 | 1.4323 | -0.0279 | -1.0385 | -1.0106 |   |
| pr | LRU | 10.5463 | 8.3747 | 4.897 | -1.0858 | -3.4777 | -2.3919 |   |
| pr | POPT | 0.6986 | 0.1102 | 0.0952 | -0.2942 | -0.015 | 0.2792 | ✅ |
| pr | SRRIP | 8.0645 | 5.2245 | 2.6473 | -1.42 | -2.5772 | -1.1572 |   |
| sssp | GRASP | 9.3045 | 1.1628 | 0.6014 | -4.0709 | -0.5614 | 3.5095 | ✅ |
| sssp | LRU | 5.6042 | 3.7118 | 1.4676 | -0.9462 | -2.2442 | -1.298 |   |
| sssp | POPT | 5.4618 | 3.5492 | 2.0248 | -0.9563 | -1.5244 | -0.5681 |   |
| sssp | SRRIP | 3.981 | 2.0178 | 0.6602 | -0.9816 | -1.3576 | -0.376 |   |
