# Oracle-gap trajectory curvature / knee (gate 58)

Per-(app, policy) discrete second derivative of the oracle-gap trajectory at the 4MB midpoint, on a log2-MB L3 axis. Positive curvature → trajectory is bending up (diminishing returns / the knee). Negative curvature → trajectory still accelerating its descent (no plateau visible).

- source: `wiki/data/oracle_gap_auc.json`
- knee threshold: curvature ≥ 0.05 pp/octave^2
- cells total: 20
- cells with knee: 7
- knee rank by policy: ['GRASP', 'POPT', 'SRRIP', 'LRU']
- verdict: **PASS**
- cross-gate-55 consistency: lead_agrees=False; gate55 rank=['LRU', 'SRRIP', 'GRASP', 'POPT']; gate58 rank=['GRASP', 'POPT', 'SRRIP', 'LRU']

## Per-policy summary

| policy | n_cells | knee_count | mean curv | median curv |
| --- | ---: | ---: | ---: | ---: |
| GRASP | 5 | 4 | 1.8029 | 1.8245 |
| LRU | 5 | 0 | -1.974 | -1.3092 |
| POPT | 5 | 3 | 0.6005 | 0.8639 |
| SRRIP | 5 | 0 | -1.2703 | -0.9065 |

## Per-(app, policy) detail

| app | policy | gap1 | gap4 | gap8 | s01 | s12 | curv | knee |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| bc | GRASP | 2.7304 | 0.3287 | 0.9523 | -1.2008 | 0.6236 | 1.8245 | ✅ |
| bc | LRU | 2.0921 | 3.8195 | 3.374 | 0.8637 | -0.4455 | -1.3092 |   |
| bc | POPT | 2.7323 | 1.6323 | 2.7507 | -0.55 | 1.1184 | 1.6684 | ✅ |
| bc | SRRIP | 1.5413 | 1.9258 | 1.4865 | 0.1923 | -0.4393 | -0.6316 |   |
| bfs | GRASP | 8.9621 | 3.3765 | 5.6495 | -2.7928 | 2.273 | 5.0658 | ✅ |
| bfs | LRU | 2.0684 | 6.5898 | 6.3357 | 2.2607 | -0.2541 | -2.5148 |   |
| bfs | POPT | 5.2856 | 0.5118 | 0.0 | -2.3869 | -0.5118 | 1.8751 | ✅ |
| bfs | SRRIP | 2.0889 | 5.949 | 5.8072 | 1.93 | -0.1418 | -2.0718 |   |
| cc | GRASP | 1.6668 | 0.0 | 0.0 | -0.8334 | 0.0 | 0.8334 | ✅ |
| cc | LRU | 11.2803 | 10.9256 | 9.6012 | -0.1774 | -1.3244 | -1.147 |   |
| cc | POPT | 2.7082 | 4.0764 | 3.3274 | 0.6841 | -0.749 | -1.4331 |   |
| cc | SRRIP | 8.6973 | 8.0394 | 6.804 | -0.3289 | -1.2354 | -0.9065 |   |
| pr | GRASP | 4.6235 | 4.7063 | 2.4117 | 0.0414 | -2.2946 | -2.336 |   |
| pr | LRU | 12.6431 | 10.6102 | 5.8758 | -1.0164 | -4.7344 | -3.718 |   |
| pr | POPT | 0.0 | 0.0 | 0.0283 | 0.0 | 0.0283 | 0.0283 |   |
| pr | SRRIP | 10.1615 | 7.4598 | 3.6263 | -1.3508 | -3.8335 | -2.4827 |   |
| sssp | GRASP | 9.4935 | 1.2254 | 0.718 | -4.134 | -0.5074 | 3.6266 | ✅ |
| sssp | LRU | 5.7932 | 3.7744 | 1.5842 | -1.0094 | -2.1902 | -1.1808 |   |
| sssp | POPT | 2.3393 | 0.2308 | 0.0404 | -1.0543 | -0.1904 | 0.8639 | ✅ |
| sssp | SRRIP | 4.1698 | 2.0806 | 0.777 | -1.0446 | -1.3036 | -0.259 |   |
