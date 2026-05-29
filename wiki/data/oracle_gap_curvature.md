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
| GRASP | 5 | 4 | 2.872 | 2.9253 |
| LRU | 5 | 0 | -1.6406 | -1.5331 |
| POPT | 5 | 3 | 0.688 | 0.8295 |
| SRRIP | 5 | 0 | -1.4264 | -1.1444 |

## Per-(app, policy) detail

| app | policy | gap1 | gap4 | gap8 | s01 | s12 | curv | knee |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| bc | GRASP | 5.8506 | 0.0 | 0.0 | -2.9253 | 0.0 | 2.9253 | ✅ |
| bc | LRU | 3.4753 | 3.9845 | 3.6875 | 0.2546 | -0.297 | -0.5516 |   |
| bc | POPT | 4.559 | 2.1285 | 2.029 | -1.2153 | -0.0995 | 1.1158 | ✅ |
| bc | SRRIP | 1.8683 | 2.1402 | 1.8357 | 0.136 | -0.3045 | -0.4405 |   |
| bfs | GRASP | 12.5304 | 0.3795 | 0.8787 | -6.0755 | 0.4992 | 6.5747 | ✅ |
| bfs | LRU | 1.4549 | 5.4555 | 6.0402 | 2.0003 | 0.5847 | -1.4156 |   |
| bfs | POPT | 5.164 | 0.0633 | 0.1152 | -2.5503 | 0.0519 | 2.6022 | ✅ |
| bfs | SRRIP | 1.6914 | 5.2022 | 5.8132 | 1.7554 | 0.611 | -1.1444 |   |
| cc | GRASP | 1.52 | 0.0 | 0.001 | -0.76 | 0.001 | 0.761 | ✅ |
| cc | LRU | 7.6288 | 8.7828 | 7.7276 | 0.577 | -1.0552 | -1.6322 |   |
| cc | POPT | 6.341 | 4.225 | 2.168 | -1.058 | -2.057 | -0.999 |   |
| cc | SRRIP | 5.0137 | 6.4844 | 4.681 | 0.7353 | -1.8034 | -2.5387 |   |
| pr | GRASP | 3.5251 | 3.346 | 1.1842 | -0.0896 | -2.1618 | -2.0723 |   |
| pr | LRU | 10.7899 | 9.2463 | 5.4042 | -0.7718 | -3.8421 | -3.0703 |   |
| pr | POPT | 0.0526 | 0.1828 | 0.1393 | 0.0651 | -0.0435 | -0.1086 |   |
| pr | SRRIP | 8.1867 | 6.3695 | 3.3605 | -0.9086 | -3.009 | -2.1004 |   |
| sssp | GRASP | 13.0337 | 0.2324 | 0.003 | -6.4006 | -0.2294 | 6.1712 | ✅ |
| sssp | LRU | 3.8558 | 3.7368 | 2.1442 | -0.0595 | -1.5926 | -1.5331 |   |
| sssp | POPT | 2.509 | 0.6016 | 0.4774 | -0.9537 | -0.1242 | 0.8295 | ✅ |
| sssp | SRRIP | 2.766 | 2.3996 | 1.3086 | -0.1832 | -1.091 | -0.9078 |   |

