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
| GRASP | 5 | 4 | 2.8311 | 3.004 |
| LRU | 5 | 0 | -1.7803 | -1.6727 |
| POPT | 5 | 3 | 0.6378 | 0.5993 |
| SRRIP | 5 | 0 | -1.5095 | -1.3858 |

## Per-(app, policy) detail

| app | policy | gap1 | gap4 | gap8 | s01 | s12 | curv | knee |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| bc | GRASP | 6.0079 | 0.0 | 0.0 | -3.004 | 0.0 | 3.004 | ✅ |
| bc | LRU | 3.3026 | 3.9427 | 3.5763 | 0.32 | -0.3664 | -0.6865 |   |
| bc | POPT | 4.4764 | 1.8168 | 1.739 | -1.3298 | -0.0778 | 1.252 | ✅ |
| bc | SRRIP | 1.7656 | 2.0555 | 1.857 | 0.1449 | -0.1985 | -0.3434 |   |
| bfs | GRASP | 12.3081 | 0.1443 | 0.723 | -6.0819 | 0.5787 | 6.6606 | ✅ |
| bfs | LRU | 2.1866 | 6.2392 | 6.519 | 2.0263 | 0.2798 | -1.7465 |   |
| bfs | POPT | 5.3117 | 0.3562 | 0.3227 | -2.4777 | -0.0335 | 2.4442 | ✅ |
| bfs | SRRIP | 2.1464 | 5.6962 | 6.0853 | 1.7749 | 0.3891 | -1.3858 |   |
| cc | GRASP | 1.52 | 0.0 | 0.001 | -0.76 | 0.001 | 0.761 | ✅ |
| cc | LRU | 7.6288 | 8.7828 | 7.7276 | 0.577 | -1.0552 | -1.6322 |   |
| cc | POPT | 6.341 | 4.225 | 2.168 | -1.058 | -2.057 | -0.999 |   |
| cc | SRRIP | 5.0137 | 6.4844 | 4.681 | 0.7353 | -1.8034 | -2.5387 |   |
| pr | GRASP | 3.6149 | 3.465 | 1.2415 | -0.075 | -2.2235 | -2.1485 |   |
| pr | LRU | 10.9053 | 9.4045 | 5.4905 | -0.7504 | -3.914 | -3.1636 |   |
| pr | POPT | 0.0434 | 0.1772 | 0.1365 | 0.0669 | -0.0407 | -0.1076 |   |
| pr | SRRIP | 8.2918 | 6.508 | 3.4092 | -0.8919 | -3.0988 | -2.2069 |   |
| sssp | GRASP | 12.4487 | 0.2324 | 0.0028 | -6.1082 | -0.2296 | 5.8786 | ✅ |
| sssp | LRU | 4.3035 | 4.0072 | 2.1864 | -0.1481 | -1.8208 | -1.6727 |   |
| sssp | POPT | 2.3642 | 0.7032 | 0.472 | -0.8305 | -0.2312 | 0.5993 | ✅ |
| sssp | SRRIP | 2.7783 | 2.4884 | 1.2708 | -0.145 | -1.2176 | -1.0726 |   |
