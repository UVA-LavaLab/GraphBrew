# Per-(app, policy) oracle-gap AUC across L3 sweep

Source: `wiki/data/oracle_gap.json`  •  Paper L3 scope: 1MB, 4MB, 8MB

AUC = trapezoidal area on x=log2(L3 size in MB), y=mean gap_pp across graphs at that (app, policy, L3). Units: **gap_pp × log2(MB)** (smaller = closer to offline oracle).

## AUC winner per app

| app | AUC winner | winner AUC | runner-up | runner-up AUC | win/run ratio | win/LRU ratio | AUC savings vs LRU |
|---|---|---:|---|---:|---:|---:|---:|
| bc | **GRASP** | 5.8506 | SRRIP | 5.9964 | 0.9757 | 0.5179 | 5.4452 |
| bfs | **POPT** | 5.3166 | SRRIP | 12.4013 | 0.4287 | 0.42 | 7.3416 |
| cc | **GRASP** | 1.5205 | POPT | 13.7625 | 0.1105 | 0.0616 | 23.1463 |
| pr | **POPT** | 0.3965 | GRASP | 9.1362 | 0.0434 | 0.0145 | 26.965 |
| sssp | **POPT** | 3.6501 | SRRIP | 7.0197 | 0.52 | 0.3465 | 6.883 |

## Per-app per-policy AUC ranking

### bc

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| GRASP | 5.8506 | 5.8506 → 0.0 → 0.0 |
| SRRIP | 5.9964 | 1.8683 → 2.1402 → 1.8357 |
| POPT | 8.7662 | 4.559 → 2.1285 → 2.029 |
| LRU | 11.2958 | 3.4753 → 3.9845 → 3.6875 |

### bfs

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| POPT | 5.3166 | 5.164 → 0.0633 → 0.1152 |
| SRRIP | 12.4013 | 1.6914 → 5.2022 → 5.8132 |
| LRU | 12.6582 | 1.4549 → 5.4555 → 6.0402 |
| GRASP | 13.539 | 12.5304 → 0.3795 → 0.8787 |

### cc

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| GRASP | 1.5205 | 1.52 → 0.0 → 0.001 |
| POPT | 13.7625 | 6.341 → 4.225 → 2.168 |
| SRRIP | 17.0808 | 5.0137 → 6.4844 → 4.681 |
| LRU | 24.6668 | 7.6288 → 8.7828 → 7.7276 |

### pr

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| POPT | 0.3965 | 0.0526 → 0.1828 → 0.1393 |
| GRASP | 9.1362 | 3.5251 → 3.346 → 1.1842 |
| SRRIP | 19.4213 | 8.1867 → 6.3695 → 3.3605 |
| LRU | 27.3615 | 10.7899 → 9.2463 → 5.4042 |

### sssp

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| POPT | 3.6501 | 2.509 → 0.6016 → 0.4774 |
| SRRIP | 7.0197 | 2.766 → 2.3996 → 1.3086 |
| LRU | 10.5331 | 3.8558 → 3.7368 → 2.1442 |
| GRASP | 13.3838 | 13.0337 → 0.2324 → 0.003 |

## Interpretation

- AUC < 1 (in gap_pp × log2(MB) units) means the policy tracks oracle *very* closely on average across the cache sweep — only pr/POPT currently achieves this.
- AUC savings vs LRU = how many `gap_pp × log2(MB)` units the winner saves over LRU integrated across the sweep. A large value indicates a policy that is closer to oracle at *every* paper L3 size.

