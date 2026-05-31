# Per-(app, policy) oracle-gap AUC across L3 sweep

Source: `wiki/data/oracle_gap.json`  •  Paper L3 scope: 1MB, 4MB, 8MB

AUC = trapezoidal area on x=log2(L3 size in MB), y=mean gap_pp across graphs at that (app, policy, L3). Units: **gap_pp × log2(MB)** (smaller = closer to offline oracle).

## AUC winner per app

| app | AUC winner | winner AUC | runner-up | runner-up AUC | win/run ratio | win/LRU ratio | AUC savings vs LRU |
|---|---|---:|---|---:|---:|---:|---:|
| bc | **SRRIP** | 5.7773 | GRASP | 6.0079 | 0.9616 | 0.525 | 5.2274 |
| bfs | **POPT** | 6.0073 | GRASP | 12.8861 | 0.4662 | 0.4058 | 8.7975 |
| cc | **GRASP** | 1.5205 | POPT | 13.7625 | 0.1105 | 0.0616 | 23.1463 |
| pr | **POPT** | 0.3774 | GRASP | 9.4331 | 0.04 | 0.0136 | 27.3798 |
| sssp | **POPT** | 3.655 | SRRIP | 7.1463 | 0.5115 | 0.3204 | 7.7525 |

## Per-app per-policy AUC ranking

### bc

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| SRRIP | 5.7773 | 1.7656 → 2.0555 → 1.857 |
| GRASP | 6.0079 | 6.0079 → 0.0 → 0.0 |
| POPT | 8.0712 | 4.4764 → 1.8168 → 1.739 |
| LRU | 11.0047 | 3.3026 → 3.9427 → 3.5763 |

### bfs

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| POPT | 6.0073 | 5.3117 → 0.3562 → 0.3227 |
| GRASP | 12.8861 | 12.3081 → 0.1443 → 0.723 |
| SRRIP | 13.7333 | 2.1464 → 5.6962 → 6.0853 |
| LRU | 14.8048 | 2.1866 → 6.2392 → 6.519 |

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
| POPT | 0.3774 | 0.0434 → 0.1772 → 0.1365 |
| GRASP | 9.4331 | 3.6149 → 3.465 → 1.2415 |
| SRRIP | 19.7583 | 8.2918 → 6.508 → 3.4092 |
| LRU | 27.7572 | 10.9053 → 9.4045 → 5.4905 |

### sssp

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| POPT | 3.655 | 2.3642 → 0.7032 → 0.472 |
| SRRIP | 7.1463 | 2.7783 → 2.4884 → 1.2708 |
| LRU | 11.4075 | 4.3035 → 4.0072 → 2.1864 |
| GRASP | 12.7987 | 12.4487 → 0.2324 → 0.0028 |

## Interpretation

- AUC < 1 (in gap_pp × log2(MB) units) means the policy tracks oracle *very* closely on average across the cache sweep — only pr/POPT currently achieves this.
- AUC savings vs LRU = how many `gap_pp × log2(MB)` units the winner saves over LRU integrated across the sweep. A large value indicates a policy that is closer to oracle at *every* paper L3 size.
