# Per-(app, policy) oracle-gap AUC across L3 sweep

Source: `wiki/data/oracle_gap.json`  •  Paper L3 scope: 1MB, 4MB, 8MB

AUC = trapezoidal area on x=log2(L3 size in MB), y=mean gap_pp across graphs at that (app, policy, L3). Units: **gap_pp × log2(MB)** (smaller = closer to offline oracle).

## AUC winner per app

| app | AUC winner | winner AUC | runner-up | runner-up AUC | win/run ratio | win/LRU ratio | AUC savings vs LRU |
|---|---|---:|---|---:|---:|---:|---:|
| bc | **GRASP** | 3.6996 | SRRIP | 5.1733 | 0.7151 | 0.3891 | 5.8088 |
| bfs | **POPT** | 6.0533 | SRRIP | 13.9159 | 0.435 | 0.4003 | 9.0677 |
| cc | **GRASP** | 1.6668 | POPT | 10.4865 | 0.1589 | 0.0513 | 30.8025 |
| pr | **POPT** | 0.0142 | GRASP | 12.8888 | 0.0011 | 0.0005 | 31.4821 |
| sssp | **POPT** | 2.7057 | SRRIP | 7.6792 | 0.3523 | 0.2209 | 9.5412 |

## Per-app per-policy AUC ranking

### bc

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| GRASP | 3.6996 | 2.7304 → 0.3287 → 0.9523 |
| SRRIP | 5.1733 | 1.5413 → 1.9258 → 1.4865 |
| POPT | 6.5561 | 2.7323 → 1.6323 → 2.7507 |
| LRU | 9.5084 | 2.0921 → 3.8195 → 3.374 |

### bfs

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| POPT | 6.0533 | 5.2856 → 0.5118 → 0.0 |
| SRRIP | 13.9159 | 2.0889 → 5.949 → 5.8072 |
| LRU | 15.121 | 2.0684 → 6.5898 → 6.3357 |
| GRASP | 16.8516 | 8.9621 → 3.3765 → 5.6495 |

### cc

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| GRASP | 1.6668 | 1.6668 → 0.0 → 0.0 |
| POPT | 10.4865 | 2.7082 → 4.0764 → 3.3274 |
| SRRIP | 24.1584 | 8.6973 → 8.0394 → 6.804 |
| LRU | 32.4693 | 11.2803 → 10.9256 → 9.6012 |

### pr

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| POPT | 0.0142 | 0.0 → 0.0 → 0.0283 |
| GRASP | 12.8888 | 4.6235 → 4.7063 → 2.4117 |
| SRRIP | 23.1644 | 10.1615 → 7.4598 → 3.6263 |
| LRU | 31.4963 | 12.6431 → 10.6102 → 5.8758 |

### sssp

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| POPT | 2.7057 | 2.3393 → 0.2308 → 0.0404 |
| SRRIP | 7.6792 | 4.1698 → 2.0806 → 0.777 |
| GRASP | 11.6906 | 9.4935 → 1.2254 → 0.718 |
| LRU | 12.2469 | 5.7932 → 3.7744 → 1.5842 |

## Interpretation

- AUC < 1 (in gap_pp × log2(MB) units) means the policy tracks oracle *very* closely on average across the cache sweep — only pr/POPT currently achieves this.
- AUC savings vs LRU = how many `gap_pp × log2(MB)` units the winner saves over LRU integrated across the sweep. A large value indicates a policy that is closer to oracle at *every* paper L3 size.
