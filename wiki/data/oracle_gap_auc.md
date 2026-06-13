# Per-(app, policy) oracle-gap AUC across L3 sweep

Source: `wiki/data/oracle_gap.json`  •  Paper L3 scope: 1MB, 4MB, 8MB

AUC = trapezoidal area on x=log2(L3 size in MB), y=mean gap_pp across graphs at that (app, policy, L3). Units: **gap_pp × log2(MB)** (smaller = closer to offline oracle).

## AUC winner per app

| app | AUC winner | winner AUC | runner-up | runner-up AUC | win/run ratio | win/LRU ratio | AUC savings vs LRU |
|---|---|---:|---|---:|---:|---:|---:|
| bc | **GRASP** | 3.547 | SRRIP | 5.0207 | 0.7065 | 0.3791 | 5.8087 |
| bfs | **POPT** | 7.6211 | SRRIP | 12.2786 | 0.6207 | 0.5652 | 5.8623 |
| cc | **POPT** | 6.9364 | GRASP | 13.2609 | 0.5231 | 0.3171 | 14.9351 |
| pr | **POPT** | 0.9115 | GRASP | 6.949 | 0.1312 | 0.0357 | 24.6453 |
| sssp | **SRRIP** | 7.3378 | GRASP | 11.3494 | 0.6465 | 0.6163 | 4.5679 |

## Per-app per-policy AUC ranking

### bc

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| GRASP | 3.547 | 2.6201 → 0.3287 → 0.8677 |
| SRRIP | 5.0207 | 1.431 → 1.9258 → 1.402 |
| LRU | 9.3557 | 1.9817 → 3.8195 → 3.2895 |
| POPT | 13.3901 | 3.9859 → 4.298 → 5.9145 |

### bfs

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| POPT | 7.6211 | 6.01 → 1.0182 → 0.1677 |
| SRRIP | 12.2786 | 2.0676 → 5.0738 → 5.2005 |
| LRU | 13.4834 | 2.047 → 5.7147 → 5.7288 |
| GRASP | 15.2143 | 8.9409 → 2.5013 → 5.0428 |

### cc

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| POPT | 6.9364 | 0.9595 → 3.6222 → 1.0872 |
| GRASP | 13.2609 | 1.3668 → 4.99 → 8.8182 |
| SRRIP | 13.5605 | 4.6833 → 5.1256 → 2.3776 |
| LRU | 21.8715 | 7.2665 → 8.0118 → 5.1746 |

### pr

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| POPT | 0.9115 | 0.6986 → 0.1102 → 0.0952 |
| GRASP | 6.949 | 2.5266 → 2.4708 → 1.4323 |
| SRRIP | 17.2249 | 8.0645 → 5.2245 → 2.6473 |
| LRU | 25.5568 | 10.5463 → 8.3747 → 4.897 |

### sssp

| policy | AUC | trajectory (1MB→4MB→8MB) |
|---|---:|---|
| SRRIP | 7.3378 | 3.981 → 2.0178 → 0.6602 |
| GRASP | 11.3494 | 9.3045 → 1.1628 → 0.6014 |
| POPT | 11.798 | 5.4618 → 3.5492 → 2.0248 |
| LRU | 11.9057 | 5.6042 → 3.7118 → 1.4676 |

## Interpretation

- AUC < 1 (in gap_pp × log2(MB) units) means the policy tracks oracle *very* closely on average across the cache sweep — only pr/POPT currently achieves this.
- AUC savings vs LRU = how many `gap_pp × log2(MB)` units the winner saves over LRU integrated across the sweep. A large value indicates a policy that is closer to oracle at *every* paper L3 size.
