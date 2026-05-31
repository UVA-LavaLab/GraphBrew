# Headline parity proof (gate 283)

Cross-simulator headline table at literature L3 scope **1MB**, joining cache_sim + gem5 + Sniper.

## Verdict summary

- Cells total: **15**
- Cells with ≥2 sims reporting: **0**
- ✅ Agree: **0**
- ❌ Disagree: **0**
- 🟡 Single sim only: **15**
- ⚪ Empty: **0**
- Winner agreement: **not yet measurable** (0 cells with multi-sim coverage)

## Per-cell winner across sims

| graph | app | L3 | cache_sim | gem5 | Sniper | verdict |
|---|---|---|---|---|---|---|
| `cit-Patents` | bc | 1MB | SRRIP | — | — | 🟡 single |
| `cit-Patents` | bfs | 1MB | GRASP | — | — | 🟡 single |
| `cit-Patents` | pr | 1MB | POPT | — | — | 🟡 single |
| `cit-Patents` | sssp | 1MB | GRASP | — | — | 🟡 single |
| `com-orkut` | pr | 1MB | POPT | — | — | 🟡 single |
| `soc-LiveJournal1` | bc | 1MB | GRASP | — | — | 🟡 single |
| `soc-LiveJournal1` | bfs | 1MB | GRASP | — | — | 🟡 single |
| `soc-LiveJournal1` | pr | 1MB | POPT | — | — | 🟡 single |
| `soc-LiveJournal1` | sssp | 1MB | POPT | — | — | 🟡 single |
| `soc-pokec` | bc | 1MB | GRASP | — | — | 🟡 single |
| `soc-pokec` | pr | 1MB | GRASP | — | — | 🟡 single |
| `soc-pokec` | sssp | 1MB | GRASP | — | — | 🟡 single |
| `web-Google` | bc | 1MB | SRRIP | — | — | 🟡 single |
| `web-Google` | bfs | 1MB | GRASP | — | — | 🟡 single |
| `web-Google` | pr | 1MB | POPT | — | — | 🟡 single |

## Per-cell per-sim miss-rate (paper-table preview)

### `cit-Patents` / bc / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8843 | 0.8796 | 0.8985 | 0.8825 | — | **SRRIP** |

### `cit-Patents` / bfs / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.9708 | 0.9667 | 0.9593 | 0.9603 | — | **GRASP** |

### `cit-Patents` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8944 | 0.8798 | 0.7855 | 0.7713 | — | **POPT** |

### `cit-Patents` / sssp / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8486 | 0.8444 | 0.8180 | 0.8208 | — | **GRASP** |

### `com-orkut` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.7047 | 0.6625 | 0.6850 | 0.6109 | — | **POPT** |

### `soc-LiveJournal1` / bc / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8396 | 0.8175 | 0.7948 | 0.8096 | — | **GRASP** |

### `soc-LiveJournal1` / bfs / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8176 | 0.7824 | 0.7413 | 0.7521 | — | **GRASP** |

### `soc-LiveJournal1` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.7319 | 0.6987 | 0.6839 | 0.6226 | — | **POPT** |

### `soc-LiveJournal1` / sssp / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.7058 | 0.6799 | 0.6584 | 0.6582 | — | **POPT** |

### `soc-pokec` / bc / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8513 | 0.8285 | 0.7646 | 0.7946 | — | **GRASP** |

### `soc-pokec` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.6796 | 0.6343 | 0.5433 | 0.5468 | — | **GRASP** |

### `soc-pokec` / sssp / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.6351 | 0.6037 | 0.5269 | 0.5650 | — | **GRASP** |

### `web-Google` / bc / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.6998 | 0.6882 | 0.7085 | 0.7036 | — | **SRRIP** |

### `web-Google` / bfs / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.9704 | 0.9699 | 0.9363 | 0.9467 | — | **GRASP** |

### `web-Google` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.6008 | 0.5442 | 0.4531 | 0.4168 | — | **POPT** |
