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
| `cit-Patents` | bc | 1MB | GRASP | — | — | 🟡 single |
| `cit-Patents` | bfs | 1MB | GRASP | — | — | 🟡 single |
| `cit-Patents` | pr | 1MB | POPT | — | — | 🟡 single |
| `cit-Patents` | sssp | 1MB | GRASP | — | — | 🟡 single |
| `com-orkut` | pr | 1MB | GRASP | — | — | 🟡 single |
| `soc-LiveJournal1` | bc | 1MB | GRASP | — | — | 🟡 single |
| `soc-LiveJournal1` | bfs | 1MB | GRASP | — | — | 🟡 single |
| `soc-LiveJournal1` | pr | 1MB | POPT | — | — | 🟡 single |
| `soc-LiveJournal1` | sssp | 1MB | GRASP | — | — | 🟡 single |
| `soc-pokec` | bc | 1MB | GRASP | — | — | 🟡 single |
| `soc-pokec` | pr | 1MB | GRASP | — | — | 🟡 single |
| `soc-pokec` | sssp | 1MB | GRASP | — | — | 🟡 single |
| `web-Google` | bc | 1MB | LRU | — | — | 🟡 single |
| `web-Google` | bfs | 1MB | GRASP | — | — | 🟡 single |
| `web-Google` | pr | 1MB | POPT | — | — | 🟡 single |

## Per-cell per-sim miss-rate (paper-table preview)

### `cit-Patents` / bc / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.9451 | 0.9426 | 0.9295 | 0.9369 | — | **GRASP** |

### `cit-Patents` / bfs / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.9815 | 0.9791 | 0.9738 | 0.9760 | — | **GRASP** |

### `cit-Patents` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8951 | 0.8803 | 0.8182 | 0.7611 | — | **POPT** |

### `cit-Patents` / sssp / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8768 | 0.8701 | 0.8336 | 0.8593 | — | **GRASP** |

### `com-orkut` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.7426 | 0.7065 | 0.6383 | 0.6738 | — | **GRASP** |

### `soc-LiveJournal1` / bc / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8649 | 0.8501 | 0.8312 | 0.8425 | — | **GRASP** |

### `soc-LiveJournal1` / bfs / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8357 | 0.8107 | 0.7837 | 0.8056 | — | **GRASP** |

### `soc-LiveJournal1` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.7654 | 0.7340 | 0.6825 | 0.6642 | — | **POPT** |

### `soc-LiveJournal1` / sssp / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.7439 | 0.7177 | 0.6820 | 0.7010 | — | **GRASP** |

### `soc-pokec` / bc / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8597 | 0.8442 | 0.8108 | 0.8325 | — | **GRASP** |

### `soc-pokec` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.6962 | 0.6517 | 0.5557 | 0.5761 | — | **GRASP** |

### `soc-pokec` / sssp / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.6719 | 0.6366 | 0.5581 | 0.6294 | — | **GRASP** |

### `web-Google` / bc / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8027 | 0.8069 | 0.8238 | 0.8378 | — | **LRU** |

### `web-Google` / bfs / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.9804 | 0.9796 | 0.9444 | 0.9619 | — | **GRASP** |

### `web-Google` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.5984 | 0.5409 | 0.4509 | 0.4299 | — | **POPT** |
