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
| `soc-LiveJournal1` | bfs | 1MB | POPT | — | — | 🟡 single |
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
| `cache_sim` | 0.8990 | 0.8927 | 0.9024 | 0.8976 | — | **SRRIP** |

### `cit-Patents` / bfs / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.9665 | 0.9625 | 0.9565 | 0.9576 | — | **GRASP** |

### `cit-Patents` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8923 | 0.8780 | 0.7858 | 0.7753 | — | **POPT** |

### `cit-Patents` / sssp / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8529 | 0.8497 | 0.8184 | 0.8270 | — | **GRASP** |

### `com-orkut` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.7104 | 0.6686 | 0.6867 | 0.6184 | — | **POPT** |

### `soc-LiveJournal1` / bc / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8432 | 0.8250 | 0.7949 | 0.8148 | — | **GRASP** |

### `soc-LiveJournal1` / bfs / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.7987 | 0.7829 | 0.7871 | 0.7716 | — | **POPT** |

### `soc-LiveJournal1` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.7322 | 0.6994 | 0.6843 | 0.6248 | — | **POPT** |

### `soc-LiveJournal1` / sssp / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.6877 | 0.6871 | 0.7083 | 0.6730 | — | **POPT** |

### `soc-pokec` / bc / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.8520 | 0.8288 | 0.7651 | 0.7959 | — | **GRASP** |

### `soc-pokec` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.6796 | 0.6344 | 0.5434 | 0.5476 | — | **GRASP** |

### `soc-pokec` / sssp / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.6397 | 0.6080 | 0.5294 | 0.5703 | — | **GRASP** |

### `web-Google` / bc / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.7073 | 0.6889 | 0.7075 | 0.7023 | — | **SRRIP** |

### `web-Google` / bfs / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.9700 | 0.9693 | 0.9363 | 0.9473 | — | **GRASP** |

### `web-Google` / pr / 1MB

| sim | LRU | SRRIP | GRASP | POPT | ECG_DBG_PRIMARY | winner |
|---|---:|---:|---:|---:|---:|---|
| `cache_sim` | 0.6009 | 0.5439 | 0.4532 | 0.4167 | — | **POPT** |
