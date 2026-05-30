# Gate 263 — ECG configuration matrix registry

Status: **active**

## Totals

- n_policies_checked: 13
- n_graphs_checked: 6
- n_kernels_checked: 7
- n_sweep_entries: 12
- n_pairs_checked: 21
- n_ecg_modes: 4
- default_l3_bytes: 8388608 (tier: 8MB)

## Rules

- **C1** — every BASELINE_POLICIES + GRAPH_AWARE_POLICIES + PREVIEW_POLICIES entry is in gate 255 CANONICAL_POLICY_NAMES
- **C2** — every EVAL_GRAPHS.name is in gate 258 CANONICAL_GRAPHS
- **C3** — every BENCHMARKS entry is a key of gate 260 KERNEL_CL_CLASS
- **C4** — DEFAULT_CACHE.CACHE_L3_SIZE parses to bytes matching exactly one tier in gate 251 CANONICAL_L3_TIERS
- **C5** — every CACHE_SIZES_SWEEP entry is a power-of-2; the sweep brackets the L3 anchor (min ≤ DEFAULT_L3 ≤ max); strictly increasing
- **C6** — every (reorder, policy, ...) pair across ACCURACY_PAIRS + REORDER_POLICY_PAIRS uses a policy in ALL_POLICIES
- **C7** — every ECG_MODE observed in ACCURACY_PAIRS env dicts is in ECG_MODES; every ECG_MODES entry is paper-pipeline-recognised as ECG_<mode> OR in ECG_PRIVATE_MODES

## Policy vocab

- BASELINE_POLICIES: `['LRU', 'FIFO', 'RANDOM', 'LFU', 'SRRIP']`
- GRAPH_AWARE_POLICIES: `['GRASP', 'POPT', 'ECG']`
- PREVIEW_POLICIES: `['LRU', 'SRRIP', 'GRASP', 'POPT', 'ECG']`
- ALL_POLICIES: `['LRU', 'FIFO', 'RANDOM', 'LFU', 'SRRIP', 'GRASP', 'POPT', 'ECG']`

## ECG modes

- ECG_MODES (declared): `['DBG_PRIMARY', 'POPT_PRIMARY', 'DBG_ONLY', 'ECG_EMBEDDED']`
- ECG_PRIVATE_MODES: `['ECG_EMBEDDED']`
- observed in ACCURACY_PAIRS: `['DBG_ONLY', 'DBG_PRIMARY', 'POPT_PRIMARY']`

## Benchmarks + graphs

- BENCHMARKS: `['pr', 'pr_spmv', 'bfs', 'cc', 'cc_sv', 'sssp', 'bc']`
- EVAL_GRAPHS: `['soc-pokec', 'soc-LiveJournal1', 'com-orkut', 'cit-Patents', 'USA-Road', 'wikipedia_link_en']`

## Cache sweep

| index | bytes | size |
|---:|---:|---|
| 0 | 32768 | 32kB |
| 1 | 65536 | 64kB |
| 2 | 131072 | 128kB |
| 3 | 262144 | 256kB |
| 4 | 524288 | 512kB |
| 5 | 1048576 | 1MB |
| 6 | 2097152 | 2MB |
| 7 | 4194304 | 4MB |
| 8 | 8388608 | 8MB |
| 9 | 16777216 | 16MB |
| 10 | 33554432 | 32MB |
| 11 | 67108864 | 64MB |

## Violations

None.
