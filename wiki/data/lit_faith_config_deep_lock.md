# Gate 269 — ECG config deep-lock registry

Locks `scripts/experiments/ecg/config.py` against silent drift in the cache-anchor numeric triplet, cache-sweep grid, benchmark partition, policy partition, ECG-mode set, EVAL_GRAPHS schema, reorder-flag vocabulary, and ACCURACY_PAIRS relation tokens. Where gate 256 covers profile NAMES, gate 269 covers profile CONTENT.

registry: 7 cache anchors; 12 cache sweep points; 7 benchmarks; 8 policies; 4 ECG modes; 6 eval graphs; 4 reorder variants; 10 accuracy pairs.

## Rules

- **C1** — DEFAULT_CACHE has canonical L1/L2/L3 size+ways and line-size anchors
- **C2** — CACHE_SIZES_SWEEP is 12-point ascending power-of-2 grid 32KiB..64MiB
- **C3** — BENCHMARKS partition: ITERATIVE∪TRAVERSAL no overlap; PREVIEW⊆BENCHMARKS
- **C4** — POLICIES partition: BASELINE∪GRAPH_AWARE==ALL no overlap; PREVIEW⊆ALL
- **C5** — ECG_MODES is canonical 4-token set (DBG_PRIMARY, POPT_PRIMARY, DBG_ONLY, ECG_EMBEDDED)
- **C6** — EVAL_GRAPHS has required keys + canonical type + positive counts
- **C7** — REORDER_VARIANTS uses recognized reorder flag + non-empty title-case unique labels
- **C8** — ACCURACY_PAIRS uses recognized reorder + policy∈ALL + relation∈canonical

## Allow-lists

- `CONFIG_CACHE_EXTRA_ALLOW` = []
- `CONFIG_RELATION_EXTRA_ALLOW` = []

## ✅ No violations
