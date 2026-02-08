# Repo Map Procedure

You must do this first — before any analysis or code changes.

---

## Step 1: Identify Entry Points

- Where algorithm 14 is invoked (CLI, builder, main)
- Experiment drivers (Python scripts)

## Step 2: Identify Core Modules

- Leiden partitioning integration
- Feature extraction for graphs/communities
- Type centroids / k-means / selection logic
- Perceptron scoring + weight storage/loading
- Reorder implementations in candidate set
- Stitching / composing permutations
- Fallbacks: OOD and margin logic

## Step 3: Output

- A dependency graph (bulleted is fine) listing files + responsibilities
- A "call chain" for AdaptiveOrder-ML from CLI → output mapping

**Deliverable:** `docs/agent_repo_map.md`

---

## Key Files

| File | Purpose |
|------|---------|
| `bench/include/graphbrew/reorder/reorder_adaptive.h` | AdaptiveOrder (algorithm 14) — ML-based selection |
| `bench/include/graphbrew/reorder/reorder_vibe.h` | VIBE reordering (~7100 lines) — all VIBE variants |
| `bench/include/graphbrew/reorder/reorder_leiden.h` | LeidenCSR reordering — enum, parser, dispatch |
| `bench/include/graphbrew/reorder/reorder_graphbrew.h` | GraphBrewOrder (algorithm 12) — composite reordering |
| `bench/include/external/gapbs/builder.h` | Graph builder — dispatches `-o` flag to reorder functions |
| `bench/include/external/rabbit/rabbit_order.hpp` | Boost RabbitOrder reference (764 lines, READ ONLY) |
| `scripts/graphbrew_experiment.py` | **THE** experiment tool — download, reorder, benchmark, weights |
| `scripts/lib/utils.py` | Variant definitions (ALGORITHMS, LEIDEN_CSR_VARIANTS, etc.) |
| `scripts/lib/training.py` | ML weight training |
| `scripts/lib/features.py` | Graph feature computation |
| `scripts/lib/weights.py` | Type-based weight management |
| `scripts/lib/download.py` | Graph catalog (DOWNLOAD_GRAPHS_SMALL/MEDIUM/LARGE/XLARGE) |
| `scripts/weights/active/` | C++ reads weights from here at runtime |

---

## Key Concepts

### Label Maps & Deterministic Reuse
- `--full` auto-enables `--precompute` (= `--generate-maps` + `--use-maps`)
- Phase 1 (reorder) generates `.lo` binary label map files in `results/mappings/{graph}/`
- Phase 2 (benchmark) loads `.lo` files via `-o 13:path.lo` for reproducible orderings
- **Re-running skips existing `.lo` files** — safe to restart after interruption
- Use `--force-reorder` to regenerate, or `--clean-reorder-cache` to delete all maps

### Graph Download & Conversion Pipeline
- `--full` downloads missing graphs automatically (SuiteSparse → `.tar.gz` → `.mtx`)
- The `converter` binary converts `.mtx` → `.sg` (serialized graph, faster I/O)
- Size categories control download scope:
  - `--size small` → 16 graphs, ~62 MB
  - `--size medium` → 28 graphs, ~1.1 GB
  - `--size large` → 37 graphs, ~25 GB
  - `--size xlarge` → 6 graphs, ~63 GB
- `--auto` uses 60% of RAM and 80% of disk as safe limits, skips oversized graphs
- After first download, graphs are cached in `results/graphs/` forever

### Variant System
- `--csr-variants X Y Z` selects specific LeidenCSR/VIBE variants (auto-enables expansion)
- `--rabbit-variants boost` adds Boost RabbitOrder as baseline
- **Don't use `--all-variants`** unless you want ALL 34 LeidenCSR variants (very slow)
- The flag name is `--all-variants` (NOT `--expand-variants` — that's the internal dest)

### Result Files
```
results/
├── benchmark_{timestamp}.json    # Array of {graph, algorithm, benchmark, time_seconds, ...}
├── reorder_{timestamp}.json      # Array of {graph, algorithm_name, time_seconds, mapping_file, ...}
├── logs/{graph}/
│   ├── benchmark_{algo}_{bench}.log   # Raw stdout from each benchmark run
│   └── reorder_{algo}.log             # Raw stdout from each reorder run
└── mappings/{graph}/
    ├── LeidenCSR_vibe:rabbit.lo       # Binary label map (reusable)
    ├── LeidenCSR_vibe:rabbit.time     # Reorder timing
    ├── RABBITORDER_boost.lo
    └── RABBITORDER_boost.time
```

---

## VIBE Variants Reference

| Variant | Option String | Algorithm ID | Description |
|---------|---------------|:---:|-------------|
| Boost RabbitOrder | `-o 8:boost` | 8 | Reference baseline (Boost library) |
| VIBE RabbitOrder | `-o "17:vibe:rabbit"` | 17 | Pure RabbitOrder, VIBE implementation |
| VIBE Hybrid BFS | `-o "17:vibe:hrab"` | 17 | Leiden → RabbitOrder inter → BFS intra |
| VIBE Hybrid Gorder | `-o "17:vibe:hrab:gordi"` | 17 | Leiden → RabbitOrder inter → Gorder intra |
| VIBE Connectivity | `-o "17:vibe:conn"` | 17 | Connectivity BFS ordering (Boost-style) |
