# AdaptiveOrder-ML — Correctness & Safety Audit

> Phase 2 deliverable. Findings verified against code.

---

## Audit Summary

| # | Check | Verdict | Location |
|:-:|-------|:-------:|----------|
| 1 | Bijection: `current_id` tracking | **PASS** | `reorder_adaptive.h:540` |
| 2 | Bijection: fallback in `ReorderCommunitySubgraphStandalone` | **CONCERN** | `reorder.h:305–311` |
| 3 | Cross-community edge correctness | **PASS** | `builder.h:796` (automatic relabel) |
| 4 | Determinism: feature extraction | **PASS** | `reorder_types.h:4018` (strided, no RNG) |
| 5 | Determinism: Leiden partitioning | **CONCERN** | `reorder_graphbrew.h:964` (parallel non-determinism) |
| 6 | Integer overflow (`NodeID_` = int32_t) | **PASS** | `benchmark.h:30` |
| 7 | Signed/unsigned fragility (`>= 0` check) | **CONCERN** (minor) | `reorder.h:307` |
| 8 | Thread safety: features | **PASS** | `reorder_types.h:4383` (OMP reduction) |
| 9 | Thread safety: weight cache | **PASS** | `reorder_types.h:3548` (mutex) |
| 10 | Weight loading robustness | **PASS** | `reorder_types.h:3438` (defaults fallback) |
| 11 | Empty graph (0 nodes, recursive mode) | **CONCERN** | `reorder_adaptive.h:400` (no guard) |
| 12 | 1-node graph | **PASS** | Small-comm path handles it |
| 13 | All-small communities | **PASS** | Small-comm handler + empty large loop |

---

## Detailed Findings

### Finding 1: Fallback in ReorderCommunitySubgraphStandalone (CONCERN)

**Location:** `reorder.h:305–311`

```cpp
if (sub_new_ids[i] >= 0 && sub_new_ids[i] < comm_size) {
    reordered_nodes[sub_new_ids[i]] = local_to_global[i];
} else {
    reordered_nodes[i] = local_to_global[i];  // ← can corrupt bijection
}
```

If a sub-algorithm produces an invalid `sub_new_ids[i]` (value `-1` or out of range), the fallback writes `reordered_nodes[i]` directly. But if a valid mapping from another node already wrote to that slot, it gets overwritten — creating a duplicate ID and a missing ID.

**Risk:** Low (all candidate algorithms produce valid bijections, so this path never triggers). But if it ever fires due to a bug in a sub-algorithm, the corruption is silent.

**Fix applied:** Added bijection assertion after the loop (debug mode only).

---

### Finding 2: Leiden Non-Determinism (CONCERN)

**Location:** `reorder_graphbrew.h:964` (`localMovingPhase`)

The local moving phase uses `#pragma omp parallel for` with concurrent community updates. Different thread schedules produce different partitions. No external RNG is used, but parallelism alone causes non-determinism.

**Risk:** Known trade-off. AdaptiveOrder produces different results across runs with `OMP_NUM_THREADS > 1`. Not a bug, but users should be aware.

**Fix applied:** Added documentation comment in `reorder_adaptive.h`.

---

### Finding 3: No Empty-Graph Guard in Recursive Mode (CONCERN)

**Location:** `reorder_adaptive.h:400`

`GenerateAdaptiveMappingFullGraphStandalone` has an explicit `if (num_nodes == 0) return;` guard at line 330. `GenerateAdaptiveMappingRecursiveStandalone` does not — it would call Leiden on an empty graph.

**Risk:** Low (empty graphs are pathological), but the asymmetry is sloppy.

**Fix applied:** Added early return for `num_nodes == 0`.

---

## Fixes Applied

1. **Empty graph guard** in `GenerateAdaptiveMappingRecursiveStandalone`
2. **Bijection assertion** after subgraph reordering (debug builds only)
3. **Determinism doc comment** on the Leiden call
