# P-OPT → gem5 Integration

## Overview

P-OPT (Practical Optimal cache replacement for Graph Analytics) uses the graph's
transpose to pre-compute a rereference matrix that predicts when each cache line
will next be accessed. It serves as the oracle baseline for graph cache policies.

**Citation**: Balaji et al., "P-OPT: Practical Optimal Cache Management for
Graph Analytics", HPCA 2021

## Step-by-Step Integration

### Step 1: Understand P-OPT Algorithm

P-OPT is a 3-phase eviction policy:

```
Phase 1: Evict non-graph data first (CSR metadata, streaming buffers)
Phase 2: Among graph vertex data → evict line with max rereference distance
Phase 3: RRIP tiebreaker among lines with equal rereference distance
```

The rereference matrix is oracle data:
- Pre-computed from the graph transpose via `makeOffsetMatrix()` (popt.h)
- Compressed 8-bit entries: `[num_epochs × num_cache_lines]`
- Entry encoding:
  - MSB=1: line IS referenced in epoch; bits[6:0] = sub-epoch of last access
  - MSB=0: line NOT referenced; bits[6:0] = distance in epochs to next reference

### Step 2: Rereference Matrix in gem5

**Critical design decision**: the base `POPT` policy in GraphBrew is an
**oracle P-OPT** path. The rereference matrix is **host-side oracle data**, NOT
loaded into simulated memory. The `GraphPoptRP` SimObject holds a pointer to the
matrix on the host and queries it during `getVictim()` calls.

This matches the standalone cache_sim approach where `POPTState.reref_matrix`
is a host pointer, and reflects the fact that Bélady's optimal algorithm
requires future knowledge that no real hardware can provide.

For paper-fair overhead comparisons, use `POPT_CHARGED` in the experiment
runners. It maps to the same dynamic P-OPT replacement policy but charges the
paper's reserved-rereference-column overhead by reducing effective L3 data ways.
The model reserves enough LLC ways to hold the current and next rereference
matrix columns (`2 * num_cache_lines` bytes by default), keeps the same number
of cache sets, and records estimated matrix-stream traffic (`num_epochs` matrix
columns) in the output CSV. gem5 timing currently models the capacity charge;
the matrix-stream traffic is reported as bandwidth metadata rather than injected
as additional simulated memory requests.

```
Pipeline (scripts/)                    gem5 (C++)
─────────────────                    ──────────────
makeOffsetMatrix() ──►  matrix.bin ──► POPTState loads matrix
(popt.h)                              via loadFromFile()
                                       │
                                       ▼
                                  findNextRef(cline_id, current_vertex)
                                  → returns distance 0..127
```

Experiment labels:
- `POPT`: uncharged oracle P-OPT ceiling.
- `POPT_CHARGED`: paper-overhead P-OPT approximation with reserved LLC ways.
- `ECG_*_CHARGED`: dynamic P-OPT ECG mode with the same reserved-way charge.

Charged smoke validation:
- Output: `results/ecg_experiments/roi_matrix/popt_charged_gem5_smoke/`
- Benchmark: `pr -g 10 -k 16 -o 5 -n 1 -i 1`, L3=4kB.
- `POPT_CHARGED` reserved one of 16 LLC ways for current+next matrix columns,
  running with effective L3 `3840B` / 15 ways and reporting 256 matrix-stream
  cache lines.
- Versus uncharged `POPT`, charged P-OPT increased ticks by `13,676,500` /
  `13,995,500` and L3 misses by `103` / `114` across the two ROI sections.
- `ECG_DBG_PRIMARY_CHARGED` used the same charge and increased ticks by
  `23,604,000` / `25,069,500` and L3 misses by `338` / `370` versus uncharged
  `ECG_DBG_PRIMARY`. All four gem5 legs exited with code 0.

### Step 3: gem5 SimObject Implementation

**File**: `bench/include/gem5_sim/overlays/mem/cache/replacement_policies/popt_rp.hh/cc`

**Per-line metadata** (`PoptReplData`):
```cpp
struct PoptReplData : public ReplacementData {
    uint8_t rrpv;            // For tiebreaking (insert at M-1, hit → 0)
    bool is_property_data;   // Whether line contains vertex data
    uint64_t line_addr;      // Cache-line-aligned address (for matrix lookup)
};
```

**Key methods**:

1. **`reset()` (insertion)**: RRPV = M-1 (6); classify as property/non-property
2. **`touch()` (hit)**: RRPV = 0
3. **`getVictim()` (eviction)**: 3-phase algorithm matching standalone

### Step 4: Epoch Tracking

P-OPT partitions the algorithm's execution into epochs. The current epoch is
determined by the current destination vertex being processed:

```
epoch_id = current_vertex / epoch_size
sub_epoch = (current_vertex % epoch_size) / sub_epoch_size
```

The `current_vertex` is updated via:
- **m5 pseudo-instruction** (x86 SE): `GEM5_SET_VERTEX(v)`, implemented as a
  GraphBrew-reserved `m5_work_begin` work ID carrying `v` in the second operand.
- **Custom instruction** (RISC-V): future CSR write / ECG hint latch.
- **Address inference fallback**: property-array accesses update the vertex only
  when no explicit hint has been seen.

Current GraphBrew status: cache_sim uses explicit `SIM_SET_VERTEX(cache, u)` at
the top of the outer graph loop and the x86 SE-mode gem5 wrappers now mirror that
with `GEM5_SET_VERTEX(u)` in graph traversal loops. gem5's pseudo-instruction handler
intercepts the reserved GraphBrew work ID before normal work-item accounting and
updates a host-side graph current-vertex hint. `GraphCacheContext::findNextRef()`
uses that explicit hint for P-OPT/ECG POPT lookups, falling back to
`updateVertexFromAddr()` only when the hint path has not been used.
`graph_se.py` enables the macro with `GEM5_ENABLE_VERTEX_HINTS=1` only for POPT
and POPT-using ECG modes so LRU/GRASP/ECG_DBG_ONLY runs do not pay the m5ops
overhead.

Implementation options reviewed on 2026-05-23:

| Option | Status | Reason |
|--------|--------|--------|
| Infer from property addresses | Fallback only | Can identify the accessed property vertex, but PR/SSSP often access neighbor/source properties while P-OPT needs the outer loop vertex. |
| Infer from edge/CSR addresses | Insufficient for final claims | L3 replacement policy only observes packets that reach L3; upper-level hits can skip the update, so epoch state can lag. |
| Memory-mapped hint array | Rejected for final evidence | A volatile access can encode the vertex in the address, but it adds simulated memory traffic and cache state. |
| m5ops/pseudo-op `GEM5_SET_VERTEX(v)` | Implemented x86 SE path | Carries the dynamic update without data-cache pollution and matches the paper's explicit `update_index` register semantics. |
| ECG custom instruction / CSR | Preferred ISA path | Same semantic endpoint as m5ops, but suitable for the final ECG ISA story. |

Initial validation after implementing the explicit hint:
- Output: `results/ecg_experiments/roi_matrix/gem5_popt_current_vertex_hint_pr_g10_smoke/`
- Benchmark: `pr -g 10 -k 16 -o 5 -n 1 -i 2`, L3=4kB.
- `POPT` and `ECG_POPT_PRIMARY` matched exactly across both ROI sections:
  zero tick delta and zero L3-miss delta.
- Output: `results/ecg_experiments/roi_matrix/gem5_popt_current_vertex_hint_sssp_g10_smoke/`
- Benchmark: `sssp -g 10 -k 16 -o 5 -n 1 -r 0 -d 1`, L3=4kB.
- `POPT` and `ECG_POPT_PRIMARY` also matched exactly across both ROI sections:
  zero tick delta and zero L3-miss delta. This exercises the helper-level
  `GEM5_SET_VERTEX(u)` call in the SSSP relaxation loop.

Selected g12 validation after the same explicit-hint fix:
- Output: `results/ecg_experiments/roi_matrix/gem5_popt_current_vertex_hint_pr_g12_selected_v2/`
- Benchmark: `pr -g 12 -k 16 -o 5 -n 1 -i 2`, L3=4kB.
- `ECG_POPT_PRIMARY` is slightly better than pure `POPT` on this point:
  section deltas are `-4,634,000` / `-3,445,500` ticks and `-125` / `-118`
  L3 misses. Average ticks/L3 misses are `9,317,427,250` / `61,215.5` for
  `POPT` and `9,313,387,500` / `61,094.0` for `ECG_POPT_PRIMARY`.
- Output: `results/ecg_experiments/roi_matrix/gem5_popt_current_vertex_hint_sssp_g12_selected_v2/`
- Benchmark: `sssp -g 12 -k 16 -o 5 -n 1 -r 0 -d 1`, L3=4kB.
- `POPT` and `ECG_POPT_PRIMARY` match exactly across both ROI sections:
  zero tick delta and zero L3-miss delta. Average ticks/L3 misses for both are
  `8,025,875,500` / `48,742.5`.

### Step 5: Configuration

```python
from m5.objects import GraphPoptRP

l3_repl = GraphPoptRP(max_rrpv=7)
# Rereference matrix loaded via graph_metadata_loader.py
```

### Step 6: Validation

**Invariants** (from ECG paper Section A2):

1. **P-OPT ≈ Bélady optimal**: Within 1-5% miss rate of true optimal
2. **Reorder-agnostic**: Within 10% across different vertex orderings
3. **Beats all RRIP variants**: Strictly better than SRRIP, GRASP
4. **>10% improvement vs LRU**: On graphs with significant irregular access

**Expected miss-rate hierarchy**:
```
Bélady ≤ P-OPT ≤ ECG(POPT_PRIMARY) ≤ ECG(DBG_PRIMARY) ≤ GRASP ≤ SRRIP ≤ LRU
```

**Current corrected SSSP observation (2026-05-24)**: On the reference
delta-stepping SSSP gem5 point `-g 12 -k 16 -o 5 -n 1 -r 0 -d 1`, L3=4kB,
pure P-OPT and `ECG_POPT_PRIMARY` match exactly with the explicit
`GEM5_SET_VERTEX` current-vertex path. This validates the P-OPT/ECG oracle-mode
parity on the corrected same-algorithm path, though the point remains a
tiny-cache stress case.

## Rereference Matrix Binary Format

```
Header (16 bytes):
  [4B] num_epochs      (uint32_t, typically 256)
  [4B] num_cache_lines (uint32_t, varies with graph size)
  [4B] epoch_size      (uint32_t, vertices per epoch)
  [4B] sub_epoch_size  (uint32_t, epoch_size / 128)

Data (num_epochs × num_cache_lines bytes):
  [1B] entry for each (epoch, cache_line) pair
  MSB=1: referenced in epoch, bits[6:0] = sub-epoch of last access
  MSB=0: not referenced, bits[6:0] = epochs until next reference
```

## Performance Considerations

- Matrix size for large graphs can be significant:
  - 1M vertices, 64B lines → ~250K cache lines → 256 × 250K = 64MB
  - 10M vertices → 640MB
  - 100M vertices → 6.4GB (may require mmap)
- In gem5, matrix lookups add host-side overhead but don't affect simulated cycles
- Consider lazy loading / mmap for very large matrices

## Reference Implementation Mapping

| Standalone (cache_sim.h) | gem5 (popt_rp.cc) |
|--------------------------|---------------------|
| `POPTState` struct (lines 146-212) | `GraphPoptRP` class |
| `POPTState::findNextRef()` | `RereferenceMatrix::findNextRef()` |
| `findVictimPOPT()` (lines 890-950) | `GraphPoptRP::getVictim()` |
| Phase 1: non-graph eviction | Same logic (`!is_property_data → evict`) |
| Phase 2: max rereference distance | `findNextRef(line_addr)` on all candidates |
| Phase 3: RRIP tiebreak | Age only tied lines until max RRPV |
