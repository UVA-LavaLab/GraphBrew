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

**Critical design decision**: The rereference matrix is **host-side oracle data**,
NOT loaded into simulated memory. The `GraphPoptRP` SimObject holds a pointer to
the matrix on the host and queries it during `getVictim()` calls.

This matches the standalone cache_sim approach where `POPTState.reref_matrix`
is a host pointer, and reflects the fact that Bélady's optimal algorithm
requires future knowledge that no real hardware can provide.

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
- **m5 pseudo-instruction** (x86): `m5_popt_set_vertex(v)`
- **Custom instruction** (RISC-V): via CSR write
- **Sideband** (simpler): gem5 config schedules vertex updates

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
