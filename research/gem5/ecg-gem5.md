# ECG → gem5 Integration

## Overview

ECG (Expressing Locality and Prefetching for Optimal Caching in Graph Structures)
is the hybrid policy that combines GRASP's degree-aware classification with
P-OPT's rereference prediction through a 3-level layered eviction strategy.

**Citation**: Mughrabi et al., "Expressing Locality and Prefetching for Optimal
Caching in Graph Structures", GrAPL @ IPDPS 2026

## Step-by-Step Integration

### Step 1: Understand ECG 3-Level Eviction

```
Level 1 (all modes): SRRIP Aging
    Age all RRPVs until ≥1 line reaches rrpv_max
    Collect candidates at max RRPV

Level 2 (mode-dependent):
    DBG_PRIMARY:  Evict highest DBG tier (coldest vertex)
    POPT_PRIMARY: Evict furthest P-OPT rereference distance
    DBG_ONLY:     Evict highest DBG tier (stop here, no L3)

Level 3 (tiebreaker):
    DBG_PRIMARY:  Break L2 ties with dynamic P-OPT distance
    POPT_PRIMARY: Break L2 ties with DBG tier
    DBG_ONLY:     N/A (L2 is final)
```

### Step 2: Per-Access Mask Hints

ECG's key innovation is the **fat-ID mask** encoding. Each CSR neighbor ID embeds
cache hints in its upper bits:

```
64-bit neighbor value:
┌──────┬──────┬──────┬──────────────────────┐
│ DBG  │ POPT │ PFX  │    Real Vertex ID    │
│(2bit)│(4bit)│(2bit)│     (56 bits)        │
└──────┴──────┴──────┴──────────────────────┘
```

In gem5, these hints are delivered via the **custom ECG instruction**:
1. Load fat-ID from CSR edge array
2. Execute `ecg.extract rd, rs1` → strips mask from vertex ID
3. Mask written to CSR 0x800 → cache controller reads at next access
4. Cache uses mask for `reset()` (insertion RRPV from DBG tier)

### Step 3: gem5 SimObject Implementation

**File**: `bench/include/gem5_sim/overlays/mem/cache/replacement_policies/ecg_rp.hh/cc`

**Per-line metadata** (`EcgReplData`):
```cpp
struct EcgReplData : public ReplacementData {
    uint8_t rrpv;             // Configurable width (3-8 bits)
    uint8_t ecg_dbg_tier;     // Stored structural degree tier
    bool is_property_data;
    uint64_t line_addr;       // For P-OPT dynamic lookup
};
```

**Key design decisions**:
- RRPV is set at **insert time** from DBG tier (structural, constant)
- P-OPT is consulted **dynamically at eviction** (avoids stale snapshots)
- `ecg_dbg_tier` is stored per-line and never changes after insertion

### Step 4: Three Operational Modes

| Mode | L2 Tiebreak | L3 Tiebreak | P-OPT Overhead | Use Case |
|------|-------------|-------------|----------------|----------|
| `DBG_PRIMARY` (default) | Highest DBG tier | Dynamic P-OPT | ~20-30% matrix queries | Best balance |
| `POPT_PRIMARY` | Furthest P-OPT | Highest DBG tier | ~100% matrix queries | Oracle validation |
| `DBG_ONLY` | Highest DBG tier | None (fast path) | Zero | Cheapest, ≈ GRASP |

### Step 5: Configuration

```python
from m5.objects import GraphEcgRP

# Default: DBG_PRIMARY mode
l3_repl = GraphEcgRP(
    rrpv_max=7,           # 3-bit RRPV
    num_buckets=11,       # 11 degree buckets (matching DBG)
    ecg_mode="DBG_PRIMARY",
)

# For oracle-quality results:
l3_repl = GraphEcgRP(
    rrpv_max=7,
    num_buckets=11,
    ecg_mode="POPT_PRIMARY",
)

# For fastest simulation (no P-OPT overhead):
l3_repl = GraphEcgRP(
    rrpv_max=7,
    num_buckets=11,
    ecg_mode="DBG_ONLY",
)
```

### Step 6: Validation

**Invariants** (from ECG paper Section A3):

1. **ECG(DBG_ONLY) ≈ GRASP**: Within 3% miss rate (same algorithm, just mask-decoded)
2. **ECG(POPT_PRIMARY) ≈ P-OPT**: Within 5% (same rereference query, DBG as tiebreak)
3. **Ordering guarantee**:
   ```
   P-OPT ≤ ECG(POPT_PRIMARY) ≤ ECG(DBG_PRIMARY) ≤ ECG(DBG_ONLY) ≤ GRASP ≤ SRRIP ≤ LRU
   ```

## Data Flow

```
              ┌─────────────┐     ┌──────────────────────┐
              │ Fat-ID CSR  │     │ JSON Sideband        │
              │ edge array  │     │ (context.json)       │
              └──────┬──────┘     └──────────┬───────────┘
                     │                       │
        ┌────────────▼──────────┐  ┌─────────▼────────────┐
        │ ecg.extract rd, rs1   │  │ graph_metadata_      │
        │ (custom RISC-V inst)  │  │ loader.py            │
        │                       │  │                      │
        │ Writes mask → CSR     │  │ Loads: PropertyRegion,│
        │ Writes vertex → rd    │  │  RereferenceMatrix,  │
        └────────────┬──────────┘  │  MaskConfig, Topology│
                     │             └──────────┬───────────┘
                     │                        │
                     ▼                        ▼
              ┌───────────────────────────────────────┐
              │            GraphEcgRP                 │
              │                                       │
              │  reset(): mask → DBG tier → RRPV      │
              │  touch(): bucket-aware promotion      │
              │  getVictim(): 3-level eviction         │
              │    L1: SRRIP aging                    │
              │    L2: mode-dependent                 │
              │    L3: tiebreaker                     │
              └───────────────────────────────────────┘
```

## Reference Implementation Mapping

| Standalone (cache_sim.h) | gem5 (ecg_rp.cc) |
|--------------------------|---------------------|
| `ECGMode` enum | `graph::ECGMode` enum |
| `MaskConfig::decodeDBG()` | `MaskConfig::decodeDBG()` (same) |
| `MaskConfig::dbgTierToRRPV()` | `MaskConfig::dbgTierToRRPV()` (same) |
| `findVictimECG()` (lines 963-1052) | `GraphEcgRP::getVictim()` |
| L1 aging loop | Same `while(true)` RRPV increment |
| L2 DBG tiebreak | `maxDBG` among candidates |
| L3 P-OPT tiebreak | `findNextRef()` among DBG-tied lines |
| POPT_PRIMARY path | Swapped L2/L3 order |
| `graph_ctx_->current_mask` | `graphCtx->current_mask` (from CSR) |
