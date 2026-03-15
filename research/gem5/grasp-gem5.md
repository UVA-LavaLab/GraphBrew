# GRASP → gem5 Integration

## Overview

GRASP (Graph-aware cache Replacement with Software Prefetching) extends SRRIP
with degree-based 3-tier insertion and promotion. It is the foundation for ECG's
DBG-aware modes.

**Citation**: Faldu et al., "Domain-Specialized Cache Management for Graph
Analytics", HPCA 2020

## Step-by-Step Integration

### Step 1: Understand GRASP Algorithm

GRASP classifies cache lines based on the degree of the vertex whose data they contain:

| Reuse Tier | Degree | Insert RRPV | Hit Behavior |
|------------|--------|-------------|--------------|
| HIGH (hot hubs) | Top 10% by edges | 1 (P_RRIP) | RRPV → 0 |
| MODERATE | Next ~20% | M-1 (6) | Decrement by 1 |
| LOW (cold) | Remaining | M (7) | Decrement by 1 |

Eviction is identical to SRRIP: scan for max RRPV, age all if none found.

### Step 2: gem5 SimObject Implementation

**File**: `bench/include/gem5_sim/overlays/mem/cache/replacement_policies/grasp_rp.hh/cc`

**Per-line metadata** (`GraspReplData`):
```cpp
struct GraspReplData : public ReplacementData {
    uint8_t rrpv;            // Re-Reference Prediction Value (0-7)
    uint8_t degree_bucket;   // Degree bucket (0=hub, N-1=cold)
    bool is_property_data;   // Whether line contains vertex data
};
```

**Key methods**:

1. **`reset()` (insertion)**: Classify address → property region → degree bucket → 3-tier RRPV
2. **`touch()` (hit)**: Bucket-0 hubs → RRPV=0; others → decrement
3. **`getVictim()` (eviction)**: Same as SRRIP (scan for max RRPV, age)

### Step 3: Address Classification

GRASP needs to know which addresses belong to vertex property data and which
degree bucket they fall in. Two approaches:

**Approach A: GraphCacheContext (recommended)**
```python
# In gem5 config:
grasp = GraphGraspRP(max_rrpv=7, num_buckets=11, hot_fraction=0.1)
# GraphCacheContext loaded from JSON sideband provides bucket classification
```

**Approach B: Legacy address-range (for DBG-reordered graphs)**
```
data_base              high_reuse_bound      moderate_reuse_bound     data_end
|---- HIGH (hubs) -----|---- MODERATE -------|---- LOW (cold) ---------|
```
After DBG reordering, high-degree vertices are at low addresses, so simple
address comparison suffices. This is the original GRASP paper's approach.

### Step 4: Configuration

```python
from m5.objects import GraphGraspRP

l3_repl = GraphGraspRP(
    max_rrpv=7,          # 3-bit RRPV (0-7)
    num_buckets=11,      # Match DBG 11 degree buckets
    hot_fraction=0.1,    # 10% LLC for hubs (paper default)
)
```

### Step 5: Validation

**Invariants** (from ECG paper Section A1):

1. **GRASP + DBG ≤ SRRIP**: With DBG-reordered graph, GRASP miss rate must be
   strictly better than SRRIP.
2. **GRASP − DBG ≈ SRRIP**: Without DBG (ORIGINAL ordering), GRASP ≈ SRRIP
   (within 5%), because all addresses map to LOW tier.
3. **Structural**: Bucket-0 (highest-degree) vertices get RRPV → 0 on hit;
   bucket-10 (lowest-degree) get RRPV → max on insert.

**Paper result**: 33-62% miss reduction vs LRU on web-Google with DBG reordering.

## Data Flow

```
                JSON Sideband (context.json)
                         │
                         ▼
              ┌─────────────────────┐
              │ graph_metadata_     │
              │ loader.py           │
              │                     │
              │ Loads: PropertyRegion│
              │   bucket_bounds[]   │
              │   num_buckets=11    │
              │   hot_fraction=0.1  │
              └──────────┬──────────┘
                         │
                         ▼
              ┌─────────────────────┐
              │ GraphGraspRP        │
              │                     │
              │ reset(): addr →     │
              │   classifyBucket()  │
              │   → 3-tier RRPV     │
              │                     │
              │ touch(): bucket →   │
              │   hub=0, others -1  │
              │                     │
              │ getVictim(): SRRIP  │
              │   scan max RRPV     │
              └─────────────────────┘
```

## Reference Implementation Mapping

| Standalone (cache_sim.h) | gem5 (grasp_rp.cc) |
|--------------------------|---------------------|
| `GRASPState::init()` | `GraphGraspRP` constructor + `setGraphContext()` |
| `GRASPState::classify()` | `classifyAddress()` → `ReuseTier` |
| Lines 755-780 (hit promotion) | `promoteOnHit()` |
| `findVictimGRASP()` line ~870 | `getVictim()` |
| 3-tier RRPV: P=1, I=M-1, M=M | `insertionRRPV(tier)` |
