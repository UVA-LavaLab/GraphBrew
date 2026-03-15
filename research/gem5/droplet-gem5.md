# DROPLET → gem5 Integration

## Overview

DROPLET (Data-awaRe decOuPLed prEfeTcher for Graphs) is a hardware prefetcher
designed specifically for graph workloads. It uses separated prefetch engines
for different data types: edge lists (stride-based) and property data (indirect).

**Citation**: Basak et al., "Analysis and Optimization of the Memory Hierarchy
for Graph Processing Workloads", HPCA 2019

## Step-by-Step Integration

### Step 1: Memory Hierarchy Characterization

DROPLET identifies four data types in graph processing with distinct cache behavior:

| Data Type | Reuse Distance | Cache Behavior | Prefetch Strategy |
|-----------|---------------|----------------|-------------------|
| Property (vertex values) | Long, irregular | Thrashes L2, misses L3 | Indirect (from edge data) |
| Edge list (CSR neighbors) | Short, streaming | Fits in L1/L2 | Stride-based |
| Index arrays (CSR offsets) | Very short | Fits in L1 | Not needed |
| Auxiliary data | Algorithm-dependent | Varies | Not targeted |

**Key finding**: L2 cache contributes negligibly for graph workloads (95%+ wasted
on property data that won't reuse within L2 capacity). L3 is where reuse aggregates.

### Step 2: Indirect Prefetch Chain

DROPLET's core innovation is the **indirect prefetching chain**:

```
Step 1: Stride-detect edge list traversal (CSR is contiguous per vertex)
        → Prefetch edge_list[i+k] for k=1..degree

Step 2: When prefetched edge data arrives, extract neighbor vertex IDs
        → neighbor_id = edge_list[i+k]

Step 3: Issue property data prefetch for each extracted neighbor
        → Prefetch property[neighbor_id]

Result: Property data arrives BEFORE the core's load instruction
        → Eliminates the pointer-chasing latency chain
```

### Step 3: gem5 SimObject Implementation

**File**: `bench/include/gem5_sim/overlays/mem/cache/prefetch/droplet.hh/cc`

**Inherits from**: `gem5::prefetch::Queued` (QueuedPrefetcher)

**Two engines**:

1. **Edge-list stride detector**: Tracks last N accesses to the CSR edge array,
   detects stride pattern, predicts next edge addresses.

2. **Indirect property prefetcher**: When an edge-list cache miss occurs, computes
   property addresses for vertices stored at upcoming edge positions and issues
   prefetch requests.

```cpp
void calculatePrefetch(const PrefetchInfo &pfi,
                       std::vector<AddrPriority> &addresses,
                       const CacheAccessor &cache) {
    Addr addr = pfi.getAddr();

    if (isEdgeArrayAccess(addr)) {
        // Engine 1: stride-based edge prefetch
        updateStrideDetector(addr, stridePredictions);
        for (pred : stridePredictions)
            addresses.push_back(AddrPriority(pred, 0));

        // Engine 2: indirect property prefetch (on miss)
        if (pfi.isCacheMiss())
            issueIndirectPrefetches(addr, addresses);
    }
}
```

### Step 4: Region Configuration

DROPLET needs to know the memory layout of graph data structures:

```python
# In gem5 config (graph_se.py):
prefetcher = GraphDropletPrefetcher(
    prefetch_degree=4,     # Edge lines to prefetch ahead
    indirect_degree=8,     # Property prefetches per edge access
    stride_table_size=16,  # Stride detector entries
)

# Regions set from graph metadata:
prefetcher.setPropertyRegion(base=0x1000000, size=N*4, elemSize=4)
prefetcher.setEdgeArrayRegion(base=0x2000000, size=M*4)
```

### Step 5: Attach to L2 Cache

DROPLET is most effective at the L2 level (where edge data fits but property
data misses):

```python
l2cache = Cache(size="256kB", assoc=4, ...)
l2cache.prefetcher = GraphDropletPrefetcher(
    prefetch_degree=4,
    indirect_degree=8,
)
```

### Step 6: Validation

**Expected results** (from paper):
- 1.37× average speedup across graph workloads
- Up to 1.76× on BFS (highest benefit from direction-optimizing phases)
- 15-45% LLC miss reduction

**Key test**: Compare baseline (no prefetch) vs DROPLET on BFS bottom-up phase
(most irregular access pattern → largest benefit).

## Complementarity with ECG

DROPLET and ECG address orthogonal aspects of the memory hierarchy:
- **DROPLET**: Brings data into cache faster (prefetch, reduces miss latency)
- **ECG**: Keeps useful data in cache longer (replacement, reduces miss rate)

Combined system benefits from both:
```
Policy          │ Miss Rate │ Miss Latency │ Combined
───────────────┼───────────┼──────────────┼──────────
LRU (baseline) │ High      │ High         │ Baseline
ECG only       │ Low       │ High         │ Better
DROPLET only   │ High      │ Low          │ Better
ECG + DROPLET  │ Low       │ Low          │ Best
```

### ECG Prefetch Hints

ECG's fat-ID encoding includes 2 prefetch hint bits that approximate DROPLET's
hardware benefit at zero metadata cost. The prefetch bits encode a prefetch
distance (0-3 vertices ahead), letting the cache controller issue software-
guided prefetches without a dedicated hardware prefetcher.

## gem5 Configuration

```python
from m5.objects import GraphDropletPrefetcher, GraphEcgRP

# ECG replacement + DROPLET prefetch (best configuration)
l3_repl = GraphEcgRP(rrpv_max=7, num_buckets=11, ecg_mode="DBG_PRIMARY")
l2_prefetcher = GraphDropletPrefetcher(
    prefetch_degree=4,
    indirect_degree=8,
)

l2cache.prefetcher = l2_prefetcher
l3cache.replacement_policy = l3_repl
```

## Performance Considerations

- DROPLET adds minimal hardware cost: stride table + small prefetch buffer
- Indirect prefetches have lower priority than direct stride prefetches
- Filter prevents redundant prefetches (recently-prefetched address tracking)
- gem5 `QueuedPrefetcher` base class handles MSHR contention and queue management
