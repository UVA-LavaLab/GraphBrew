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
    detects stride pattern at cache-line granularity, predicts next structure
    cache-line addresses.

2. **Indirect property prefetcher**: When a predicted edge-list cache line is
    selected, scans the exported structure-cache-line contents for 4B/8B neighbor
    IDs and issues `property_base + elem_size * neighbor_id` prefetch requests.

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

        // Engine 2: scan predicted structure line for neighbor IDs
        issueIndirectPrefetches(pred, addresses);
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
    use_virtual_addresses=True,
    prefetch_on_access=True,
    on_inst=False,
)

# Regions set from graph metadata:
prefetcher.setPropertyRegion(base=0x1000000, size=N*4, elemSize=4)
prefetcher.setEdgeArrayRegion(base=0x2000000, size=M*4)
```

GraphBrew exports runtime sideband regions from benchmark virtual addresses.
The gem5 `QueuedPrefetcher` base uses physical addresses by default, so DROPLET
must be configured with `use_virtual_addresses=True`. Without that, the
prefetcher loads sideband metadata but never recognizes edge-array accesses, and
all `pf*` counters stay at zero. DROPLET also trains on every data access
(`prefetch_on_access=True`, `on_inst=False`) so the edge stream can build stride
confidence before it becomes a miss-only signal.

For paper-faithful indirect prefetching, the benchmark sideband exports a shadow
copy of the preferred CSR edge array to `/tmp/gem5_graphbrew_{in,out}_edges.bin`.
The first `edge_regions[]` entry is marked `"preferred": true`; PR marks
`in_edges` because pull PageRank traverses incoming rows, while BFS/SSSP/CC-like
kernels default to `out_edges`. DROPLET loads that shadow structure array and
scans real neighbor IDs from predicted structure cache lines. If the element
size is 8B, as in weighted SSSP (`NodeWeight<int,int>`), the property address
generator reads the low 4B vertex ID and skips the weight field, matching the
paper's 4B/8B scan-granularity model.

### Step 5: Attach to L1D or L2 Cache

DROPLET can be attached at L1D or L2 using `graph_se.py --prefetcher-level`.
The current active GraphBrew proof path uses L1D because it directly observes the
core's virtual edge-array stream and matches the original DROPLET L1 stream
prefetcher structure. L2 remains available for follow-up sensitivity runs.

```python
l1d_cache = Cache(size="32kB", assoc=8, ...)
l1d_cache.prefetcher = GraphDropletPrefetcher(
    prefetch_degree=4,
    indirect_degree=8,
    use_virtual_addresses=True,
    prefetch_on_access=True,
    on_inst=False,
)
```

### Step 6: Validation

**Expected results** (from paper):
- 1.37× average speedup across graph workloads
- Up to 1.76× on BFS (highest benefit from direction-optimizing phases)
- 15-45% LLC miss reduction

**Key test**: Compare baseline (no prefetch) vs DROPLET on BFS bottom-up phase
(most irregular access pattern → largest benefit).

GraphBrew activation smoke:
- PR synthetic point: `-g 10 -k 16 -o 5 -n 1 -i 1`, L1D=1kB, L2=2kB, L3=4kB.
- Output: `results/ecg_experiments/roi_matrix/pr_g10_l3_4kb_droplet_fresh_sideband_smoke/`.
- Diagnostics show the benchmark context export before sideband loading, then
    first edge-array access.
- L1D prefetch counters are nonzero (`pfIdentified=1136`, `pfIssued=1136`,
    `pfUseful=111` in ROI section 1), proving the baseline is active.

Paper-faithful actual-edge smoke:
- Output: `results/ecg_experiments/roi_matrix/pr_g10_l3_4kb_droplet_line_actual_edge_smoke/`.
- DROPLET loaded `edge_data=20992`, selected the preferred `in_edges` stream,
    and issued `pfIssued=1166`, `pfUseful=106` in ROI section 1.
- This replaces the earlier edge-offset approximation with actual neighbor-ID
    scanning from the predicted structure line.

Faithfulness checklist:
- Structure accesses are identified by benchmark-exported sideband ranges.
- Structure streaming trains at cache-line granularity.
- Property prefetches are generated from actual neighbor IDs in the predicted
    structure line, not from edge-array offsets.
- The property address equation is `property_base + elem_size * neighbor_id`.
- The model uses benchmark virtual addresses, matching the paper's virtual
    address buffers and specialized-malloc communication path.

Remaining model boundary: gem5 does not model the paper's memory-controller MTLB
or explicit coherence-engine query as separate hardware structures. It also does
not yet schedule property prefetch generation on the exact structure-cache-line
fill event inside the memory controller; instead, it uses the exported shadow
structure line when the structure streamer predicts that line. The
`QueuedPrefetcher` path still accounts for queueing, cache snooping, page checks,
and useful/late/drop stats, but area/timing claims for those DROPLET hardware
blocks must not be inferred from this SimObject.

Clean PR g12 activation proof before actual-edge scanner:
- Output: `results/ecg_experiments/roi_matrix/pr_g12_l3_4kb_gem5_droplet_vaddr_repl_proof/`.
- Benchmark: `pr -g 12 -k 16 -o 5 -n 1 -i 2`, L1D=1kB, L2=2kB, L3=4kB.
- All policy legs loaded the same current sideband ranges:
    property `[0xb54000,0xb58000)`, edge `[0xb6c000,0xbca810)`.
- Average per ROI section: LRU+DROPLET `pfIssued=11411`, `pfUseful=867`;
    POPT+DROPLET `pfIssued=11411`, `pfUseful=861`; ECG_POPT_PRIMARY+DROPLET
    `pfIssued=11411`, `pfUseful=875`.
- POPT+DROPLET was the strongest prior-method row on this activation point;
    ECG_POPT_PRIMARY remained near POPT parity while producing slightly more useful
    prefetch fills.
- This g12 matrix predates the paper-faithful actual-edge-data scanner and
    cache-line-oriented structure streamer. Rerun it before final paper claims.

Post-audit PR g12 actual-edge proof:
- Output: `results/ecg_experiments/roi_matrix/pr_g12_l3_4kb_gem5_droplet_actual_edge_proof/`
- Benchmark: `pr -g 12 -k 16 -o 5 -n 1 -i 2`, L1D=1kB, L2=2kB, L3=4kB.
- Placement: L1D DROPLET.
- All policy legs loaded actual CSR shadow data: `edge_data=96772`.
- Every policy issued `pfIssued=11967` per ROI section.

Average over two ROI sections:

| Policy | Avg ticks | Avg L3 misses | Avg IPC | Avg pfIssued | Avg pfUseful | Avg pfLate | Readout |
|--------|-----------|---------------|---------|--------------|--------------|------------|---------|
| LRU + DROPLET | 9,046,504,500 | 73,547.5 | 0.091179 | 11,967 | 1,038 | 337 | active graph-prefetch baseline |
| GRASP + DROPLET | 9,058,299,500 | 73,331.5 | 0.091060 | 11,967 | 1,041 | 341 | paper-faithful GRASP path active; slightly fewer L3 misses than LRU but slower |
| POPT + DROPLET | 8,604,178,250 | 64,175.5 | 0.095866 | 11,967 | 1,031 | 343 | strongest prior replacement + DROPLET row |
| ECG_POPT_PRIMARY + DROPLET | 8,591,908,500 | 64,313.0 | 0.096004 | 11,967 | 1,027 | 341 | near POPT miss parity and slightly faster ticks on this point |

Readout: the final actual-edge scanner is active and measurable on PR/g12. P-OPT
remains the strongest prior replacement by L3 misses under DROPLET, while
ECG_POPT_PRIMARY is within about `0.21%` L3 misses of POPT and is about `0.14%`
faster in ticks on this stress point. These numbers replace the older
activation-only `pr_g12_l3_4kb_gem5_droplet_vaddr_repl_proof` matrix for DROPLET
baseline discussion, subject to the remaining MC-fill/MTLB hardware boundary
above.

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
