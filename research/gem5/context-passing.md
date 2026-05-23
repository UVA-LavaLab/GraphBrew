# Context Passing: JSON Sideband → gem5 → SimObject

## Overview

Graph-aware cache policies (GRASP, P-OPT, ECG) and the DROPLET prefetcher
require structural metadata about the graph being processed:
- Property region addresses and degree bucket boundaries
- Degree distribution (for GRASP/ECG tier classification)
- Rereference matrix (for P-OPT/ECG oracle lookups)
- Mask configuration (for ECG fat-ID decoding)

This document describes how this metadata flows from the GraphBrew pipeline
into gem5 SimObjects.

## Design Decision: Hybrid Approach

We use a **hybrid** of static and dynamic context passing:

| Data Type | Passing Method | Why |
|-----------|---------------|-----|
| Degree distribution | JSON sideband → gem5 config | Static per-graph, doesn't change during execution |
| Bucket boundaries | JSON sideband → gem5 config | Same |
| Rereference matrix | Binary file → host-side load | Oracle data, not architecturally visible |
| Property region addrs | JSON sideband → gem5 config | Determined at load time |
| Per-access mask hints | Custom instruction → CSR | Dynamic, changes every edge access |
| Current vertex | Custom instruction → CSR | Dynamic, changes every iteration |

## Data Flow Diagram

```
Phase 1: Pipeline produces metadata
──────────────────────────────────────

graphbrew_experiment.py
    │
    ├── Load graph.sg → Build CSR
    ├── Compute degree distribution → bucket_bounds[]
    ├── Run makeOffsetMatrix() → matrix.bin
    ├── Compute MaskArray → per-edge masks
    │
    ▼
results/gem5_metadata/{graph}/context.json
    │
    ├── property_regions[]: base, upper_bound, num_buckets, bucket_bounds[]
    ├── topology: num_vertices, num_edges, avg_degree, bucket_vertex_counts[]
    ├── mask_config: mask_width, dbg_bits, popt_bits, ecg_mode, rrpv_max
    ├── rereference: matrix_file, num_epochs, num_cache_lines, epoch_size
    └── edge_array: base_address, size, elem_size


Phase 2: gem5 config loads metadata
──────────────────────────────────────

graph_se.py (gem5 Python config)
    │
    ├── graph_metadata_loader.py: load_graph_metadata(context.json)
    │     → Returns dict with all sections
    │
    ├── graph_cache_config.py: make_l3_cache(policy="ECG", ...)
    │     → Creates GraphEcgRP SimObject with params from metadata
    │
    ├── GraphGraspRP.setGraphContext(ctx)  # Sets metadata reference
    │   GraphPoptRP.setGraphContext(ctx)
    │   GraphEcgRP.setGraphContext(ctx)
    │   GraphDropletPrefetcher.setPropertyRegion(base, size, elemSize)
    │   GraphDropletPrefetcher.setEdgeArrayRegion(base, size)
    │
    └── m5.instantiate() → Simulation starts


Phase 3: During simulation
──────────────────────────────────────

CPU executes benchmark binary
    │
    ├── Vertex loop: ecg.extract rd, rs1 → mask written to CSR 0x800
    │                                     → vertex_id written to rd
    │
    ├── Memory access: load property[vertex_id]
    │     │
    │     ▼
    │   Cache Controller checks ECG CSR → mask
    │     │
    │     ▼
    │   GraphEcgRP::reset() → Uses mask for DBG tier → RRPV
    │   GraphEcgRP::getVictim() → Dynamic P-OPT via rereference matrix
    │
    └── DROPLET: Edge access detected → stride predict → indirect property prefetch
```

## JSON Sideband Format

```json
{
  "graph_name": "soc-pokec",
  "graph_path": "results/graphs/soc-pokec/soc-pokec.sg",

  "property_regions": [
    {
      "name": "scores",
      "base_address": 140234567890,
      "upper_bound": 140234574000,
      "num_elements": 1632803,
      "elem_size": 4,
      "num_buckets": 11,
      "bucket_bounds": [140234567900, 140234568000, ...]
    }
  ],

  "topology": {
    "num_vertices": 1632803,
    "num_edges": 30622564,
    "avg_degree": 18.75,
    "num_buckets": 11,
    "bucket_vertex_counts": [1633, 16328, 81640, ...]
  },

  "mask_config": {
    "mask_width": 8,
    "dbg_bits": 2,
    "popt_bits": 4,
    "prefetch_bits": 2,
    "num_buckets": 11,
    "rrpv_max": 7,
    "ecg_mode": "DBG_PRIMARY",
    "enabled": true
  },

  "rereference": {
    "matrix_file": "results/gem5_metadata/soc-pokec/reref_matrix.bin",
    "num_epochs": 256,
    "num_cache_lines": 25513,
    "epoch_size": 6378,
    "sub_epoch_size": 49,
    "base_address": 140234567890,
    "cache_line_size": 64,
    "enabled": true
  },

  "edge_array": {
    "base_address": 140234600000,
    "size": 122490256,
    "elem_size": 4
  }
}
```

## Address Challenges in SE Mode

In gem5 SE (syscall emulation) mode, the process address space is determined by
the ELF loader. Property array addresses are **not known until runtime** because
they depend on `malloc()` allocations.

**Solutions**:

1. **Static addresses** (recommended for initial validation):
   Pre-compute graph data layout and place arrays at known addresses using
   custom memory allocation. Export exact addresses to JSON.

2. **Runtime address registration** (realistic):
   Add m5 pseudo-instructions at the start of the benchmark to register
   property array addresses with the replacement policy SimObject:
   ```cpp
   m5_register_property_region(scores_ptr, num_vertices, sizeof(float));
   ```

3. **Address-range-free classification** (GRASP legacy):
   For DBG-reordered graphs, GRASP's 3-tier classification works purely on
   relative position within the array (hot = front, cold = back). The
   `hot_fraction` parameter determines the boundary without needing exact
   addresses.

## Alternatives Considered

| Approach | Pros | Cons | Decision |
|----------|------|------|----------|
| **JSON sideband** (chosen) | No binary changes, easy to generate | Addresses may differ between runs | Use for static metadata |
| **Magic instructions** | Per-access precision | Requires cross-compiled binary | Use for dynamic hints |
| **Memory-mapped region** | Simple shared memory | Fixed address fragility | Rejected |
| **SimObject parameters** | Clean gem5 API | Can't handle dynamic data | Used for static config |
| **Hardware registers** | Realistic | Complex gem5 modifications | Future work |
