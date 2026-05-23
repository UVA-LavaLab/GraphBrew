# ECG — Expressing Locality and Prefetching for Optimal Caching in Graph Structures

## Citation

```bibtex
@inproceedings{ecg-grapl,
  title     = {Expressing Locality and Prefetching for Optimal Caching in Graph Structures},
  author    = {Mughrabi, Abdullah and others},
  booktitle = {Workshop on Graphs, Architectures, Programming, and Learning (GrAPL), co-located with IEEE IPDPS},
  year      = {2026}
}
```

## Why Faithful Implementation Matters

ECG is **our paper**. The entire ECG validation depends on:
1. GRASP and P-OPT being correctly implemented (they are the component policies)
2. The 3-level tiebreaking logic correctly layering SRRIP → structural → oracle signals
3. Mode equivalences (DBG_ONLY ≈ GRASP, POPT_PRIMARY ≈ P-OPT) holding within tolerances
4. Fat-ID MASK encoding correctly packing DBG tier + P-OPT distance + prefetch hints

If the tiebreaking order is wrong, or MASK decoding is incorrect, all Section A3 results and Section B7 mode comparisons are invalid.

## Key Contributions

1. **Hybrid cache replacement**: Combines GRASP (structural) + P-OPT (oracle) via layered tiebreaking
2. **Three operational modes**: DBG_PRIMARY (sweet spot), POPT_PRIMARY (heavy), DBG_ONLY (cheapest)
3. **Fat-ID MASK encoding**: Packs graph metadata into vertex IDs for zero-overhead cache-line annotation
4. **Multi-region awareness**: Different heuristics for vertex data vs edge data vs auxiliary arrays
5. **Integrated prefetching**: Replacement policy + software prefetch guided by graph structure

## Algorithm Pseudo-Code

### ECG 3-Level Tiebreaking Eviction (from GraphBrew cache_sim.h)

```
function findVictimECG(cache_set, mode):
    rrpv_max = 7  // (or from mask_config if enabled)

    // ══ Level 1: SRRIP Aging — find lines at max RRPV ══
    while true:
        if any cache_set[i].rrpv >= rrpv_max: break
        for i in 0..associativity-1:
            if cache_set[i].rrpv < rrpv_max:
                cache_set[i].rrpv += 1        // Age all lines

    // Collect candidates at max RRPV
    candidates = [i for i where cache_set[i].rrpv >= rrpv_max]
    if |candidates| == 1: return candidates[0]

    // ══ Level 2: Mode-dependent tiebreak ══
    if mode == DBG_PRIMARY or mode == DBG_ONLY or mode == ECG_EMBEDDED:
        // DBG tiebreak: evict highest ecg_dbg_tier (coldest / lowest-degree)
        max_dbg = max(cache_set[c].ecg_dbg_tier for c in candidates)
        narrowed = [c in candidates where cache_set[c].ecg_dbg_tier == max_dbg]

        if |narrowed| == 1 or mode == DBG_ONLY:
            return narrowed[0]                // DBG_ONLY stops here (no Level 3)

        // ══ Level 3: Oracle tiebreak ══
        if mode == ECG_EMBEDDED:
            // Use stored rereference hint (no matrix access)
            return argmax(cache_set[c].ecg_srd_hint for c in narrowed)

        // DBG_PRIMARY: dynamic P-OPT via matrix lookup
        if rereference_matrix exists:
            return argmax(findNextRef(cache_set[c].line_addr) for c in narrowed)
        return narrowed[0]

    else:  // POPT_PRIMARY
        // P-OPT tiebreak: evict furthest next-reference
        if rereference_matrix exists:
            max_dist = max(findNextRef(cache_set[c].line_addr) for c in candidates)
            narrowed = [c where findNextRef(addr) == max_dist]

            if |narrowed| == 1: return narrowed[0]

            // ══ Level 3: DBG tiebreak among P-OPT ties ══
            return argmax(cache_set[c].ecg_dbg_tier for c in narrowed)

        // No matrix: fall back to DBG
        return argmax(cache_set[c].ecg_dbg_tier for c in candidates)
```

### ECG Insertion with MASK Encoding

```
function ECG_INSERT(cache_set, victim_idx, address, mask_hint):
    if mask_array enabled:
        dbg_tier = decodeDBG(mask_hint)       // Extract DBG bits from fat-ID
        srd_hint = decodePOPT(mask_hint)      // Extract quantized rereference distance
        rrpv = dbgTierToRRPV(dbg_tier)        // Tier → RRPV mapping
        cache_set[victim_idx].rrpv = rrpv
        cache_set[victim_idx].ecg_dbg_tier = dbg_tier
        cache_set[victim_idx].ecg_srd_hint = srd_hint  // Stored for ECG_EMBEDDED
    else:
        bucket = classifyBucket(address)      // Fallback: address → bucket
        rrpv = bucketToRRPV(bucket)
        cache_set[victim_idx].rrpv = rrpv
        cache_set[victim_idx].ecg_dbg_tier = bucket
```

### Fat-ID MASK Encoding

```
MASK bit layout (8-bit default):
  | DBG tier (2 bits) | P-OPT hint (4 bits) | Prefetch hint (2 bits) |

  DBG tier:    0 = hub (highest-degree), 3 = cold (lowest-degree)
  P-OPT hint:  Quantized rereference distance (0-15)
  Prefetch:    Prefetch distance hint (0-3)

  Auto-allocation: bit widths adjusted based on graph size and available bits
```

### Mode Equivalences

| ECG Mode | Equivalent To | Level 2 | Level 3 | Matrix Access |
|----------|--------------|---------|---------|---------------|
| DBG_ONLY | Pure GRASP | DBG tier | *(none — stops here)* | None |
| ECG_EMBEDDED | **Embedded Oracle (NEW)** | DBG tier | Stored SRD hint | **None (zero overhead)** |
| DBG_PRIMARY | Novel hybrid (default) | DBG tier | P-OPT distance | On ties only (~20-30%) |
| POPT_PRIMARY | ~Pure P-OPT | P-OPT distance | DBG tier | Every eviction |

## GraphBrew Validation Invariants (ECG Section A3)

| # | Invariant | Tolerance | Formula |
|---|-----------|-----------|---------|
| 1 | ECG(DBG_ONLY) ≈ GRASP | < 3% | `|ECG_DBG_ONLY - GRASP| / GRASP < 0.03` |
| 2 | ECG(POPT_PRIMARY) ≈ P-OPT | < 5% | `|ECG_POPT_PRIMARY - POPT| / POPT < 0.05` |
| 3 | DBG_PRIMARY ordering guarantee | structural | `P-OPT ≤ ECG(DBG_PRIMARY) ≤ GRASP` (miss rate) |

### Expected Miss-Rate Hierarchy (lower is better)
```
P-OPT ≤ ECG(DBG_PRIMARY) ≤ ECG(POPT_PRIMARY) ≤ ECG(DBG_ONLY) ≤ GRASP ≤ SRRIP ≤ LRU
```

## ECG Paper Experiments

### Section A — Accuracy Validation

| Exp | What it validates | Source paper |
|-----|-------------------|-------------|
| A1 | GRASP faithfulness (3 claims) | Faldu et al. HPCA 2020 |
| A2 | P-OPT faithfulness (4 claims) | Balaji et al. HPCA 2021 |
| A3 | ECG mode equivalences (3 checks) | This paper |

### Section B — Performance Showcase

| Exp | What it measures |
|-----|-----------------|
| B1 | All 9 policies side by side |
| B2 | Reorder effect: Original vs DBG vs RabbitOrder vs GraphBrewOrder |
| B3 | Full policy × reorder interaction matrix |
| B4 | L3 cache sweep: 32KB → 64MB |
| B5 | Algorithm type: iterative (PR) vs traversal (BFS) |
| B6 | Graph type: social vs road vs citation |
| B7 | ECG mode trade-offs: DBG_PRIMARY / POPT_PRIMARY / DBG_ONLY |
| B8 | Adaptive fat-ID bit allocation |

### Evaluation Configuration

- **Graphs (6)**: pokec, livejournal, orkut (social); patents (citation); road; wikipedia (web)
- **Benchmarks (7)**: pr, pr_spmv, bfs, cc, cc_sv, sssp, bc
- **Cache**: L1=32KB/8-way, L2=256KB/4-way, L3=8MB/16-way, 64B lines
- **Policies (9)**: LRU, FIFO, RANDOM, LFU, SRRIP, GRASP, P-OPT, ECG (3 modes)

## GraphBrew Integration

- **Cache Policy**: `EvictionPolicy::ECG` in `bench/include/cache_sim/cache_sim.h`
- **ECG Modes**: `ECGMode` enum in `bench/include/cache_sim/graph_cache_context.h`
- **MASK Config**: `MaskConfig` struct in `graph_cache_context.h` — bit allocation, decode, RRPV mapping
- **Experiment Config**: `scripts/experiments/ecg_config.py`
- **Experiment Runner**: `scripts/experiments/ecg_paper_experiments.py`
- **Cache Sim Binaries**: `bench/src_sim/` — one per benchmark kernel
- **Build**: `make all-sim`
