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
    if mode == ECG_EMBEDDED:
        // Embedded-oracle tiebreak: no dynamic matrix lookup
        max_hint = max(cache_set[c].ecg_srd_hint for c in candidates)
        narrowed = [c in candidates where cache_set[c].ecg_srd_hint == max_hint]

        if |narrowed| == 1:
            return narrowed[0]

        // DBG fallback among same stored-hint lines
        return argmax(cache_set[c].ecg_dbg_tier for c in narrowed)

    if mode == DBG_PRIMARY or mode == DBG_ONLY:
        // DBG tiebreak: evict highest ecg_dbg_tier (coldest / lowest-degree)
        max_dbg = max(cache_set[c].ecg_dbg_tier for c in candidates)
        narrowed = [c in candidates where cache_set[c].ecg_dbg_tier == max_dbg]

        if |narrowed| == 1 or mode == DBG_ONLY:
            return narrowed[0]                // DBG_ONLY stops here (no Level 3)

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
    Prefetch:    Prefetch target hint; 0 = no prefetch

  Auto-allocation: bit widths adjusted based on graph size and available bits
```

Current cache_sim mask details:
- `MaskConfig` packs fields from low to high bits as `PFX | POPT | DBG`.
- Direct prefetch mode is used when the PFX field can hold a raw vertex ID.
- Table prefetch mode is used when PFX is narrower than the vertex ID. Since
    PFX value `0` means no prefetch, table indices are encoded as `index + 1` and
    the hot table capacity is `2^PFX - 1` entries.
- `ECG_PREFETCH_MODE=1` selects the highest-degree neighbor not already present
    in the construction-time dedup window. `ECG_PREFETCH_MODE=2` selects the
    nearest P-OPT neighbor when a rereference matrix is available.
- `ECG_PREFETCH_WINDOW` controls both construction-time target dedup and runtime
    prefetch suppression. It is clamped to the implementation maximum of 16.
- `ECG_PREFETCH_LOOKAHEAD` is a PR cache_sim experiment knob. When set above 0,
    PR prefetches a future incoming-neighbor contribution from the next N edges
    instead of issuing the current vertex's encoded PFX target after the demand
    read. Mode 1 selects the highest-degree future neighbor; mode 2 selects the
    future neighbor with the smallest P-OPT mask hint.
- `src_sim` kernels keep full 32-bit mask entries so automatically allocated
    DBG/POPT/PFX fields are not silently truncated when the spare-bit budget is
    wider than 8 bits.
- Cache_sim logs now print `ECG Mask Build Time` and `ECG Mask Stats` with
    `pfx_encoded`, construction dedup skips, runtime issued prefetches, duplicate
    suppressions, and no-target counts. `roi_matrix.py` also carries these fields
    into aggregate JSON/CSV as `ecg_*` columns.
- Accurate cache_sim JSON/CSV also separates demand misses from prefetch traffic:
    `memory_accesses` is demand-only, `prefetch_fills` is prefetch-caused memory
    traffic, `total_memory_traffic = memory_accesses + prefetch_fills`, and
    `prefetch_useful` counts prefetched lines later hit by demand before they are
    observed as evicted or missed.
- Architectural intent: PFX/lookahead is a pipeline-assisted hardware path. A
    custom `ecg.extract` instruction strips fat-ID metadata into the real vertex
    ID and deposits DBG/POPT/PFX into internal hint state for the next property
    access and graph prefetch unit. The cache_sim lookahead experiments model
    this internal metadata path; they are not claiming that software can scan
    future edge IDs for free. See [plans/ecg-custom-instruction-pipeline-plan.md](../../plans/ecg-custom-instruction-pipeline-plan.md).

### PR PFX Prefetch Sweep Snapshot (2026-05-23)

Focused cache_sim sweep:
- Benchmark: `pr -g 12 -k 16 -o 5 -n 1 -i 3`
- Policies: `ECG_DBG_ONLY`, `ECG_POPT_PRIMARY`
- L3 sizes: 4kB, 8kB, 16kB
- PFX modes: `0` none, `1` highest-degree neighbor, `2` nearest P-OPT neighbor
- Windows: 4, 8, 16

Key observations:
- `ECG_POPT_PRIMARY`, L3=4kB: PFX helps clearly. Best was mode 2/window 16,
    reducing L3 misses from 65,796 to 63,549 (`-2,247`).
- `ECG_POPT_PRIMARY`, L3=8kB/16kB: PFX benefit is small (`-71` and `-25`
    misses at best).
- `ECG_DBG_ONLY`: PFX is mixed. It helps slightly at 4kB and 16kB, but hurts
    at 8kB, suggesting prefetch pollution can dominate when the replacement
    policy is degree-only.
- Larger windows issue more prefetches and suppress fewer duplicates. Window 16
    often wins at high pressure but should not be assumed globally optimal.
- Follow-up accounting smoke on PR g10, L3=4kB, `ECG_POPT_PRIMARY`, mode 2/window
    16: prefetch reduced demand misses 7,421 -> 7,351 and total memory traffic
    7,421 -> 7,362, but 27,319 runtime prefetch requests produced only 11 fills;
    27,308 requests hit lines already resident, and only 4 fills were later useful.
    This confirms that current hot/P-OPT target selection is usually too late or
    redundant on that point, even when the demand-miss metric improves slightly.

### PR Temporal Lookahead PFX Snapshot (2026-05-23)

The first temporal lookahead experiment keeps ECG replacement unchanged but changes
the PR software prefetch issue point. Instead of resolving the current vertex's
PFX target after `contrib[v]` is read, `ECG_PREFETCH_LOOKAHEAD=N` scans upcoming
incoming-neighbor IDs in the same pull row and prefetches a future contribution.

Deterministic cache_sim setup:
- `OMP_NUM_THREADS=1`
- Benchmark: `pr -g 12 -k 16 -o 5 -n 1 -i 2`
- Policy: `ECG_POPT_PRIMARY`
- L3: 4kB
- PFX: mode 2, runtime/build dedup window 16

Results against no PFX (`memory_accesses=44,204`, `total_memory_traffic=44,204`):

| Runtime prefetch path | Demand misses | Total traffic | Useful fills | Interpretation |
|-----------------------|---------------|---------------|--------------|----------------|
| Current PFX target | 42,650 (`-1,554`) | 42,928 (`-1,276`) | 177 / 278 | Some useful bandwidth reduction, modest latency coverage |
| Lookahead 4 | 25,513 (`-18,691`) | 44,211 (`+7`) | 18,698 / 18,698 | Best latency-coverage point in this sweep |
| Lookahead 8 | 29,166 (`-15,038`) | 44,141 (`-63`) | 14,975 / 14,975 | Slightly lower traffic, fewer demand misses hidden |
| Lookahead 16 | 31,316 (`-12,888`) | 44,233 (`+29`) | 12,917 / 12,917 | Strong but worse than 4/8 |
| Lookahead 32 | 32,550 (`-11,654`) | 44,155 (`-49`) | 11,602 / 11,605 | Too far ahead for this point |

Interpretation: the previous PFX path was mostly redundant because it was tied to
the vertex already being processed. Temporal lookahead turns prefetching into a
latency-hiding mechanism: it does not materially reduce bandwidth, but it can
convert a large share of future demand misses into useful prefetch fills.

### Algorithm-Specific Prefetch Strategy Snapshot (2026-05-23)

The same prefetch policy should not be applied blindly across kernels. The useful
prefetch source is the next predictable property-array stream, and that stream is
different for each graph algorithm.

| Kernel phase | Dominant irregular property access | Predictable future stream | Recommended PFX strategy |
|--------------|------------------------------------|---------------------------|--------------------------|
| PR pull | `outgoing_contrib[v]` for incoming neighbors of destination `u` | Remaining IDs in `g.in_neigh(u)` | Temporal row lookahead, P-OPT-ranked, N around 4-8 |
| BFS top-down | `parent[v]` for outgoing neighbors of frontier vertex `u` | Remaining IDs in `g.out_neigh(u)` | Temporal row lookahead, smaller N; prefetch only TD step |
| BFS bottom-up | `front.get_bit(v)` with early exit, plus `parent[u]` stream | Bitmap words and outer vertex stream | Do not use vertex-property PFX yet; needs bitmap-word/outer-stream prefetch |
| SSSP delta relax | `dist[wn.v]` for outgoing weighted neighbors of active vertex `u` | Remaining weighted neighbors in `g.out_neigh(u)` | Temporal row lookahead with strong dedup; N around 4-8 |

Deterministic cache_sim matrix:
- `OMP_NUM_THREADS=1`
- Synthetic graph: `-g 12 -k 16 -o 5 -n 1`
- Policy: `ECG_POPT_PRIMARY`
- L3: 4kB
- PFX mode: 2 (P-OPT-ranked target selection), dedup window 16

| Kernel | Variant | Demand misses | Total traffic | Useful fills | Readout |
|--------|---------|---------------|---------------|--------------|---------|
| PR | none | 44,204 | 44,204 | 0 | Baseline |
| PR | current PFX target | 42,650 (`-1,554`) | 42,928 (`-1,276`) | 177 / 278 | modest bandwidth help |
| PR | lookahead 4 | 25,513 (`-18,691`) | 44,211 (`+7`) | 18,698 / 18,698 | best latency hiding |
| PR | lookahead 8 | 29,166 (`-15,038`) | 44,141 (`-63`) | 14,975 / 14,975 | slightly lower traffic |
| BFS | none | 2,394 | 2,394 | 0 | Baseline TD+BU traversal |
| BFS | lookahead 4 | 1,461 (`-933`) | 2,387 (`-7`) | 926 / 926 | TD neighbor lookahead helps |
| BFS | lookahead 8 | 1,549 (`-845`) | 2,383 (`-11`) | 834 / 834 | fewer requests, slightly less hiding |
| SSSP | none | 16,947 | 16,947 | 0 | Baseline delta stepping |
| SSSP | lookahead 4 | 3,690 (`-13,257`) | 16,836 (`-111`) | 13,146 / 13,146 | strongest latency hiding |
| SSSP | lookahead 8 | 4,679 (`-12,268`) | 16,804 (`-143`) | 12,125 / 12,125 | lower traffic, slightly less hiding |

Interpretation:
- PR and SSSP benefit most because their inner neighbor rows expose a future
    property-array access stream before the demand read.
- BFS benefits only in top-down mode; bottom-up BFS likely needs a separate
    bitmap-word strategy rather than vertex-property PFX.
- Lookahead converts demand misses into useful prefetch fills. It mostly hides
    latency rather than reducing bandwidth, so gem5 validation must model
    prefetch timing and MSHR/bandwidth pressure.
- Current cache_sim lookahead treats future neighbor-ID visibility as a graph
    prefetch-engine capability. If implemented as plain software prefetching,
    we must also charge extra edge-ID lookahead probes or model a companion edge
    stream prefetcher.
- The intended paper architecture is the companion edge-stream decode path: edge
    fat IDs are decoded in the pipeline by `ecg.extract`, then a small internal
    window selects future property prefetch targets. Software exposes the edge
    stream but does not execute extra edge probes for the hardware model.

Next optimization candidates:
- Use cache-line-level runtime dedup for temporal lookahead, not just vertex-ID
    dedup, because property arrays are fetched by line.
- Add a BFS bottom-up bitmap-word prefetch path.
- For SSSP, try frontier/bin lookahead that prefetches `dist[u]` for upcoming
    active vertices before their relax phase.

### Mode Equivalences

| ECG Mode | Equivalent To | Level 2 | Level 3 | Matrix Access |
|----------|--------------|---------|---------|---------------|
| DBG_ONLY | Pure GRASP | DBG tier | *(none — stops here)* | None |
| ECG_EMBEDDED | **Embedded Oracle (NEW)** | Stored SRD hint | DBG tier | **None (zero overhead)** |
| DBG_PRIMARY | Novel hybrid (default) | DBG tier | P-OPT distance | On ties only (~20-30%) |
| POPT_PRIMARY | ~Pure P-OPT | P-OPT distance | DBG tier | Every eviction |
| ECG_COMBINED | Hawkeye-style insert | Combined insertion RRPV | SRRIP aging only | None at eviction |

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
- **Policy Scope**: accurate cache_sim supports `CACHE_L1_POLICY`, `CACHE_L2_POLICY`, and `CACHE_L3_POLICY` overrides so L3-only gem5 policy runs can be mirrored directly
- **MASK Config**: `MaskConfig` struct in `graph_cache_context.h` — bit allocation, decode, RRPV mapping
- **Prefetch Config**: `ECG_PREFETCH_MODE`, `ECG_PREFETCH_WINDOW`, and `ECG_*_BITS` environment overrides control PFX target selection, duplicate suppression, and bit layout
- **Experiment Config**: `scripts/experiments/ecg_config.py`
- **Experiment Runner**: `scripts/experiments/ecg_paper_experiments.py`
- **Cache Sim Binaries**: `bench/src_sim/` — one per benchmark kernel
- **Build**: `make all-sim`
