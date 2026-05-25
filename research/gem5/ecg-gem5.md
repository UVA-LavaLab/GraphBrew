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
    DBG_ONLY:     N/A; use the Level 1 SRRIP victim for GRASP parity

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

Current validation note: the x86 SE-mode gem5 runs in `graph_se.py` do not yet
exercise this custom fat-ID instruction/prefetch path. The current x86 gem5
replacement-policy validation uses sideband property regions, degree metadata,
and the P-OPT matrix. Cache_sim is the active implementation for validating the
full MASK/PFX bit layout, temporal lookahead target policy, and prefetch dedup
behavior.

The intended architecture is pipeline internal. Software loads a fat-ID edge
value and executes `ecg.extract`; the instruction returns the real vertex ID and
stores DBG/POPT/PFX metadata in internal hint state. The next property-array
access carries that hint to the cache hierarchy, while a graph prefetch unit can
inspect a small edge-window buffer and issue future property prefetches. This is
the hardware model behind `ECG_PREFETCH_LOOKAHEAD`; it is not a claim that normal
software lookahead probes are free. The detailed contract is in
[plans/ecg-custom-instruction-pipeline-plan.md](../../plans/ecg-custom-instruction-pipeline-plan.md).

Cache_sim PFX semantics to preserve when porting to gem5:
- Fields are packed as `PFX | POPT | DBG` from low to high bits.
- PFX value `0` means no prefetch.
- In table mode, PFX values are 1-based hot-table indices, so a PFX field with
    `b` bits can encode `2^b - 1` hot vertices.
- `ECG_PREFETCH_WINDOW` suppresses repeated hot targets in both mask construction
    and runtime prefetch issue, and is clamped to 16 entries.

### Step 3: gem5 SimObject Implementation

**File**: `bench/include/gem5_sim/overlays/mem/cache/replacement_policies/ecg_rp.hh/cc`

**Per-line metadata** (`EcgReplData`):
```cpp
struct EcgReplData : public ReplacementData {
    uint8_t rrpv;             // Configurable width (3-8 bits)
    uint8_t ecg_dbg_tier;     // Stored structural degree tier
    uint8_t ecg_popt_hint;    // Stored quantized P-OPT hint for ECG_EMBEDDED
    bool is_property_data;
    uint64_t line_addr;       // For P-OPT dynamic lookup
};
```

**Key design decisions**:
- RRPV is set at **insert time** from DBG tier (structural, constant)
- P-OPT is consulted **dynamically at eviction** (avoids stale snapshots)
- ECG_EMBEDDED uses the stored P-OPT hint at eviction, then DBG as fallback
- `ecg_dbg_tier` is stored per-line and never changes after insertion
- gem5 policies re-check property-region membership dynamically after sideband
    load; lines inserted before the benchmark exports sideband must not keep stale
    non-property labels
- Benchmark sideband regions are virtual addresses, so GRASP/P-OPT/ECG classify
    graph property lines with `Request::getVaddr()` when available. Using
    physical addresses silently breaks sideband-region matching in SE mode.
- In gem5's mixed real-memory cache, POPT_PRIMARY uses the same mixed-set path
    as pure P-OPT: full oracle only when all ways are property lines; otherwise
    SRRIP with a P-OPT boost for far-future property lines. This avoids blindly
    evicting instructions, stack, CSR, and other non-property traffic.

### Step 4: Operational Modes

| Mode | L2 Tiebreak | L3 Tiebreak | P-OPT Overhead | Use Case |
|------|-------------|-------------|----------------|----------|
| `DBG_PRIMARY` (default) | Highest DBG tier | Dynamic P-OPT | ~20-30% matrix queries | Best balance |
| `POPT_PRIMARY` | Pure P-OPT 3-phase | Highest DBG tier among RRIP ties | ~100% matrix queries | Oracle validation |
| `DBG_ONLY` | None; plain SRRIP victim | None | Zero | GRASP parity / insertion-hit isolation |
| `ECG_EMBEDDED` | Stored P-OPT hint | Highest DBG tier | Zero matrix queries | Paper headline: embedded oracle |
| `ECG_COMBINED` | Combined insertion RRPV | None | Zero matrix queries | Hawkeye-style insertion experiment |

### Step 5: Configuration

Graph-aware replacement uses the actual simulated LLC capacity for GRASP/ECG
hot-region classification. In the GraphBrew SE config, `--l3-size` is passed to
both the `Cache(size=...)` object and the replacement policy's
`llc_size_bytes`; keep those values in sync when adding new configs.

```python
from m5.objects import GraphEcgRP

# Default: DBG_PRIMARY mode
l3_repl = GraphEcgRP(
    rrpv_max=7,           # 3-bit RRPV
    num_buckets=11,       # 11 degree buckets (matching DBG)
    ecg_mode="DBG_PRIMARY",
    llc_size_bytes=32 * 1024,
)

# For oracle-quality results:
l3_repl = GraphEcgRP(
    rrpv_max=7,
    num_buckets=11,
    ecg_mode="POPT_PRIMARY",
    llc_size_bytes=32 * 1024,
)

# For fastest simulation (no P-OPT overhead):
l3_repl = GraphEcgRP(
    rrpv_max=7,
    num_buckets=11,
    ecg_mode="DBG_ONLY",
    llc_size_bytes=32 * 1024,
)
```

For ROI-scoped timing, build the m5ops-enabled benchmark with
`make gem5-m5ops-pr` and run `bench/bin_gem5/pr_m5ops`. The default
`gem5-pr` binary is built with `-DNO_M5OPS` for native/gem5 compatibility, so
its stats include setup and graph generation.

For cache_sim-to-gem5 comparisons, keep policy scope aligned. gem5 currently
varies the L3 policy while L1/L2 remain LRU, so accurate cache_sim should use
`CACHE_L1_POLICY=LRU CACHE_L2_POLICY=LRU CACHE_L3_POLICY=<policy>` rather than
a single all-level `CACHE_POLICY=<policy>` sweep.

For P-OPT/ECG POPT_PRIMARY, x86 SE-mode gem5 now mirrors cache_sim's explicit
epoch tracking. The graph wrappers call `GEM5_SET_VERTEX(v)` at the same outer
graph-loop points where cache_sim calls `SIM_SET_VERTEX`, and `graph_se.py`
enables that path only for POPT and POPT-using ECG modes through
`GEM5_ENABLE_VERTEX_HINTS=1`. Property-address inference remains only as a
fallback for binaries that do not emit the explicit hint.

For DROPLET comparisons, remember that benchmark sideband regions are virtual
addresses. `GraphDropletPrefetcher` must be created with
`use_virtual_addresses=True`, `prefetch_on_access=True`, and `on_inst=False`; the
ROI runner exposes placement through `--prefetcher-level {l1d,l2}` and parses
prefetch counters from either `system.cpu.dcache.prefetcher.*` or
`system.l2cache.prefetcher.*`.

Clean active-DROPLET PR/g12 activation result before actual-edge scanner:
- Output: `results/ecg_experiments/roi_matrix/pr_g12_l3_4kb_gem5_droplet_vaddr_repl_proof/`
- POPT + DROPLET: avg ticks `8,700,122,000`, avg L3 misses `65,018.5`,
    avg `pfIssued=11,411`, avg `pfUseful=861`.
- ECG_POPT_PRIMARY + DROPLET: avg ticks `8,720,473,000`, avg L3 misses
    `65,112.5`, avg `pfIssued=11,411`, avg `pfUseful=875`.
- Interpretation: ECG_POPT_PRIMARY remained near pure POPT under an active graph
    prefetcher in the activation model. This matrix predates the actual-edge-data
    scanner; rerun it before treating POPT + DROPLET as the final PR/g12
    prior-method baseline.

Post-audit active-DROPLET PR/g12 actual-edge result:
- Output: `results/ecg_experiments/roi_matrix/pr_g12_l3_4kb_gem5_droplet_actual_edge_proof/`
- All policy legs loaded `edge_data=96772` and issued `pfIssued=11,967` per ROI
    section from the actual CSR shadow stream.
- Average over two ROI sections: LRU+DROPLET ticks/L3 misses
    `9,046,504,500` / `73,547.5`; GRASP+DROPLET `9,058,299,500` / `73,331.5`;
    POPT+DROPLET `8,604,178,250` / `64,175.5`; ECG_POPT_PRIMARY+DROPLET
    `8,591,908,500` / `64,313.0`.
- Interpretation: ECG_POPT_PRIMARY remains near pure POPT under the actual-edge
    DROPLET scanner, with about `+0.21%` L3 misses versus POPT and about
    `-0.14%` ticks on this stress point. Use this matrix, not the older
    activation-only run, for PR/g12 DROPLET baseline discussion.

### Step 6: Validation

**Invariants** (from ECG paper Section A3):

1. **ECG(DBG_ONLY) ≈ GRASP**: Within 3% miss rate; same insertion/hit policy and plain SRRIP victim selection
2. **ECG(POPT_PRIMARY) ≈ P-OPT**: Within 5% (same rereference query, DBG as tiebreak)
3. **Ordering guarantee**:
   ```
    P-OPT ≤ ECG(POPT_PRIMARY) ≤ ECG(DBG_PRIMARY) ≤ ECG(DBG_ONLY) = GRASP ≤ SRRIP ≤ LRU
   ```

**2026-05-24 gem5 audit note:** cache_sim already used GRASP-faithful victim
selection for `ECG_DBG_ONLY`, but gem5 still applied a DBG tiebreak among
max-RRPV candidates. A follow-up check also found insertion-side drift: ECG used
line-aligned addresses and a different pre-sideband fallback than gem5 GRASP.
The gem5 overlay and active source copies now use plain SRRIP victim selection,
packet-address insertion classification, and the same `RRPV=2` uncategorized-fill
fallback as GRASP. Post-fix PR/BFS/SSSP g10 spot checks match exactly: zero tick
delta and zero L3-miss delta between `GRASP` and `ECG_DBG_ONLY` across both ROI
sections. Output: `results/ecg_experiments/roi_matrix/gem5_grasp_dbg_parity_g10_post_insertion_fix/`.

**2026-05-24 P-OPT current-vertex update:** x86 SE-mode gem5 now has explicit
`GEM5_SET_VERTEX(v)` hints implemented on top of a GraphBrew-reserved m5ops work
ID. P-OPT/ECG POPT lookups prefer this hint over property-address inference.
PR/g10 and SSSP/g10 POPT-vs-ECG_POPT_PRIMARY spot checks match exactly across
both ROI sections. Outputs:
`results/ecg_experiments/roi_matrix/gem5_popt_current_vertex_hint_pr_g10_smoke/`
and `results/ecg_experiments/roi_matrix/gem5_popt_current_vertex_hint_sssp_g10_smoke/`.

Selected g12 explicit-hint checks are also complete. PR/g12 in
`results/ecg_experiments/roi_matrix/gem5_popt_current_vertex_hint_pr_g12_selected_v2/`
has `ECG_POPT_PRIMARY` slightly ahead of pure POPT: average ticks/L3 misses
`9,313,387,500` / `61,094.0` versus `9,317,427,250` / `61,215.5`. SSSP/g12 in
`results/ecg_experiments/roi_matrix/gem5_popt_current_vertex_hint_sssp_g12_selected_v2/`
matches exactly across both sections, with average ticks/L3 misses
`8,025,875,500` / `48,742.5` for both policies.

### Corrected SSSP Delta-Stepping Validation (2026-05-23)

The SSSP gem5 wrapper now follows the reference GAPBS delta-stepping kernel
from `bench/src/sssp.cc`, matching the cache_sim wrapper's algorithmic path.
The previous Dijkstra-based gem5 SSSP results should be treated as obsolete.

Validation point:
- Benchmark: `sssp -g 12 -k 16 -o 5 -n 1 -r 0 -d 1`
- Cache scope: L1D=1kB LRU, L2=2kB LRU, L3=4kB varied policy
- Results: `results/ecg_experiments/roi_matrix/sssp_ref_delta_g12_l3_4kb_gem5_selected/`

Observed gem5 invariants:
- `ECG_DBG_ONLY` matches `GRASP` exactly for both ROI sections.
- `ECG_POPT_PRIMARY` matches pure `POPT` exactly for both ROI sections.
- `POPT` / `ECG_POPT_PRIMARY` improve over LRU by about 6.0% ticks and
    about 8.8k L3 misses per ROI section.
- `GRASP` / `ECG_DBG_ONLY` beat SRRIP by about 6.2k L3 misses, but are
    about 1.25% slower than LRU on this tiny 4kB gem5 point because SRRIP itself
    is worse than LRU here. This means the paper hierarchy holds for
    `POPT <= GRASP <= SRRIP`, but not for `SRRIP <= LRU` on this stress point.

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

## Pipeline-Internal Prefetch Path

The ECG prefetch path should be modeled as a timing-visible hardware prefetcher,
not as a demand miss counter trick.

1. `ecg.extract` decodes fat-ID metadata and strips the real vertex ID.
2. The LSU attaches the decoded mask to the dependent property access.
3. The cache replacement policy uses DBG/POPT metadata on insertion and victim
    selection.
4. The graph prefetch unit uses PFX plus the algorithm-specific edge-window
    selector to issue a prefetch for a future property address.
5. gem5 must count prefetch fills as memory traffic and expose whether the fill
    later becomes a useful demand hit.

Per-algorithm starting points from cache_sim:
- PR: incoming-row lookahead for `outgoing_contrib[v]`, N=4 or 8.
- BFS top-down: outgoing-row lookahead for `parent[v]`, small N.
- BFS bottom-up: bitmap-word prefetch, not vertex-property PFX.
- SSSP: weighted outgoing-row lookahead for `dist[wn.v]`, N=4 or 8.

Recommended gem5 milestones:
- First, add x86 pseudo-op or sideband hint queue support to validate timing and
  bandwidth effects without switching ISA.
- Then implement RISC-V `ecg.extract` in custom-0 and connect decoded metadata to
  request/hint state.
- Finally, compare demand misses, prefetch fills, useful fills, MSHR pressure,
  memory bandwidth, ticks, and IPC.

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
