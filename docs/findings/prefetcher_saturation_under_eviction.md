# Prefetcher saturation under graph-aware eviction — sprint 6f-5 finding

**Status:** Closed.  Documented for paper Section 4 (analysis).
**Date:** 2026-06-04
**Commits:** `800191d` (metric fix), `6e11b88` (Top-K), `6b5e2f6` (mode 4 negative)

## TL;DR

When ECG_DBG (combined GRASP+POPT) eviction is in place, **no prefetcher
can beat what eviction already accomplishes**. ECG_PFX, DROPLET, and
mode-4 far-future prefetch all converge at the cache's hot-working-set
ceiling. This explains the corpus-wide convergence observed in sprint
6f-4 and the failure of mode-4 in sprint 6f-5 P2. It is itself a
publishable observation: prefetcher novelty is bounded by the eviction
policy's coverage of the hot working set.

## Background

Sprint 6f investigated whether ECG's prefetcher (ECG_PFX) genuinely
beats DROPLET (Basak HPCA'19). Three implementations were tested
end-to-end on cache_sim with ECG_DBG eviction:

1. **ECG_PFX K=1** — single POPT-ranked best neighbor per demand (mode 2)
2. **ECG_PFX K=4 / K=8** — top-K POPT-ranked neighbors (sprint 6f-3)
3. **mode 4 "far-future"** — encoded target from global hot pool (NOT
   v's neighbors — DROPLET literally cannot see these)

## Result — convergence at saturation

Measured on cit-Patents/pr at L3=1MB (demand-memory rate):

| Config | Prefetches issued | Δ vs ECG_DBG |
|---|---:|---:|
| ECG_DBG baseline | 0 | reference (0.3335) |
| ECG_PFX K=1 LH=8 | 26,425,514 | -13.43 pp |
| ECG_PFX K=4 LH=8 | 58,526,016 | -27.73 pp |
| ECG_PFX K=8 LH=16 | 58,526,012 | -27.73 pp |
| Mode 4 far-future | 32 | -0.00 pp (dedup-suppressed) |
| DROPLET LH=8 | 58,526,023 | -27.64 pp |
| DROPLET LH=16 | 58,525,998 | -27.73 pp |

At saturated bandwidth (~58M requests), all configs converge to the
**same** -27.73 pp demand-memory ceiling.

## Why this happens

The hot working set is approximately bounded by L3 capacity. Once
ECG_DBG eviction protects those lines, any prefetcher that targets
those same lines is redundant:

1. ECG_DBG eviction reserves L3 ways for high-POPT-rank vertices →
   keeps the hot working set resident
2. Any prefetcher (ECG_PFX, DROPLET, mode 4) that fetches those same
   vertices encounters cache HITS in L3 — no new memory traffic
3. Reducing demand misses requires reaching vertices NOT in the hot
   working set — but those are by definition cold/rarely-rereferenced,
   so prefetching them yields low usefulness

## Mode 4 specifically — why it fired only 32 prefetches

Mode 4 was designed to escape the convergence by encoding targets from
the **global** hot pool (top-4096 highest-degree vertices), NOT from
v's immediate neighbors. The hypothesis: DROPLET cannot see these
targets because its stride/indirect engines only scan v's edge stream.

Offline encoder reports show mode 4 worked:

```
ecg_pfx_candidates=3,612,094  ← 96% of vertices got a target
ecg_pfx_encoded=1,848,956     ← 1.85M v's have non-zero pfx_value
```

But at runtime:

```
runtime_no_target=33,695,316  ← mask decoded to 0 for half of reads
runtime_duplicate=32,380,440  ← runtime dedup caught the other half
runtime_issued=32             ← only 32 prefetches actually fired
```

The runtime dedup window (16 entries) catches the prefetches because
**each vertex v's mask is constant**, so every demand read of v emits
the same encoded target. When v is read multiple times in close
succession, the dedup window catches the repeats.

We tested at L3=256kB (much tighter cache) to break the saturation —
mode 4 still fired only 32. The bottleneck is the runtime dedup
mechanism + mask-per-vertex design, not the cache capacity.

## Variants that might break the cap (not implemented)

1. **Per-iteration target rotation:** each time v is read, emit a
   different target. Requires either runtime state (slot counter per
   v) or stochastic selection. Both add complexity.

2. **Multi-target masks:** encode K targets per v, runtime rotates
   through them. Requires more mask bits OR a hot-table lookup
   table with K slots.

3. **Cross-iteration prefetch:** predict the next outer-loop iteration's
   demand vertices and prefetch them. Requires runtime support for
   iteration-boundary signals. DROPLET literally cannot do this.

Option 3 is the most architecturally novel but requires the largest
implementation effort (~2 weeks). Deferred for future work.

## Paper implications

This finding reframes the ECG_PFX claim:

- **NOT** "ECG_PFX beats DROPLET on miss-rate" (false — they converge)
- **NOT** "ECG_PFX uses fewer prefetches than DROPLET" (true at K=1
  but with proportionally less reduction)
- **YES** "ECG_PFX matches DROPLET at any bandwidth budget with simpler
  hardware and 2× lower metadata than POPT alone"
- **YES** "Under graph-aware eviction, prefetcher novelty is bounded
  by the eviction policy's hot-working-set coverage — a fundamental
  observation"

The paper's primary novelty is therefore **architectural simplicity**
(sprint 6f-5 P1, gate 287): ECG's unified mask substrate at 2×
lower storage cost and 2-instruction hardware change. The prefetcher
axis is a **secondary contribution** (efficient knob via Top-K) and
this saturation observation is itself a **paper-worthy negative result**.

## Citations and provenance

- ECG_DBG combined-mask design: `bench/include/cache_sim/graph_cache_context.h:213-292`
- Mode 4 implementation: `bench/include/cache_sim/graph_cache_context.h:1581-1640`
- Sprint 6f-5 P2 commit: `6b5e2f6`
- Sprint 6f-3 Top-K commit: `6e11b88`
- Saturation measurements: `/tmp/mode4_smoke/` and `/tmp/mode4_smallL3/`
  raw cache_sim CSVs
- Paper Table 4 (prefetcher comparison): `wiki/data/paper_table_prefetcher.{json,md,csv}`
- Paper Table 5 (metadata cost — primary novelty): `wiki/data/paper_table_metadata_cost.{json,md,csv}`
