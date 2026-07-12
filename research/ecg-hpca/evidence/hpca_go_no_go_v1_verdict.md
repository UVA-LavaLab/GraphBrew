# HPCA Mode 6 — Go/No-Go v1 Verdict (HONEST FRAMING REVISION)

**Date:** 2026-06-07
**Run dir:** `results/ecg_experiments/hpca_mode6/go_no_go_v1/`
**Manifest:** `scripts/experiments/ecg/hpca_mode6_manifest.json` (profile `go_no_go`)
**Wall:** ~2h, 10 stages all `status=ok`
**Revision:** v2 (rubber-duck pass identified that "99% demand miss
reduction" was misleading — mode 6 path skips `SIM_CACHE_READ_EDGE`
that baseline counts, so mem_acc deltas mix CSR-elimination with
prefetch quality. Headline metric is now `total_memory_traffic`.)

## Verdict: 🟢 GO

Both kill switches passed. Phase 2 (full baseline sweep) is now running
(`results/ecg_experiments/hpca_mode6/baselines_v1/`).

## KILL-1: PASS on all 4 cells (honest framing)

`popt_off__isa__k2` beats `seq__sw__k8` AND `seq__sw__k16` on
**`total_memory_traffic`** (= memory_accesses + prefetch_fills, in cache
lines) on EVERY tested (graph, L3) cell.

### Headline metric — TOTAL DRAM TRAFFIC (cache lines)

| graph | L3 | baseline | DROPLET k8 | DROPLET k16 | **mode 6 isa k2** | mode 6 / baseline | mode 6 / DROPLET k8 |
|---|---|---:|---:|---:|---:|---:|---:|
| com-orkut | 2MB | 173.8M | 173.4M | 172.8M | **139.5M** | **0.80×** (-20%) | **0.80×** (-20%) |
| com-orkut | 8MB | 46.8M | 46.8M | 46.8M | **14.1M** | **0.30×** (-70%) | **0.30×** (-70%) |
| soc-LJ | 2MB | 56.2M | 56.2M | 56.3M | **44.5M** | **0.79×** (-21%) | **0.79×** (-21%) |
| soc-LJ | 8MB | 21.5M | 21.5M | 21.5M | **10.1M** | **0.47×** (-53%) | **0.47×** (-53%) |

### What this 20-70% DRAM reduction is composed of

The reduction has 3 contributing factors (per rubber-duck audit):

1. **Prefetch effectiveness** (real prefetch quality)
2. **CSR-elimination** (mode 6 decodes destination from mask, skips CSR load)
3. **Working-set effects** (POPT-rank picks hubs that survive cache pressure)

To isolate (1) from (2)+(3), Phase 3 includes the `popt_off__sw__k2_negctrl`
arm which keeps the CSR-elimination but charges the mask DRAM. The delta
between `popt_off__isa__k2` and `popt_off__sw__k2_negctrl` is the
ISA-delivery benefit (which is what an HPCA reviewer cares about for
the architectural claim).

### Selectivity advantage (saved/fill — INTERPRET WITH CSR CAVEAT)

| cell | DROPLET k8 saved/fill | mode 6 isa k2 saved/fill | apparent advantage |
|---|---:|---:|---:|
| com-orkut L3=2MB | 1.003× | 1.249× | +24% |
| com-orkut L3=8MB | 1.000× | **3.476×** | **+247%** |
| soc-LJ L3=2MB | 1.000× | 1.278× | +28% |
| soc-LJ L3=8MB | 1.001× | 2.367× | +136% |

**HONEST CAVEAT**: mode 6's `saved/fill > 1` partly reflects:
- (a) real cross-edge reuse of HOT vertices that POPT identifies, AND
- (b) CSR-elimination inflating the "saved" numerator (mode 6's mem_acc
  doesn't include CSR reads that baseline does)

For the paper, report saved/fill alongside `total_memory_traffic`
reduction (the absolute metric), not as a standalone claim.

## KILL-2: PASS — policy parity within noise

GRASP vs ECG:DBG_ONLY on web-Google L3=2MB:
- GRASP: 3,046,368 demand misses
- ECG:DBG_ONLY: 3,046,777 demand misses
- delta: +0.013% ✅ within noise

POPT vs ECG:POPT_PRIMARY on web-Google L3=2MB:
- POPT: 2,967,915 demand misses
- ECG:POPT_PRIMARY: 2,965,011 demand misses
- delta: -0.098% (ECG:POPT_PRIMARY slightly BETTER due to DBG tiebreak)

All 5 policies (LRU, GRASP, POPT, ECG:DBG_ONLY, ECG:POPT_PRIMARY) loaded
and ran with `status=ok` on both email-Eu-core and web-Google.

## Mechanism explanation (paper §4)

For PR-style graph workloads with large property arrays:
- DROPLET-style sequential prefetching: high coverage (8 prefetches/edge)
  → each prefetched line ≈ one-use (saved/fill ≈ 1.0)
- Mode 6 POPT-ranked offline selection: identifies HOT vertices (hubs in
  power-law graphs) that are demanded by multiple source vertices in the
  same iteration → fewer prefetches but higher reuse (saved/fill > 1.0)

At larger L3, mode 6's advantage GROWS because:
- More cache → more lines survive between accesses
- POPT-selected hubs benefit from longer cache residency (multi-use)
- DROPLET's blanket coverage hits more lines but most are one-use

## Caveats from the Phase 0 audit (still apply)

- Our DROPLET implementation is **streamMPP1-class**, not the full
  decoupled L2-streamer + MC-property-prefetcher architecture from
  Basak HPCA'19. Per paper Section IV.B, full DROPLET beats streamMPP1
  by 4-12.5%, so mode 6's 20-70% advantage may shrink against a
  faithful DROPLET. Document explicitly in paper §5.

- ISA-delivered mode 6 (CHARGED=0) is still an idealized cache_sim model.
  Cycle-accurate Sniper validation deferred (todo s67-future-sniper-magic).

- Single-core results don't address shared-cache contention. Canonical
  L3=2MB matches DROPLET/GRASP per-core LLC (2MB/core × 4-8 cores).

- "Mode 6 saves 99% of demand misses" is MISLEADING due to CSR-elimination
  in the mode 6 path; use `total_memory_traffic` reduction (20-70%) as
  the headline metric.

## What's next

**RUNNING NOW**: Phase 2 baselines sweep (5 graphs × 5 policies + DROPLET-style)
at `results/ecg_experiments/hpca_mode6/baselines_v1/` (~2-3h wall).

**THEN**: Phase 3 ECG build-up (mode 2 + mode 6 amp=0/1 ISA + mode 6 sw
negctrl) on 5 graphs (~1-2h wall).

**THEN**: Phase 4 sensitivity (L3 sweep {1MB, 2MB, 4MB, 8MB, 16MB} on 1-2
cells to validate the "advantage grows with L3" claim).

**DEFERRED**: Phase 5 cycle-accurate Sniper validation.

