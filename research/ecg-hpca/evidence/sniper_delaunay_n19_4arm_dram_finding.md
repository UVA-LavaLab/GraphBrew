# Finding — Sniper delaunay_n19/pr mode 6 cycle-accurate measurement

**Original date:** 2026-06-06 (sprint 6f-6 closeout)
**Substantial revision:** 2026-06-07 (sprint 6f-7 Phase 1.4 + 3.1 corrections)
**Cross-reference:** `research/ecg-hpca/evidence/sprint_6f-7_mode6_charged_audit.md`
**Status:** Pre-sprint-6f-7 numbers SUPERSEDED. See "Current honest picture" below.

> **⚠️ Important context:** The pre-sprint-6f-7 numbers in this doc
> (mode 6 = 8.15× DRAM, "DROPLET wins absolutely") were measurement
> artifacts caused by 4 separate bugs identified in the sprint 6f-7
> rubber-duck audit:
> 1. ROI placement bug (sg_kernel cycle-accurately simulated offline mask construction)
> 2. O(E × 256) linear-scan dedup (1.6B cycles of simulated comparisons)
> 3. cache_sim CSR-double-read bug (`bench/src_sim/pr.cc` mode 6/7 inner loop)
> 4. Treating CHARGED=1 (software-delivered mask) as the paper-faithful model
>    when the design intent is CHARGED=0 (ISA-delivered)
>
> See `sprint_6f-7_mode6_charged_audit.md` for the full audit.

## Current honest picture (sprint 6f-7 Phase 1.4 + 3.1)

After all 4 rubber-duck fixes landed in commits `9812edf9` (Sniper fat-mask
only), `3ea738fd` (ROI placement + O(1) bitmap dedup), and `28ffaede`
(Sniper AMPLIFY support), the delaunay_n19/pr cycle-accurate measurements
on Sniper (LRU L3=1MB, prefetcher attached at L2, -i 1 iteration) are:

| Arm | DRAM requests | DRAM/base | l3_miss_rate | pf_useful | hints emitted | Wall |
|---|---:|---:|---:|---:|---:|---:|
| baseline (no-pfx) | 594,187 | 1.00× | 0.9621 | 0 | 0 | 33min |
| DROPLET LH=8 (L2) | 593,708 | **1.00×** | 0.9609 | **223,165** | 0 | 45min |
| mode 6 amp=0 CHARGED=1 (L2) | 969,530 | **1.63×** | 0.9566 | 22,142 | 111,171 | 35min |
| mode 6 amp=1 CHARGED=1 (L2) | 968,740 | **1.63×** | 0.9570 | 30,023 | 131,210 | 31min |

### Key cycle-accurate findings

1. **The "8× DRAM regression" was an artifact** of pre-fix bugs. Real ratio
   is **1.63× baseline** for CHARGED=1 mode 6 (per-edge software-delivered
   mask). Confirmed across two independent runs (amp=0 and amp=1).

2. **AMPLIFY=1 adds useful prefetches but doesn't reduce DRAM** in CHARGED=1.
   pf_useful goes from 22K → 30K (+35%); DRAM stays flat at 968-969K.
   The mask-read DRAM cost (~375K extra cache lines from reading the
   25 MB mask array on a 1 MB L3) dominates total DRAM regardless of
   AMPLIFY. The extra prefetches are real, just don't help DRAM accounting
   when mask reads dominate.

3. **DROPLET LH=8 is DRAM-neutral** (1.00× baseline) — its prefetches
   replace demand misses 1:1 (cleanly converting demand → prefetch fills,
   same total DRAM bytes). Mode 6 CHARGED=1 ADDS to DRAM because mask
   reads are net-new traffic that doesn't replace anything in the
   baseline workload.

4. **In CHARGED=1 mode, mode 6 is uncompetitive with DROPLET.** This is
   the honest cycle-accurate answer to "does mode 6 beat DROPLET on real
   hardware?" — under software-delivered mask, NO.

## What sprint 6f-7 added beyond the cycle-accurate picture

Sprint 6f-7 Phase 2 ran the same comparison in cache_sim WITHOUT charging
the mask DRAM cost (CHARGED=0, modeling an ISA-extension that delivers
the mask payload at zero memory cost — gem5's `ecg_extract` custom-0
opcode design intent). At CHARGED=0 + AMPLIFY=1, mode 6 DOES beat DROPLET
on the large corpus graphs (soc-LiveJournal1, com-orkut) — +13-16% more
demand misses saved at -12-13% less total DRAM, fully validated against
DROPLET LH=16 (paper default).

**The defensible cross-sim paper claim** is therefore CONDITIONAL:
- ✅ **With ISA-extension hardware (CHARGED=0)**: mode 6 amp=1 is a
   Pareto-improvement over DROPLET on large graphs. Validated in cache_sim
   only; Sniper validation pending the magic-instruction work
   (todo `s67-future-sniper-magic`).
- ❌ **Without ISA hardware (CHARGED=1)**: mode 6 is uncompetitive with
   DROPLET. Confirmed in BOTH cache_sim and Sniper (delaunay_n19 here +
   4 corpus cells in cache_sim).

## Implications for the paper

The original sprint 6f-5 framing — "Mode 6 is +14% bandwidth-efficient
vs mode 2" — does NOT survive the audit. It was an artifact of the
cache_sim CSR-double-read bug (the denominator was inflated, making
mode 6's rate look favorable).

The defensible HPCA-grade claims after sprint 6f-7:

- **Headline (conditional)**: "ECG mode 6 with ISA-delivered POPT-ranked
  fat-mask prefetcher (AMPLIFY=1) saves 13-16% more demand misses than
  DROPLET LH=8 at matched bandwidth and reduces total DRAM by 12-13% on
  large PR graphs. The benefit requires ISA-extension hardware delivery
  (CHARGED=0); software-delivered fat-mask (CHARGED=1) is uncompetitive
  due to per-edge memory cost."

- **Substrate (unconditional)**: ECG_DBG ≡ GRASP eviction equivalence
  holds (-7.24pp L3 miss rate vs LRU), with 2× more compact metadata
  than POPT (gate 287). Independent of the prefetch axis.

- **Cycle-accurate corroboration**: cache_sim CHARGED=1 prediction
  (mode 6 ~1.6× DRAM vs baseline, no demand benefit) is confirmed in
  Sniper on delaunay_n19. The CHARGED=0 claim remains unvalidated in
  cycle-accurate.

## Confidence notes

- **Sample size**: 1 Sniper cell (delaunay_n19/pr) at 1 attach level (L2)
  with 1 iteration count (-i 1). The 4 cache_sim corpus cells (cit-Patents,
  soc-LiveJournal1, com-orkut, web-Google) cover broader topology
  diversity but at idealized cache modeling.
- **What's NOT in this finding**: Sniper validation at L1 attach (paper-
  default DROPLET attachment); Sniper validation at CHARGED=0 (requires
  Sniper backend magic-instruction work).
- **Wall time budget**: each Sniper delaunay run is ~30-90 min. com-orkut
  / soc-LiveJournal1 would be 13-22 hours per arm — intractable for
  routine validation.

## Provenance

- Sniper cell dirs (CURRENT, post-sprint-6f-7):
  - baseline: `/tmp/graphbrew-pfx-sniper-delaunay-i1-none/`
  - DROPLET: `/tmp/graphbrew-pfx-sniper-delaunay-i1-droplet/`
  - mode 6 amp=0 CHARGED=1: `/tmp/graphbrew-pfx-sniper-delaunay-i1-fatmask-v2/`
  - mode 6 amp=1 CHARGED=1: `/tmp/graphbrew-pfx-sniper-delaunay-i1-fatmask-v2-amp1/`
- Sniper cell dirs (DEPRECATED, pre-sprint-6f-7 buggy):
  - mode 6 CHARGED=1 (buggy): `/tmp/graphbrew-pfx-sniper-delaunay-i1/` (3.23M L3 accesses, 4.84M DRAM)
  - mode 6 CHARGED=0 (buggy): `/tmp/graphbrew-pfx-sniper-delaunay-i1-charged0/`
- Binary: `bench/bin_sniper/sg_kernel` post-commit `28ffaede` (with
  AMPLIFY support, kernel-side bitmap dedup, fat-mask-only iteration,
  ROI placement fixed)
- Configuration: LRU L3=1MB/16w, L2=256KB/8w, L1d=32KB/8w, 64B line,
  DDR4-2400, prefetcher attached at L2, 256-entry kernel dedup window
