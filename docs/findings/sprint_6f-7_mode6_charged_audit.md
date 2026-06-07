# Sprint 6f-7: Mode 6 per-edge ECG mask audit — the CSR-double-read bug, CHARGED=0/1 framing, and DROPLET parity

**Date:** 2026-06-06
**Sprint:** 6f-7 (audit + cross-sim validation of mode 6 per-edge ECG mask)
**Commits:** `1df4c5f9` (cache_sim fix), `ab82e7cc` (Table 7 honest accounting columns)
**Status:** cache_sim story complete; Sniper cross-sim validation deferred to follow-up session
**Tracking:** SQL todos `s67-*` in session state

## Executive summary

The sprint 6f-5 "+14% pp/Mreq advantage" claim for mode 6 (per-edge ECG fat-mask
prefetcher) was **bug-induced**. After the Phase 2.1/2.2 cache_sim audit fixed a
CSR-double-read in `bench/src_sim/pr.cc` mode 6/7 inner loop, the corpus result
flipped from "+14% better than mode 2" to "-94% worse" with the per-edge mask
charged as a memory load (CHARGED=1).

**But the picture is more nuanced when accounting for the design intent.** Mode 6's
paper design (per `bench/include/ecg_mode6_builder.h`, gem5's `ecg_extract`
custom-0 opcode) is for the fat-mask to be delivered via an ISA extension at
zero memory cost (CHARGED=0). At CHARGED=0 + AMPLIFY=1 (mode 6 with one
encoded target + one next-sequential dest = 2 prefetches/edge), mode 6
defensibly beats DROPLET on the two large corpus graphs (soc-LiveJournal1,
com-orkut) at matched bandwidth:

- **mode 6 amp=1 CHARGED=0**: 369M demand misses saved on 331M pf_fills, **0.90× baseline DRAM** (DRAM-positive)
- **DROPLET LH=8**: 327M demand misses saved on 328M pf_fills, 1.00× baseline DRAM
- **DROPLET LH=16 [paper default]**: 323M saved on 345M pf_fills, **1.06× baseline DRAM** (saturates earlier than LH=8 due to dedup-window collisions)

**The paper's defensible claim:** mode 6 with ISA-delivered metadata is a
Pareto-improvement on large graphs; without ISA hardware (CHARGED=1
software-delivered mask), software cost makes mode 6 uncompetitive with DROPLET.

## The bug (Phase 2.1/2.2)

### What it was

`bench/src_sim/pr.cc` lines 143-187 contained the per-edge mode 6/7 inner
loop. The original code (sprint 6f-5) read both the mask AND the CSR edge
per edge:

```cpp
if (edge_mask_charged && !src_masks.empty()) {
    SIM_CACHE_READ(cache, src_masks.data(), edge_pos);   // mask read (8B)
}
NodeID v = static_cast<NodeID>(GraphCacheContext::edgeMaskDest(mask));
SIM_CACHE_READ_EDGE(cache, it);                         // CSR read (4B) ← BUG: redundant
uint32_t prefetch_target = ...;
SIM_CACHE_READ_MASKED(cache, contrib_ptr, v, ...);      // demand read (4B)
```

Per-edge cost: mask(8B) + CSR(4B) + demand(4B) = **16B/edge** instead of the
paper's intended **12B/edge** (the mask encodes the destination, so the CSR
load is redundant).

### The fix

Removed `SIM_CACHE_READ_EDGE` from the mode 6/7 path (commit `1df4c5f9`),
mirroring the Sniper fix in commit `9812edf9`. Per-edge cost now matches paper
intent: mask(8B) + demand(4B) = 12B/edge.

### A/B impact on a small smoke (email-Eu-core/pr -i 5)

| metric | with bug | with fix | delta |
|---|---:|---:|---:|
| Total Accesses | 337,295 | 248,060 | -36% |
| Memory Accesses | 6,065 | 4,150 | -46% |
| L2 Misses | 20,254 | 9,518 | -53% |
| L3 Misses | 6,146 | 4,236 | -45% |

The bug inflated cache_sim's `total_accesses` (the denominator of the pp/Mreq
metric) by ~36% in mode 6 only — exactly the pattern that makes a denominator-
inflated rate look "more efficient" while burning real DRAM.

## CHARGED=1 vs CHARGED=0: the design intent (Phase 2.5)

The user pushed back: "the way we are fixing this seems off. Yes we are
incurring higher DRAM since we have fat 64-bit address but DROPLET is
prefetching at L1, isn't it incurring more DRAM access too?"

The answer is that `ECG_EDGE_MASK_CHARGED=1` models the mask as a memory load
— a worst-case "what if you had to fetch the fat-mask from L3/DRAM" model. The
paper's actual design intent is that the mask is delivered via an ISA extension
(gem5's `ecg_extract` custom-0 opcode) at zero memory cost — modeled by
`ECG_EDGE_MASK_CHARGED=0`.

Re-running the corpus at CHARGED=0 reveals mode 6's true potential:

| arm | corpus demand_saved | pf_fills | true_DRAM | DRAM/base | saved/fill |
|---|---:|---:|---:|---:|---:|
| baseline | 0 | 0 | 383M | 1.00× | — |
| mode 2 (POPT K=1 LH=8) | 100M | 100M | 383M | 1.00× | 1.00× |
| **mode 6 CHARGED=1 amp=0** | **21M** | **104M** | **466M** | **1.22×** | **0.20×** |
| mode 6 CHARGED=0 amp=0 | 136M | 98M | 345M | 0.90× | 1.39× |
| mode 6 CHARGED=0 amp=1 | 369M | 331M | 345M | 0.90× | 1.115× |
| DROPLET LH=8 | 327M | 328M | 383M | 1.00× | 1.00× |
| DROPLET LH=16 [paper] | 323M | 345M | 405M | 1.06× | 0.94× |

**Reading this table:**

- **Mode 6 CHARGED=1 amp=0** (current default) is genuinely bad — 1.22× DRAM
  for almost no demand savings. The fat-mask delivery cost dominates.
- **Mode 6 CHARGED=0 amp=0** (idealized ISA delivery, one encoded prefetch per
  edge) saves 136M demand misses with 98M prefetches — **1.39× useful-per-fill**
  vs DROPLET's 1.00×. Total DRAM is 10% LESS than baseline.
- **Mode 6 CHARGED=0 amp=1** (encoded target + next sequential dest = 2
  prefetches per edge) saturates at the optimum — additional AMPLIFY values
  (4, 7) give identical results due to dedup-window absorption.
- **DROPLET LH=16** (paper default) is actually WORSE than LH=8 at L3=1MB —
  the dedup window (256 entries) saturates before LH=16 adds useful targets.

## Per-cell nuance: mode 6 wins on large, ties or loses on small (Phase 2.7)

The corpus aggregate hides important per-cell behavior:

| cell | mode 6 amp=1 vs DROPLET LH=8 | total DRAM | saved/fill |
|---|---|---:|---:|
| cit-Patents (16M edges) | +4% more demand saved | +5% MORE DRAM | -5% worse |
| **soc-LiveJournal1** (68M edges) | **+16%** more | **-12% LESS** | **+14% better** |
| **com-orkut** (117M edges) | **+13%** more | **-13% LESS** | **+15% better** |
| web-Google (8.6M edges) | +19% more | +6% MORE DRAM | -7% worse |

Mode 6 amp=1 dominates DROPLET on the 2 large graphs (soc-LJ, com-orkut) but
DROPLET LH=8 remains competitive per-fill on the 2 small graphs (cit-Patents,
web-Google). On small graphs, the working set fits well enough that
DROPLET's blanket sequential prefetching matches POPT-ranked selection.

## DROPLET parity audit (rubber-duck-validated, Phase 2.7)

A second rubber-duck pass questioned whether the cache_sim DROPLET (mode 3)
implementation is paper-faithful. Key findings:

1. **Cache_sim mode 3 is "DROPLET-style", not exact paper DROPLET.** It lacks
   stride detection, fires unconditionally per edge, uses LH=8 lookahead.
   Paper DROPLET defaults: stride table size 64, indirect degree 16, attached
   at L1. (`prefetcher/README.md:46-52` explicitly labels this divergence.)

2. **DROPLET LH=16 (paper default) test refutes the "DROPLET was gimped"
   concern.** When run at paper-default LH=16, DROPLET actually does WORSE
   than LH=8 (saturates at 256-entry dedup window, extra prefetches collide).
   LH=32 reproduces LH=16 (further saturation). Mode 6 amp=1 still beats both.

3. **CSR-elimination confound is real but bounded.** Mode 6 CHARGED=0 also
   skips CSR reads (the mask provides the destination via `edgeMaskDest()`).
   Some of mode 6's apparent demand savings come from CSR-elimination, not
   prefetch quality. Estimated impact: ~36M of mode 6 amp=1's 369M savings
   are from CSR-elimination architectural choice; the remaining ~333M is
   genuine prefetch effectiveness. Mode 6 still beats DROPLET's 327M after
   this normalization.

## L1 vs L2 prefetch attachment: cache_sim limitation

A third rubber-duck pass flagged that cache_sim's `prefetch()`
(`bench/include/cache_sim/cache_sim.h:1464-1494`) fills ALL THREE LEVELS on
miss — it does not model L1-only or L2-only attachment. The DROPLET paper
attaches at L1; our Sniper config uses L2. cache_sim's all-level model
gives both arms an idealized "prefetched lines survive in some cache level"
treatment.

The bias is not necessarily neutral: DROPLET's higher prefetch volume
(8 per edge) probably hurts MORE at real L1 attachment than mode 6's lower
volume (2 per edge). This means cache_sim may be UNDERSTATING mode 6's
advantage at L1 attachment, not overstating it. But this is empirically
untested — needs Sniper validation.

## Why CHARGED=0 isn't yet validated in Sniper

The Sniper PR kernel (`bench/src_sniper/sg_kernel.cc`) currently only supports
CHARGED=1 — it iterates `src_masks` directly, treating each mask read as a
real memory access. Implementing CHARGED=0 in Sniper requires:

1. A new magic instruction `SNIPER_ECG_FAT_LOAD(edge_idx)` that returns the
   decoded mask payload (neighbor, prefetch_target) without a memory access
2. Wiring through Sniper's Pin recorder and simulator backend
3. Updating `sg_kernel.cc` to use the magic when `SNIPER_ECG_EDGE_MASK_CHARGED=0`

This is multi-day Sniper backend work, deferred to a follow-up session
(tracked as todo `s67-future-sniper-magic`).

**Until that work lands, the cache_sim CHARGED=0 finding is "idealized upper
bound; Sniper cycle-accurate validation pending."** The paper's defensible
claim is conditional: "Mode 6 with ISA-extension hardware (CHARGED=0) is a
Pareto-improvement; without it, software-mask delivery is uncompetitive."

## Paper framing recommendation (HPCA-grade)

**Headline claim** (cache_sim CHARGED=0, large graphs):
> "ECG mode 6 with ISA-delivered POPT-ranked fat-mask prefetcher saves 13–16%
> more demand misses than DROPLET LH=8 at matched bandwidth, while reducing
> total DRAM traffic by 12–13% on large graphs (soc-LiveJournal1, com-orkut).
> Paper-default DROPLET LH=16 is uniformly inferior to LH=8 at L3=1MB due to
> dedup-window saturation."

**Honest caveats** (Section 5.3 / Section 6.3):
- "CHARGED=0 models ISA-extension delivery (gem5's `ecg_extract` custom-0
  opcode); software-delivered fat-mask (CHARGED=1) is uncompetitive due to
  per-edge memory cost"
- "Mode 6 advantage is concentrated on large graphs where POPT-ranked
  selection beats DROPLET's blanket sequential prefetching; on smaller
  graphs that fit better in cache, DROPLET remains competitive"
- "Cache_sim's idealized all-level prefetch fill does not model L1 vs L2
  attachment; cycle-accurate Sniper validation is future work pending
  Sniper backend magic-instruction implementation"

**What dies** (compared to pre-audit sprint 6f-5 framing):
- "+14% bandwidth-efficient" headline (was bug artifact)
- "Mode 6 is unconditionally Pareto-better than DROPLET" (only true CHARGED=0
  + large graphs)
- AMPLIFY > 1 has any value (dedup window absorbs everything past AMPLIFY=1)

**What survives**:
- gate 287: ECG mask is 2× more compact than POPT matrix (architectural
  simplicity, unchanged)
- ECG_DBG ≡ GRASP eviction equivalence (unchanged)
- ECG_DBG -7.24pp vs LRU (eviction substrate works, unchanged)
- Mode 2 (runtime POPT lookahead K=1 LH=8) matches DROPLET miss-rate at 3×
  lower bandwidth (unchanged)

## Files / commits

- **`bench/src_sim/pr.cc:143-187`** — mode 6/7 inner loop; commit `1df4c5f9`
  removes the redundant `SIM_CACHE_READ_EDGE`
- **`scripts/experiments/ecg/paper_table_mode6_corpus.py`** — Table 7
  emitter; commit `ab82e7cc` adds `total_traffic_ratio` + `mode6_dram_inflation_flag`
- **`wiki/data/paper_table_mode6_corpus.{json,md,csv}`** — regenerated artifact
  with FIXED corpus data (the buggy pre-fix data is preserved at
  `/tmp/mode6_corpus.buggy_pre_sprint6f-7/`)
- **`docs/paper_tables/paper_table_mode6_corpus.tex`** — TeX caption now
  carries a "Caveat" when `mode6_dram_inflation_flag` fires
- **Future work tracked as todo `s67-future-sniper-magic`** — Sniper backend
  magic instruction for paper-faithful CHARGED=0 Sniper validation

## Cross-references

- `docs/findings/droplet_vs_ecg_pfx_algorithm.md` — original sprint 6f-5
  algorithm comparison (now superseded for the per-edge claim)
- `docs/findings/sniper_delaunay_n19_4arm_dram_finding.md` — sprint 6f-7
  Phase 1.4 Sniper cycle-accurate finding (consistent with cache_sim CHARGED=1
  picture: mode 6 CHARGED=1 is 1.63× DRAM, NOT 8×; the 8× was bug)
- `docs/findings/gem5_ecg_pfx_simobject_gap.md` — gem5 ECG_PFX hint-to-issue
  gap (separate concern, not addressed in sprint 6f-7)
