# Phase 0 — Baseline Faithfulness Audit v1

**Date:** 2026-06-07 (HPCA mode 6 reset)
**Status:** Draft v1 — based on source-code inspection. Empirical parity
checks deferred to KILL-2 (relaxed to sanity) and later sweeps.
**Cross-references:** `docs/findings/hpca_evaluation_plan_v1.md`,
`docs/findings/hpca_naming_convention_v1.md`

## Purpose

This audit documents what each baseline policy/prefetcher in our codebase
actually implements vs the paper it claims to reproduce. Mismatches must
be either fixed or explicitly labeled in the paper as "PaperName-style"
rather than "PaperName" reproduction.

## Eviction policies

Source: `bench/include/cache_sim/cache_sim.h:34-56` (EvictionPolicy enum),
`graph_cache_context.h:164-195` (ECGMode enum).

| our policy | paper | impl details | status |
|---|---|---|---|
| `LRU` | (classical) | least-recently-used, standard baseline | ✅ paper-faithful |
| `FIFO`, `RANDOM`, `LFU`, `PLRU` | (standard) | reference implementations | ✅ paper-faithful |
| `SRRIP` | Jaleel ISCA'10 | 3-bit RRPV, Static RRIP | ✅ paper-faithful |
| `PIN` | Faldu HPCA'20 baseline | LRU + pin high-reuse graph regions | ⚠️ approximation of GRASP's pinning |
| **`GRASP`** | **Faldu HPCA'20** | **SRRIP eviction + tier-aware insertion via `classifyGRASP()` (3-tier: HOT/WARM/COLD by degree)** | **✅ paper-faithful intent** |
| **`POPT`** | **Balaji HPCA'21** | **Phase 1: evict non-graph data first. Phase 2: evict by furthest next-reference distance using rereference matrix.** | **✅ paper-faithful intent** (no DRRIP NUCA emulation) |
| `ECG` mode `DBG_PRIMARY` | (ours) | DBG tier primary, P-OPT secondary | n/a — our variant |
| `ECG` mode **`POPT_PRIMARY`** | (ours, derived) | **P-OPT signal primary, DBG as TIEBREAK** | ⚠️ **NOT identical to pure POPT** (added DBG tiebreak) |
| `ECG` mode **`DBG_ONLY`** | (ours, GRASP-equiv) | **GRASP-style insertion + plain SRRIP victim** | ⚠️ **intended GRASP-equivalent** (verify with KILL-2 sanity) |
| `ECG` mode `ECG_EMBEDDED` | (ours) | per-edge embedded P-OPT hint primary | n/a — mode 6 variant |
| `ECG` mode `ECG_EPOCH_EMBEDDED` | (ours) | epoch-compact P-OPT hint primary | n/a |
| `ECG` mode `ECG_COMBINED` | (ours, Hawkeye-inspired) | combined DBG+POPT at insertion | n/a |

### Critical clarifications for KILL-2

The original KILL-2 demanded `GRASP == ECG:DBG_ONLY` AND `POPT == ECG:POPT_PRIMARY` within 5%. After this audit:

- **`GRASP` vs `ECG:DBG_ONLY`**: should be tight (both are intentionally GRASP-class). Sanity check appropriate.
- **`POPT` vs `ECG:POPT_PRIMARY`**: **expected to diverge** because `ECG:POPT_PRIMARY` adds a DBG tiebreak that pure `POPT` doesn't have. Not a bug — by design.

KILL-2 was correctly relaxed to "activity sanity" only.

## Prefetchers

Source: `bench/src_sim/pr.cc` (cache_sim), `bench/include/sniper_sim/snipersim/.../droplet_prefetcher.cc` (Sniper), repo README files.

| our impl | paper | impl details | status |
|---|---|---|---|
| no prefetcher (`prefetcher: none`) | (baseline) | no prefetch | ✅ |
| stride prefetcher | (HW default) | **NOT implemented in cache_sim**; Sniper inherits the default Sniper config but we don't have an "HW stride" arm explicitly | ⚠️ **gap**: P-OPT/GRASP papers compare against HW stride; we don't |
| `ECG_PFX` mode 1 (degree-ranked) | (ours) | runtime degree-ranked pick from K-window | n/a — our variant |
| `ECG_PFX` mode 2 (POPT-ranked, runtime) | (ours, runtime POPT) | runtime POPT-ranked pick from K-window | ✅ self-contained |
| **`DROPLET` (mode 3 in cache_sim)** | **Basak HPCA'19** | **next-K sequential property prefetches per edge; NO stride detector, NO L2-streamer/MC-property decoupled architecture, NO L1 attachment** | ❌ **streamMPP1-class** (paper-DROPLET is decoupled L2-streamer + MC-property-prefetcher; per paper Section IV.B DROPLET beats streamMPP1 by 4-12.5%) |
| `ECG_PFX` mode 4 (far-future from hot table) | (ours) | global hot-table | n/a — our variant |
| **`ECG_PFX` mode 6 (per-edge fat-mask, paper HEADLINE)** | (ours, paper) | offline POPT-ranked per-edge mask; CSR doubling bug fixed in sprint 6f-7 commit `1df4c5f9` | ✅ self-contained; **CHARGED knob distinguishes software (`sw`) vs ISA-extension (`isa`) delivery** |
| `ECG_PFX` mode 7 (cross-iter) | (ours) | mode 6 + cross-iteration | n/a — our variant |
| Sniper DROPLET overlay (`prefetcher/droplet`) | Basak HPCA'19 | sideband-based; `prefetch_degree=1` default, `indirect_degree=16` (matches paper), `stride_table_size=64` (matches paper), attached at L2 (paper says L3) | ⚠️ **partial paper faithfulness**: degree mismatch + attach level mismatch + no MC-property decoupled architecture |
| gem5 DROPLET overlay | Basak HPCA'19 | separate impl, untested in HPCA mode 6 sweep | ⚠️ untested |
| gem5 `ecg_extract` opcode | (ours, paper's ISA design) | RISC-V custom-0; ECG_PFX SimObject receives hints but pfIssued=0 documented gap | ⚠️ **blocked by SimObject hint-to-issue gap** (`docs/findings/gem5_ecg_pfx_simobject_gap.md`) |

### Critical clarification for paper claims

When the paper says "we compare vs DROPLET":
- **What we actually compare against**: streamMPP1-class (L2 streamer + per-edge sequential property prefetcher).
- **NOT**: paper-faithful full DROPLET (decoupled L2-streamer + MC-property-prefetcher architecture)
- The DROPLET paper shows the full decoupled architecture beats streamMPP1 by 4-12.5%
- Our mode 6 advantage of 13-16% over our DROPLET-style implies a potential narrower margin over full DROPLET

**Paper-honest phrasing**: "We compare mode 6 against DROPLET-style sequential prefetching, an approximation of DROPLET (Basak HPCA'19) that omits the decoupled L2-streamer + MC-property architecture. Full DROPLET reproduction is left for follow-up work."

## Cache hierarchy

| param | DROPLET paper | P-OPT paper | GRASP paper | **Ours (canonical)** |
|---|---|---|---|---|
| Cores | 4-core | 8-core | 8-core | **1-core** |
| L1d | 32KB/8w, 4c | 32KB/8w, 3c | 32KB/4-8w + 16-stream stride | 32KB/8w |
| L2 | 256KB/8w, 8c | 256KB/8w, 8c | 256KB/8w, 6c | 256KB/8w |
| **LLC** | **8MB shared (2MB/core)** | **24MB DRRIP NUCA (3MB/core)** | **16MB NUCA (2MB/core)** | **2MB (1MB/core in scaled-down form)** |
| Prefetcher attach | L3 (decoupled L2 + MC) | none | none | L2 (Sniper) / all-level (cache_sim) |
| DRAM | DDR3, 45ns | unspec | unspec | DDR4-2400 |
| Simulator | Sniper | Sniper+Pin-cache-sim | Sniper | Sniper+cache_sim |

**Canonical for our paper**: 1-core, L3=2MB. This matches DROPLET/GRASP per-core LLC (2MB/core) and is a defensible scaled-down config for single-thread evaluation. Sensitivity sweep at L3 ∈ {1, 2, 4, 8, 16 MB}.

**Acknowledged divergences:**
1. **Single-core** vs paper 4-8 core: no shared-cache contention modeled. Mitigation: future multicore sanity check.
2. **No HW stride prefetcher arm** in cache_sim: P-OPT and GRASP papers test against HW stride + their replacement policy. Mitigation: add stride arm in Phase 2 (or document as a paper limitation).
3. **cache_sim's all-level prefetch fill** doesn't model L1/L2/L3 attachment differences: irrelevant since cache_sim is internally fair, but matters for cycle-accurate cross-validation.

## Mode 6 specifics (our contribution)

Source: `bench/include/ecg_mode6_builder.h`, `bench/include/graphbrew/partition/cagra/popt.h`

### What mode 6 actually does

For each CSR edge `(u, v)`:
1. **Offline (preprocessing)**: compute P-OPT rereference matrix.
2. **Offline (mask build)**: for a window of K next edges of u's in-neighbor list, find the candidate with the lowest AVERAGE rereference rank (not exact Belady) and encode as `prefetch_target` in the 64-bit fat-mask.
3. **Runtime**: read the fat-mask per edge; decode `(dest_id, prefetch_target)`; emit a prefetch hint for `prefetch_target`.

### Per the rubber-duck (sprint 6f-7):

> "Mode 6 uses P-OPT-derived **average line rank**, not exact per-edge next-reference distance. The result is more interesting this way: a coarse offline rank still beats blanket prefetching on large/skewed graphs."

**Paper-honest phrasing**: "Mode 6 encodes a P-OPT-derived line-rank prefetch hint per CSR edge; the encoding is offline-precomputed using the rereference matrix's lookahead-window average, not an exact Belady oracle."

### CHARGED axis (delivery model)

| value | what it models |
|---|---|
| `CHARGED=1` (sw delivery) | mask is loaded from a separate 64-bit array per edge → real memory traffic |
| `CHARGED=0` (isa delivery) | mask is delivered via ISA extension (gem5's `ecg_extract`) → no separate memory load |

In cache_sim, `CHARGED=0` simply skips the `SIM_CACHE_READ` for the mask. This is an upper bound (oracle); a fully-faithful Model A (packed-edge / fat-CSR) would still pay 8B/edge load instead of 4B/edge CSR. We currently report `CHARGED=0` as the upper bound (no separate Model A column).

### Bandwidth axis (AMPLIFY)

| AMPLIFY=N | emissions per edge |
|---|---|
| 0 | 1 (just the encoded target) |
| 1 | 2 (encoded + 1 sequential next-dest) |
| 4 | 5 |
| 7 | 8 (matches DROPLET LH=8 bandwidth) |

Sprint 6f-7 Phase 2.6 found AMPLIFY saturates at 1 in cache_sim (dedup window absorbs extras). The canonical headline is `popt_off__isa__k2`.

## What's MISSING from the audit (deferred work)

- [ ] Verify GRASP `classifyGRASP()` tier boundaries match Faldu paper's hot/warm/cold definitions
- [ ] Verify our P-OPT rereference matrix encoding matches Balaji paper's compression
- [ ] Empirical GRASP vs ECG:DBG_ONLY parity check (KILL-2 sanity)
- [ ] Verify DROPLET Sniper overlay's stride table behavior matches paper's confidence ≥ 2 semantics
- [ ] HW stride prefetcher arm in cache_sim for paper-faithful comparison vs GRASP/P-OPT baselines

## Summary verdict

| baseline | confidence | label for paper |
|---|---|---|
| LRU, SRRIP, RANDOM | ✅ high | "standard LRU/SRRIP/random" (no qualifier) |
| GRASP | ✅ high (intent matches paper) | "GRASP (Faldu HPCA'20)" — flag in audit any tier boundary mismatch found later |
| P-OPT (POPT policy) | ✅ high (intent matches paper) | "P-OPT (Balaji HPCA'21)" — single-core scaled L3 acknowledged |
| DROPLET (cache_sim mode 3) | ⚠️ medium | "DROPLET-style sequential prefetcher (approximation of Basak HPCA'19; lacks decoupled L2-streamer/MC-property architecture)" |
| DROPLET (Sniper overlay) | ⚠️ medium | same as above + "with stride table size 64, indirect degree 16 matching paper defaults" |
| mode 6 (our contribution) | ✅ high | "ECG mode 6 — per-edge offline-POPT fat-mask prefetcher with `{sw,isa}` delivery options" |

For HPCA submission, the paper must include a "Methodology" subsection
acknowledging the DROPLET-style approximation and the single-core
scaled-down L3 config explicitly.
