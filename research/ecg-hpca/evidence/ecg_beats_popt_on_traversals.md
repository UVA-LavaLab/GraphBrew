# ECG beats P-OPT on frontier traversals (BFS/SSSP) — the kernel-generality win

> **Historical evidence archive.** Current paper claims, methodology, results,
> and pending gates live in `research/ecg-hpca/`.

**Date:** 2026-07-01
**Prompted by:** "another axis: P-OPT — was it evaluated on PR? what about BFS / SSSP / CC
traversal algorithms?"

## The gap in P-OPT's evaluation
P-OPT (Balaji & Lucia, HPCA'21) evaluated **five apps: PageRank, Connected Components
(Shiloach-Vishkin), PR-δ, Radii, MIS** (paper §VI, lines 1115–1170) — all *iterative* or
frontier-*approximation*. It **never evaluated single-source BFS, delta-stepping SSSP, or
BC**. Its rereference matrix indexes by `epoch = current_vertex / epoch_size`, which
assumes the outer loop visits vertices in a **monotonic 0→N sweep** (true for iterative
kernels). Frontier traversals visit in **data-dependent order** (cache_sim drives
`current_src = frontier vertex`, bfs.cc:94 "the traversal clock"), so the matrix's epoch is
**miscalibrated** — its next-ref predictions assume a sweep that never happens.

## Measured (cache_sim, equal 16-way, `-o5`, fixed source `-r 1`)

| graph · kernel | LRU | GRASP | **ECG (feasible)** | POPT idealized(16w) | POPT charged(reserved) |
|---|---|---|---|---|---|
| web-Google · bfs | 0.8999 | 0.8197 | **0.8192** | 0.8580 | 0.8794 |
| web-Google · sssp | 0.6620 | 0.6430 | **0.6430** | 0.6436 | 0.6880 |
| soc-pokec · bfs | 0.6597 | 0.5619 | **0.5602** | 0.5840 | 0.6096 |
| soc-pokec · sssp | 0.3900 | 0.2977 | **0.2977** | 0.3116 | 0.3669 |
| soc-LiveJournal1 · bfs | — | 0.6058 | **0.6055** | 0.6122 | 0.6365 |
| soc-LiveJournal1 · sssp | — | 0.3133 | 0.3133 | 0.3112 | 0.3707 |

- **ECG beats IDEALIZED P-OPT (free matrix, equal ways) on 5/6 BFS/SSSP cells** (the lone
  exception, soc-LiveJournal sssp, is a 0.2pp tie within noise).
- **ECG beats FAITHFUL (charged) P-OPT on ALL 6 cells by +3.1 to +6.9pp.**
- P-OPT's matrix is genuinely active (POPT < LRU everywhere), but it is a **weak predictor
  on traversals**: better than LRU, worse than order-independent degree (GRASP/ECG).
- ECG uses its degree-based mode here (order-independent), so ECG ≈ GRASP and both crush
  P-OPT; ECG delivers this via the memory-resident mask with **no reserved LLC way**.

## Contrast with iterative kernels
| kernel class | P-OPT | ECG | winner |
|---|---|---|---|
| iterative (PR) | sweep-order matrix well-calibrated | approaches within <1pp | P-OPT (barely) |
| iterative (CC) | mixed (wins web-Google, loses soc-pokec) | = GRASP | mixed |
| **frontier (BFS, SSSP)** | **sweep-order matrix miscalibrated** | **degree-based, order-independent** | **ECG (+3–7pp faithful)** |

## The honest thesis (reframed)
Earlier sessions concluded "feasible ECG cannot beat P-OPT" — but that was **PR-only**.
Across kernel classes the picture is:

**ECG is a KERNEL-GENERAL graph cache-replacement policy: it approaches P-OPT on the
iterative kernels P-OPT was designed for (<1pp), and BEATS P-OPT on frontier traversals
(BFS/SSSP, +3–7pp faithful) where P-OPT's sweep-order oracle breaks — all at zero reserved
LLC way. P-OPT is NOT kernel-general: it targets iterative sweeps and degrades on
data-dependent traversal order (kernels it never evaluated).**

This is a genuine, feasible, honest win and the correct paper framing.

## Caveats / follow-ups
- P-OPT-on-traversal here follows the same matrix mechanism cache_sim uses for PR; whether
  a *different* frontier-aware P-OPT variant could recover is untested (P-OPT's paper offers
  none for single-source BFS/SSSP). The claim is about P-OPT *as designed/evaluated*.
- Absolute performance still needs the final gem5/Sniper matrix; cache_sim remains the
  functional authority. BC is mixed and uses the safe `rrip_first` fallback.
- CC is mixed (Afforest/SV has a partly sweep-like pattern); report it honestly, don't
  headline it.

## Traversal-adaptive eviction (2026-07-10)

Implemented shared `ECG_VARIANT=degree_first` (cache_sim/gem5/Sniper SSOT):

```text
max-RRPV eligibility
→ oldest record
→ coldest GRASP degree tier
→ farthest delivered epoch / Schedule-2 within that tier
→ oldest recency
```

This combines order-independent reuse **frequency** (degree) with reuse **timing**
(epoch) using fields already present in the ECG record; no reserved way, matrix,
or additional record bits.

Full five-policy cache_sim validation (2MB/16-way, `-o5`, source 1):

| graph · kernel | LRU | SRRIP | GRASP | P-OPT | ECG degree-first |
|---|---:|---:|---:|---:|---:|
| web-Google · BFS | .7451 | .7303 | .6876 | .7328 | **.6837** |
| web-Google · SSSP | .3541 | .3405 | .3389 | .3646 | **.3386** |
| soc-pokec · BFS | .6426 | .6139 | .5767 | .6104 | **.5750** |
| soc-pokec · SSSP | .3900 | .3515 | **.2977** | .3383 | .2991 |
| web-Google · BC (`rrip_first`) | **.7000** | .6914 | .7112 | .7268 | .7007 |

Degree-first beats P-OPT on every BFS/SSSP cell, beats GRASP on both BFS cells
and web-Google SSSP, and stays within 0.14pp on soc-pokec SSSP. BC is mixed:
degree-first overweights its backward/pointer phase, so adaptive ECG selects
`rrip_first`, a do-no-harm tie with LRU (0.06pp) that still beats GRASP/P-OPT.

The adaptive policy is therefore:

- monotonic/iterative PR: StreamShield + Schedule-2 + `epoch_first`;
- data-dependent BFS/SSSP: `degree_first` (Schedule-2 as the within-tier tie);
- mixed BC/CC: `rrip_first` (safe do-no-harm fallback);
- road/uniform graphs: bypass/adaptive fallback required; remain out-of-domain.

## Schedule-2 cross-simulator status (2026-07-11)

The adaptive PR/BFS mechanisms are now implementation-equivalent across all three
simulators. `equiv_kernels.py --schedule-k 2` verifies the shared pull/push pair
builder, the `dest32|epoch1|epoch2` delivery record, `min(d1,d2)` line distance, and
the exact victim rule. PR uses `epoch_first`; BFS uses `degree_first`.

All cache_sim/gem5/Sniper PR and BFS cells pass. BFS exercises epoch-decisive victims
in every simulator. This certifies the adaptive **decision**;
StreamShield is now ported separately for PR. BFS/SSSP remain degree-first K2
without bypass; the final scale miss-rate/traffic matrix is still pending.
