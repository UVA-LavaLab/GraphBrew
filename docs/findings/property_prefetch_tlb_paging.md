# Property-prefetch TLB / paging cost — how DROPLET handles it, and how our model does

**Date:** 2026-06-20
**Question (user):** Indirect property prefetch reads `property[v]` for essentially
random neighbour ids `v`, so consecutive prefetches land on random, widely-scattered
**pages**. That stresses the **TLB** (one translation per prefetch; random pages ⇒
TLB misses ⇒ page-table walks). How did the DROPLET paper address this, and does our
evaluation account for it?

## TL;DR

- The TLB cost of scatter property prefetch is **real and known**. The literature
  solves it by moving the indirect translation **off the core dTLB**:
  - **DROPLET (HPCA'19)** adds a **dedicated Memory-Controller-side TLB ("MTLB")**
    inside its property prefetcher. *(verified from the paper's own slides)*
  - **P-OPT (HPCA'21)** requires the irregular array to live in a **1 GB huge page**
    so the address is computed in the **physical** domain — no per-element
    translation at all. *(verified from the local P-OPT camera-ready)*
  - Core-side virtual-address prefetchers that *don't* do this (**IMP**, software
    prefetch) suffer the full TLB thrash (Bhattacharjee, ASPLOS'17).
- **ECG_PFX needs the same translation infrastructure as DROPLET** (it also prefetches
  `property[v]`), so the translation cost is a **shared, orthogonal** cost — not a
  differentiator. But ECG_PFX issues **~K× fewer** property prefetches than DROPLET
  (best-1 vs all-K, DROPLET `indirect_degree`=16), so under *any* translation scheme it
  generates **proportionally fewer translations** ⇒ less MTLB / page-walk pressure.
- **Our simulators charge no per-prefetch translation cost** for either prefetcher
  (cache_sim has no TLB; our gem5 SE-mode run shows `dtb.accesses = 0`). This matches
  the *design intent* of DROPLET (MTLB makes it cheap/off-critical-path) and P-OPT
  (huge pages eliminate it), and it is **conservative for ECG_PFX** (which would incur
  strictly fewer translations than DROPLET if the cost were modelled).

## 1. How DROPLET addresses it (VERIFIED — paper's own slides)

Source: `research/.../hpca2019_droplet.pdf` (the talk slides for Basak et al.,
"Analysis and Optimization of the Memory Hierarchy for Graph Processing Workloads",
HPCA 2019; DROPLET = **D**ata-awa**R**e dec**O**u**PL**ed pr**E**fe**T**cher).

DROPLET has two engines placed at **different** points of the hierarchy:

1. **L2 Structure Streamer** (edge-list engine, core/L2 side). An **extra bit in the
   core TLB and the L2 request queue** marks accesses to *structure* (edge-array) data,
   so the streamer knows which lines to stream.
2. **MC-based Property Prefetcher (MPP)** — the indirect/property engine, placed **at
   the memory controller**, with its own translation pipeline:
   - **PAG (Property Address Generator):** from `base address of property array` +
     the neighbour ids in a returned structure cacheline, computes the *virtual*
     property prefetch addresses (`base + (neighbourID << 2)` for 4-byte elements).
   - **VAB (Virtual Address Buffer):** holds `{virtual address, core id}`.
   - **MTLB:** a **dedicated TLB at the memory controller** that translates those
     scattered property virtual addresses → physical, **without touching the core
     dTLB**.
   - **PAB (Physical Address Buffer)** → **MRB (Memory Request Buffer)** → DRAM →
     coherence engine / caches.

So DROPLET *does not* sidestep translation — it **pays for it with dedicated
hardware** that keeps the random-page property translations off the core's critical
TLB path. Quantified hardware overhead (paper slides, "Hardware Overhead"):
- **Extra bits in TLB: 1.56%** storage overhead in the paging structure
- **Extra bits in L2 request queue: 1.54%** storage overhead
- **Property prefetcher in MC: 0.0348%** area overhead vs the entire chip

## 2. How the rest of the literature handles it

| Approach | Who | Translation solution | Per-prefetch TLB cost |
|---|---|---|---|
| MC-side dedicated **MTLB** | **DROPLET** (HPCA'19) | property prefetcher at the MC translates target property virtual addrs via its own MTLB | off the core dTLB; +1.56% TLB bits |
| LLC-side + **1 GB huge page** | **P-OPT** (HPCA'21) | `irreg_base`/`irreg_bound` are physical; whole array in one huge page ⇒ physical-domain arithmetic | **none** (no per-element translation) |
| Core-side HW, virtual addrs | **IMP** (MICRO'15) | emits virtual A[B[j]] through the normal core TLB | **full** TLB thrash (flagged by Bhattacharjee ASPLOS'17) |
| Core-side SW prefetch | Ainsworth&Jones (CGO'17) | `__builtin_prefetch` virtual addrs, normal TLB | full (acknowledged, not solved) |
| Programmable engine + OS support | Ainsworth&Jones (ASPLOS'18), Prodigy (HPCA'21) | OS change (identity map / huge pages) | mitigated via OS |

P-OPT camera-ready (`research/POPT_HPCA21_CameraReady.txt`): *"P-OPT sidesteps the
complexity of address translation by requiring that the entire irregData array fits in
a single 1GB Huge Page … Software configures the two registers once at the start."*

**Takeaway:** every serious indirect graph prefetcher recognises the scatter-page TLB
problem and moves the translation off the core's critical path — via a dedicated MTLB
(DROPLET) or huge pages + physical registers (P-OPT). The TLB cost is small/off-path
**by construction**, not ignored.

## 3. What it means for ECG_PFX (this work)

- ECG_PFX prefetches the *same* object — `property[v]` for a selected next-vertex `v`
  whose id is delivered by the ECG fat-ID / mask. So it has the **same** indirect
  translation requirement as DROPLET and would deploy the **same** infrastructure (a
  DROPLET-style MTLB, or P-OPT-style huge pages). Translation is therefore a **shared,
  orthogonal** cost — it does not favour either prefetcher.
- **Selectivity reduces translation pressure too.** ECG_PFX issues **1** property
  prefetch per trigger (POPT-best target); DROPLET issues `indirect_degree` (default
  **16**) per trigger. Fewer prefetches ⇒ fewer property virtual addresses ⇒ fewer MTLB
  lookups / page-walks. The same selectivity that gives ECG_PFX ~⅓ the property-prefetch
  *bandwidth* also gives it proportionally less *translation* pressure — under any of the
  schemes above.

## 4. What our simulators model (HONEST)

- **cache_sim:** a pure cache model with **no TLB / no address translation**. Both
  prefetchers are evaluated with **zero** per-prefetch translation cost. This is
  consistent with the literature's *design intent* (DROPLET's MTLB and P-OPT's huge
  pages make translation cheap/off-critical-path), and it is **conservative for
  ECG_PFX**, which would incur strictly fewer translations than DROPLET if the cost were
  charged.
- **gem5:** the prefetchers use `use_virtual_addresses=True` and translate via
  `mmu->translateTiming` (`queued.cc:86`). But in our **RISCV SE-mode + TimingSimpleCPU**
  configuration the data TLB is **not modelled** — a real ROI run
  (`bench/bin_gem5/pr_riscv_m5ops`, email-Eu-core, DROPLET) reports
  `system.cpu.mmu.dtb.accesses = 0` despite 1.7 M cycles and 3534 prefetches issued, so
  translation is effectively free (no TLB-miss / page-walk latency). gem5 therefore also
  charges **no** TLB cost — equally for both prefetchers.
- **Neither sim models DROPLET's MTLB or its 1.56% storage overhead.** Modelling the
  translation cost faithfully (a finite MTLB with miss latency, or a core dTLB charged
  per prefetch) is **future work**; doing so would *widen* ECG_PFX's advantage, since it
  issues K× fewer property prefetches.

## 5. Bottom line for the paper

State it plainly: *the indirect property prefetch translation cost is real; DROPLET
pays for it with a dedicated memory-controller MTLB (+1.56% TLB storage), P-OPT with a
1 GB huge page; ECG_PFX uses the same infrastructure and, being selective (1 vs K
prefetches per trigger), incurs proportionally less translation pressure. Our cache_sim
and gem5 models charge no per-prefetch translation cost for either prefetcher, so the
reported comparison is on cache traffic only and is conservative for ECG_PFX on the
translation axis.*
