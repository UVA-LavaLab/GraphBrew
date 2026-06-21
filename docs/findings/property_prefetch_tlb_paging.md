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
  `property[v]`), so translation is a **shared** cost — neither target is harder to
  translate. **Measured at page granularity** (the faithful TLB metric — a 4 KB page
  holds ~1024 properties): both prefetchers touch the **same** distinct pages (895 on
  web-Google), so with a working-set-sized MTLB or huge pages their translation cost is
  **equal** (naive "ECG_PFX touches fewer pages" is **false**). Only under an *undersized*
  MTLB (entries ≪ working-set pages ⇒ thrash) does ECG_PFX's selectivity give ~K× fewer
  misses, tracking its request reduction.
- **Our simulators charge no translation *latency*** (cache_sim has no TLB timing — we
  added a distinct-page + finite-MTLB-miss *proxy* but it costs no cycles; our gem5
  SE-mode run shows `dtb.accesses = 0`). Per the measurement this omission is
  **neutral-to-favourable for ECG_PFX, never adverse**: equal under a working-set MTLB /
  huge pages, favourable under an undersized one.

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

## 3. What it means for ECG_PFX (this work) — MEASURED at page granularity

- ECG_PFX prefetches the *same* object — `property[v]` for a selected next-vertex `v`
  whose id is delivered by the ECG fat-ID / mask. So it has the **same** indirect
  translation requirement as DROPLET and would deploy the **same** infrastructure (a
  DROPLET-style MTLB, or P-OPT-style huge pages). Translation is therefore a **shared**
  cost — neither prefetcher's target object is harder to translate than the other's.
- **But raw prefetch count is *not* a faithful TLB-pressure metric** (a 4 KB page holds
  ~1024 4-byte properties), so we instrumented cache_sim to count the **distinct pages**
  the prefetch targets touch and the **misses of a finite LRU MTLB**
  (`CACHE_PFX_MTLB_ENTRIES`). Measured (web-Google, L3 512 kB, lookahead 8; property
  array = **895** 4 KB pages = 2 MB-pages; artifact `wiki/data/ecg_pfx_tlb_pressure.md`):

  | prefetcher | fills | distinct 4 KB pages | distinct 2 MB pages | MTLB-128 misses | MTLB-8192 misses |
  |---|---:|---:|---:|---:|---:|
  | DROPLET (all-K) | 6.27 M | **895** | 2 | 6.37 M | **895** |
  | ECG_PFX (best-1) | 2.28 M | **895** | 2 | 2.17 M | **895** |

- **Reading it (this corrects the naive "fewer prefetches ⇒ fewer translations"):**
  - With an **infinite TLB, huge pages, or any MTLB ≥ the working set** (the 8192-entry
    column, which covers the 895-page footprint), the two prefetchers are **identical**:
    both incur exactly the 895 *compulsory* page translations, because both ultimately
    sweep the whole property array. So under DROPLET's own MTLB (sized to the working
    set) or P-OPT's 1 GB huge page, **translation cost is EQUAL — neither prefetcher has
    an advantage**, and the naive "ECG_PFX touches fewer pages" is **false**.
  - Only when the **MTLB is too small to hold the working set** (the 128-entry column,
    128 ≪ 895 ⇒ thrash) does ECG_PFX's selectivity help: misses then track the *request*
    count, so ECG_PFX has ~2.9× fewer MTLB misses — the same factor as its bandwidth
    advantage.
- **Honest conclusion:** ECG_PFX's translation cost is **never worse** than DROPLET's and
  is **strictly better only in the small/thrashing-MTLB regime**; under the working-set-
  sized MTLB or huge pages that DROPLET and P-OPT actually deploy, the two are equal.

## 4. What our simulators model (HONEST)

- **cache_sim:** a pure cache model with **no TLB latency** in the timing path. We added
  a **translation-pressure proxy** (distinct-4 KB/2 MB-page counters + a finite LRU MTLB
  miss model, `CACHE_PFX_MTLB_ENTRIES`, default 128) so the page-level comparison above
  can be made, but it charges **no cycles** for translation — both prefetchers are timed
  with zero translation latency. Per §3 this is **at worst neutral for ECG_PFX** (equal
  under a working-set MTLB; favourable under a small one), never adverse.
- **gem5:** the prefetchers use `use_virtual_addresses=True` and translate via
  `mmu->translateTiming` (`queued.cc:86`). But in our **RISCV SE-mode + TimingSimpleCPU**
  configuration the data TLB is **not modelled** — a real ROI run
  (`bench/bin_gem5/pr_riscv_m5ops`, email-Eu-core, DROPLET) reports
  `system.cpu.mmu.dtb.accesses = 0` despite 1.7 M cycles and 3534 prefetches issued, so
  translation is effectively free (no TLB-miss / page-walk latency). gem5 therefore also
  charges **no** TLB cost — equally for both prefetchers.
- **Neither sim charges translation *latency*** (DROPLET's MTLB miss penalty / page
  walks). Adding a finite-MTLB *timing* model is future work; per the measured page data
  it would leave the comparison unchanged under a working-set-sized MTLB and only widen
  ECG_PFX's edge under an undersized one.

## 5. Bottom line for the paper

State it plainly and precisely: *indirect property-prefetch translation is a real cost;
DROPLET pays for it with a dedicated memory-controller MTLB (+1.56 % TLB storage), P-OPT
with a 1 GB huge page. ECG_PFX prefetches the same object and would use the same
infrastructure. At page granularity both prefetchers touch the **same** property pages
(895 on web-Google), so with a working-set-sized MTLB or huge pages their translation
cost is **equal**; under an undersized MTLB ECG_PFX incurs ~K× fewer misses (tracking its
request reduction). Our models charge no translation latency, so the reported comparison
is on cache traffic; the omission is neutral-to-favourable for ECG_PFX on the translation
axis, never adverse.*
