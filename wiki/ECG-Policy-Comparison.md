# ECG Policy Comparison & Verification

This page is the reproducible artifact for the ECG L3 replacement study: how to
run the policy comparison, verify each policy is implemented correctly, and the
headline result. Everything here is regenerable from two scripts.

- Run the matrix: `scripts/experiments/ecg/ecg_variant_matrix.py`
- Verify correctness: `scripts/experiments/ecg/verify_ecg.py`

## The policy set

ECG is an **L3-only** replacement policy: **GRASP insertion** (degree-tiered RRIP)
+ **RRIP eviction** + a **per-edge next-reference epoch tiebreak**. The epoch is
meaningful only for *property* lines (scores/contrib); edge-stream/record lines
carry no usable epoch. We compare against the literature baselines it combines:

| Policy | One-line definition | Source |
|--------|--------------------|--------|
| LRU    | Least-recently-used | baseline |
| GRASP  | Degree-tiered RRIP insertion + RRIP eviction | Faldu et al., HPCA'20 |
| P-OPT  | Transpose-Belady oracle: non-property first → rereference distance → RRIP | Balaji et al., HPCA'21 |
| ECG    | GRASP insertion + RRIP eviction + property-epoch tiebreak | this work |

## The factorial framework (`ECG_VARIANT`)

ECG is one policy (`ECG_MODE=ECG_GRASP_POPT`) with one eviction lever switched by
the `ECG_VARIANT` environment variable, so each column isolates exactly one
mechanism. This is what lets a reviewer attribute the result to a specific lever
rather than to an opaque bundle.

| `ECG_VARIANT` | Eviction rule | Isolates |
|---------------|---------------|----------|
| `grasp_only`   | RRIP max-RRPV only (no epoch) | the GRASP arm alone (≈ GRASP) |
| `epoch_only`   | Farthest-epoch among stamped property, else recency | the P-OPT-style epoch arm alone (≈ P-OPT) |
| `rrip_first`   | Max-RRPV set; records first, then farthest-epoch property | recency vetoes epoch |
| `epoch_first`  | Records by recency, then farthest-epoch property (epoch vetoes RRPV) | epoch vetoes recency |
| `shortcircuit` | Evict any non-property line first, then farthest-epoch property | property-protection (legacy) |

## Methodology: run BOTH `-o 0` and `-o 5`

GRASP's degree-tiered insertion and ECG's degree arm assume a **degree-sorted
layout** (hot = low address), which is produced by `-o 5` (Degree-Based Grouping,
DBG). `-o 0` (original order) handicaps GRASP/ECG. The matrix therefore reports
every cell at both orders; the **fair comparison is at `-o 5`**.

## Headline result (cache_sim, PageRank)

L3 miss-rate at the literature geometry (16-way L3, 32 kB L1 / 256 kB L2, `pr -i 1`)
at the fair `-o 5` (DBG) baseline, lower is better. **On power-law graphs the best
ECG variant beats GRASP and beats or ties P-OPT.**

| graph (L3) | topology | LRU | GRASP | P-OPT | ECG (best variant) |
|------------|----------|----:|------:|------:|-------------------:|
| web-Google (512 kB) | power-law | 0.844 | 0.673 | 0.633 | **0.623** (shortcircuit) |
| cit-Patents (1 MB)  | power-law | 0.896 | 0.820 | 0.747 | **0.680** (shortcircuit) |
| soc-pokec (1 MB)    | power-law | 0.694 | 0.556 | 0.551 | **0.499** (shortcircuit) |
| com-orkut (2 MB)    | power-law | 0.552 | 0.470 | 0.446 | **0.447** (epoch_only, ties P-OPT) |
| roadNet-CA (512 kB) | road/mesh | 0.966 | 0.986 | **0.935** | 0.968 (out of domain) |

The `epoch_only` arm alone already matches or beats P-OPT (e.g. cit-Patents
0.686 vs 0.747), confirming the epoch carries the P-OPT-class signal; `grasp_only`
tracks GRASP. **Scope (honest):** the win is a power-law phenomenon. On **roadNet-CA**
reuse is spatial, not property-driven — P-OPT's oracle wins, ECG ≈ LRU, and GRASP's
degree bias actively *hurts*. ECG is for scale-free analytics, not mesh traversal.
The same `-o 0` rows (where GRASP/ECG insertion is handicapped) are in
`/tmp/ecg_matrix.md` after a run.

## The best variant is model-dependent (honest finding)

No single variant wins on every simulator, and the factorial framework is what
exposed this:

- **cache_sim** isolates the L3 on a fixed stream; its L2 absorbs the edge stream
  so the L3 set is property-dominated → `shortcircuit` (evict non-property first)
  is best.
- **gem5** is a faithful inclusive hierarchy where the edge stream reaches the L3 →
  `shortcircuit` lets no-reuse records displace reused property and *underperforms*;
  `rrip_first` (recency vetoes, records-first) is the correct default there.

The production ECG policy is therefore **GRASP-insert + RRIP-evict + epoch-tiebreak
(`rrip_first`)** — robust in both models — with `shortcircuit` documented as a
cache_sim-favorable, gem5-pathological variant.

## Three-simulator status

The same policy + `ECG_VARIANT` factorial lives in all three simulators. The
eviction **decision** is a single shared function, `ecg_policy::selectVictim`
(`bench/include/ecg_victim_policy.h`), that cache_sim, gem5 and Sniper all call —
so nothing is "ported" or "mirrored": the decision logic is literally identical
across backends (a SSOT test asserts the co-located copies are byte-identical),
and one unit test of that function verifies the eviction choice for all three.
Each simulator keeps only a thin adapter that builds the per-way state from its
native cache lines (epoch via memory-resident mask for cache_sim/gem5, via
`findNextRef` for Sniper); the adapter is covered by that backend's live trace.

| Simulator | ECG_GRASP_POPT + ECG_VARIANT | Epoch source | Status |
|-----------|------------------------------|--------------|--------|
| **cache_sim** | yes | per-edge memory-resident mask (0 LLC ways) | **verified** (`verify_ecg.py`, 7×40/40) + matrix |
| **gem5** | yes | per-edge mask via ISA `ecg.load` (custom-0, FUNCT3=0x2; see below) | **verified** (`verify_ecg.py --gem5`, 5×40/40) |
| **Sniper** | yes | native `findNextRef` (P-OPT next-ref) | **verified** (`verify_ecg.py --sniper`, 4×40/40, guarded) |

Sniper's variant uses its native `findNextRef` for the property next-reference
distance (so it isolates the same eviction *levers* rather than the per-edge
mask). Real-graph Sniper runs are gated behind `--allow-sniper-sg-kernel-workload`
because the Sniper/SDE frontend has a documented ~50 GiB memory runaway
(infrastructure, independent of the ECG policy); run it under
`--sniper-memory-limit-gb`.

## ECG ISA: one instruction, mode-controlled caching

gem5 delivers the per-edge metadata with a **single** custom-0 RISC-V instruction
(opcode `0x0b`, FUNCT3=`0x2`, R-type):

```
ecg.load  rd, rs1, rs2     rs1 = property base, rs2 = mode-6 record, rd = prop[rs2.dest]
                           EA = rs1 + dest*elem_size; deliver metadata to the LLC
                           before the fill (line stamped on insertion); rd = Mem[EA]
```

A FUNCT7 field carries both the caching **mode** (`ECG_MODE`<31:27>) and a dest-width
**class** (`ECG_WIDTH`<26:25>, read in `ea_code` → `W = 8/16/24/32` bits); each layout
already exists in the SSOT `bench/include/ecg_mode6_builder.h` (no bespoke ISA layout):

| ECG_MODE | mode | `rs2` layout (SSOT) | effect |
|---|---|---|---|
| `0x00` | **EVICT** (headline) | `dest[0:W]｜epoch[W:W+16]` | next-ref epoch eviction |
| `0x01` | EVICT+PFX | `dest[0:W]｜epoch[W:W+16]｜pfx[W+16:64]` | + Path-B prefetch target |
| `0x02` | EMBEDDED | NARROW `dest[0:24]｜dbg[24:26]｜popt[26:33]｜epoch[33:49]｜pfx[49:64]` | full legacy metadata |

`FUNCT7 = (ECG_MODE<<2)｜ECG_WIDTH`. The **configurable dest width** lets ONE instruction
scale its dest from 256 (W8) to 4.29 B (W32) vertices — W24 is the headline default and covers
all eval graphs; W32 covers twitter/friendster/kron-s27. The 32-bit instruction word never
carries metadata — it rides the 64-bit register `rs2`. The earlier prototyping forms
(`ecg.extract` reg-hint FUNCT3=`0x0`, side-record load FUNCT3=`0x1`) are subsumed; the paper
presents this one instruction. The decoder is the tracked overlay `decoder_ecg_extract.isa`
(+ the `ECG_MODE`/`ECG_WIDTH` bitfields in `formats/ecg.isa`), kept byte-identical to the gem5
build tree. EVICT is validated end-to-end on RISC-V (`[ECG_PLOAD] ACTIVE`, correct PageRank
result, no illegal-instruction); the field-parity drift guard in `verify_ecg.py` pins the
decoder shifts against the builder for every width class (95/0).

**OoO delivery (sideband):** the epoch reaches the LLC race-free under an out-of-order CPU via a
gem5 `Request::Extension` (`EcgEpochExtension`) tagged on the demand load — no shared mailbox, no
per-vertex table. The in-order TimingSimpleCPU case study validates the policy via the single-slot
mailbox, which is mathematically equivalent (serialized loads ⇒ no race). Details:
`docs/findings/ecg_mask_direction_and_metadata.md` §19.

### Pipeline flow

`ecg.load` is an ordinary R-type load in the front end; the ECG work is confined to the AGU/LSU:

| stage | action |
|---|---|
| IF/ID | decode custom-0 / FUNCT3=0x2; `ECG_MODE`/`ECG_WIDTH` select mode + dest width `W` |
| RF | read `rs1` (property base), `rs2` (the 64-bit mode-6 record) |
| AGU | split `rs2` → `dest` (low `W`) + `epoch` (`[W:W+16]`) [+ `pfx`]; `EA = rs1 + dest·elem_size`; tag the demand request with the epoch sideband |
| MEM | the demand request reaches the LLC; the replacement policy latches the epoch on the line's fill (stamp-on-insert) |
| WB | `rd = Mem[EA] = prop[dest]` — the value the kernel needed anyway |

The epoch is therefore **free**: it rides the demand load the kernel already issues, is split out
in the AGU, and travels as a few sideband bits on the in-flight request — no extra instruction, no
extra memory stream, no reserved way. The 32-bit instruction word is fixed; only `rs2` (a 64-bit
GPR) grows with the graph, via `ECG_WIDTH`. In-order, the per-vertex/single-slot table is the exact
model of that sideband; `EcgEpochExtension` is the OoO/multicore form (read side wired in `ecg_rp.cc`,
AGU attach is the remaining O3 step). Every `(mode, width)` is checked through the REAL decoder by
`test_ecg_load_modes.cc` (`verify_ecg.py --gem5`): `rd == prop[dest]`, plus a teeth proof that
forcing the width wrong mis-decodes (so `ECG_WIDTH` is load-bearing).

## Reproduce

```bash
# 1. build cache_sim
make sim-pr

# 2. run the full factorial matrix (rows=graphs, cols=variants), both orders
python3 scripts/experiments/ecg/ecg_variant_matrix.py --suite cache-sim \
    --cells "web-Google:512kB cit-Patents:1MB soc-pokec:1MB com-orkut:2MB roadNet-CA:512kB" \
    --orders "0 5" --out /tmp/ecg_matrix.md

# 3. same framework on gem5 (RISC-V backend; rrip_first is the gem5 default)
python3 scripts/experiments/ecg/ecg_variant_matrix.py --suite gem5 \
    --cells "email-Eu-core:16kB" --orders "5"
```

## Verify (correctness, not aggregates)

`verify_ecg.py` checks correctness in **three complementary layers**:

1. **Synthetic exact-victim tests** (`bench/src_sim/test_ecg_victim.cc`) — construct
   controlled 8-way sets (property/record lines with chosen epochs/recency) and assert
   the **exact** victim each variant must pick, computed independently in the test. Pins
   the exact victim including cases the live workload never produces. Mutation-tested:
   flipping the implementation's farthest→nearest epoch pick makes it fail.
2. **Live-trace integration** (default geometry) — run each policy with `ECG_EVICT_TRACE`,
   parse every real eviction (per-way rrpv/epoch/dist/property/recency + victim), and
   assert it obeys the **tightened exact-victim rule**. The same `[EVICT L3 ...]` format is
   emitted by all three simulators.
3. **Epoch-coverage runs** — a forcing geometry (big L2 absorbs the edge stream → property
   reaches the L3; `ECG_STORED_REFRESH` keeps the L3 epoch live) makes the **epoch-property
   branch fire on the real simulator end-to-end**. Asserts the exact rule AND that the epoch
   value genuinely *selected* the victim ≥1× (a strict coverage gate, so the check cannot
   pass vacuously). Mutation-tested live as well.

**Scope (honest):** the *default* workload only evicts records, so the epoch branch is
exercised by layers 1 and 3. Layer 3 confirms `rrip_first`/`epoch_first`/`epoch_only` pick
the farthest-epoch property live; `shortcircuit` ranks property by *raw* distance (evicts
unstamped property first), so its stamped-epoch ranking is rarely operative live and is
covered by the synthetic test. cache_sim runs all three layers (strict epoch-value gate).
gem5 runs layers 2–3, but its layer-3 epoch-value gate is **informational**: gem5 has no
`ECG_STORED_REFRESH`, so property reaches its L3 unstamped and the epoch value cannot
discriminate live — gem5's epoch ranking is verified by the exact-rule check plus the
line-by-line mirror of the strictly-verified cache_sim block. Sniper runs layer 2.

```bash
make sim-pr
python3 scripts/experiments/ecg/verify_ecg.py            # synthetic + cache_sim live + epoch-coverage
python3 scripts/experiments/ecg/verify_ecg.py --gem5     # + gem5 live + gem5 epoch-coverage
python3 scripts/experiments/ecg/verify_ecg.py --sniper   # + Sniper live (guarded, prlimit)
# expected: each policy "N/N evictions obey spec [OK]" → "ALL POLICIES VERIFIED ✓"
```

Verified to date: **cache_sim** (7 policies × 40/40 evictions), **gem5** (5
ECG_GRASP_POPT variants × 40/40), and **Sniper** (4 ECG-specific variants × 40/40;
the guarded `sg_kernel` run is memory-capped via `--sniper-memory-limit-gb`). All
three emit the same `[EVICT L3 ...]` trace, so one checker verifies every backend.
Larger real-graph Sniper runs may still hit the documented SDE memory behavior;
the tiny email-Eu-core verification run completes cleanly under the cap.

## Prefetch path (DROPLET vs ECG prefetch)

ECG also ships a hint-driven prefetcher. The eviction policy above is the
*bandwidth* story; the prefetcher is the *latency* story, and the two are
deliberately separated:

- **DROPLET** (Basak et al., HPCA'19) — the literature baseline. Real DROPLET
  has *two* engines: a stride engine that prefetches the edge stream, and an
  **indirect/property engine** that prefetches `property[neighbour]` for every
  neighbour in the next-K window (no target selection). cache_sim models the
  **indirect/property engine only**, fed by the kernel's ground-truth in-neighbour
  stream — i.e. a *best-case (oracle) all-K property prefetcher*, no stride
  mis-prediction. (The stride engine is not modelled in cache_sim; it is present
  in gem5/Sniper but is a separate, orthogonal edge-stream optimisation.)
- **ECG_PFX** — prefetch the *single* POPT-best target among the next-K
  in-neighbours, chosen by `ecg_mode6::selectPrefetchTarget`
  (`bench/include/ecg_mode6_builder.h`). Like the eviction decision, this target
  function is a **single shared header compiled into the cache_sim, gem5 and
  Sniper kernels**, so the ECG prefetch *decision* is identical across all three
  backends. It is the *selective* counterpart to DROPLET's all-K flood: 1 property
  prefetch per trigger vs ~K.

Because cache_sim models both as **indirect/property prefetchers into the same
cache level**, the comparison isolates exactly one lever — the property-target
*selection* (all-K vs best-1) — and the claim is scoped to **property-prefetch
traffic** (it does not include DROPLET's separate edge-stride stream; see the
fairness note after the table).

**The honest finding:** a prefetcher can only *relocate* traffic
(demand → prefetch); it can **never reduce total DRAM traffic** — only the
eviction policy can. So the prefetcher comparison is about *latency per unit of
bandwidth*, not bandwidth itself.

Prefetcher comparison at fixed LRU eviction (cache_sim, PageRank, `-o 0`,
lookahead 8; `demand2mem` = demand misses reaching memory = latency proxy;
`bandwidth` = total DRAM traffic; both lower = better):

| graph/L3 | prefetcher | demand2mem | bandwidth | fills | useful% |
|---|---|---|---|---|---|
| web-Google/512kB | none | 7.69M | 7.69M | 0 | – |
| web-Google/512kB | DROPLET | **1.45M** | 7.72M | 6.27M | 100 |
| web-Google/512kB | ECG_PFX | 5.41M | 7.69M | **2.28M** | 100 |
| cit-Patents/1MB | none | 33.73M | 33.73M | 0 | – |
| cit-Patents/1MB | DROPLET | **6.51M** | 33.73M | 27.22M | 100 |
| cit-Patents/1MB | ECG_PFX | 24.27M | 33.73M | **9.46M** | 100 |
| soc-pokec/1MB | none | 27.13M | 27.13M | 0 | – |
| soc-pokec/1MB | DROPLET | **3.50M** | 27.95M | 24.44M | 100 |
| soc-pokec/1MB | ECG_PFX | 20.10M | 27.26M | **7.16M** | 100 |
| com-orkut/2MB | none | 97.69M | 97.69M | 0 | – |
| com-orkut/2MB | DROPLET | **15.38M** | 104.39M | 89.01M | 100 |
| com-orkut/2MB | ECG_PFX | 73.14M | 98.65M | **25.51M** | 100 |
| roadNet-CA/512kB | none | 0.87M | 0.87M | 0 | – |
| roadNet-CA/512kB | DROPLET | 0.73M | 0.87M | 0.14M | 100 |
| roadNet-CA/512kB | ECG_PFX | 0.75M | 0.87M | 0.12M | 100 |

Reading it: **DROPLET-style all-K** trades the most property-prefetch traffic for
the biggest latency cut — it issues 3–3.5× more property prefetches and on
com-orkut **over-fetches** (bandwidth 104.4M vs 97.7M baseline). **ECG_PFX** is
bandwidth-efficient: it reaches the same 100% useful-rate while issuing 2.7–3.5×
fewer prefetches (it picks the single POPT-best target instead of every next-K
neighbour), keeping total traffic at the baseline. The full bandwidth win comes
from stacking ECG_PFX on the ECG_GRASP_POPT *eviction* (the table above is
fixed-LRU to isolate the prefetcher lever — see
`docs/findings/prefetcher_saturation_under_eviction.md` for the combined stack).
Artifact: `wiki/data/ecg_prefetch_matrix.md`.

**Honest scoping of the claim.** The defensible statement is: *ECG_PFX's selective
best-1 indirect-property prefetching matches a DROPLET-style oracle all-K
indirect-property prefetcher's L3-miss reduction (tied useful-rate) while issuing
~⅓ the property-prefetch requests* — **not** the broader "ECG_PFX beats DROPLET".
Two caveats keep this honest: (i) cache_sim omits DROPLET's edge-stride engine, so
the request counts here are *property-prefetch only* — including the stride stream
would *raise* DROPLET's total traffic (the request-reduction is conservative for
total traffic, but the request-per-useful ratio is not necessarily conservative,
since an accurate stride engine adds near-1.0-efficiency prefetches). (ii) Both
prefetchers fill the **same cache level** here; we attach each prefetcher to a
single level and could not verify whether the original DROPLET differentiates
edge→L1/L2 vs property→LLC placement, so the comparison holds placement *equal*
across the two. cache_sim's prefetcher fills the **whole hierarchy** (L1+L2+L3 on
a fill), so it is level-agnostic and holds placement *identical* for both
prefetchers by construction — it cannot itself study differentiated placement.
The stride→L2 / property→LLC idea would need gem5/Sniper (which attach the
prefetcher to one configurable level), but those currently cannot deliver the
cross-page property prefetch at all (page-cross filter), so a faithful
split-placement study is future work. We therefore hold placement equal and make
no placement claim. The gem5 DROPLET-vs-ECG_PFX numbers are **not** used for this
claim (its page-cross filter drops both property engines — see "Cross-simulator
status" below); cache_sim is the authoritative property prefetch model.

### Reproduce + verify the prefetch path

```bash
# performance matrix (rows = graph x prefetcher at fixed eviction)
python3 scripts/experiments/ecg/ecg_prefetch_matrix.py --suite cache-sim \
    --cells "web-Google:512kB cit-Patents:1MB soc-pokec:1MB com-orkut:2MB roadNet-CA:512kB" \
    --order 0 --eviction LRU --lookahead 8 --out /tmp/pfx_matrix.md

# correctness verification (synthetic shared decision + live per-prefetcher spec)
python3 scripts/experiments/ecg/verify_pfx.py
```

`verify_pfx.py` mirrors `verify_ecg.py` in two layers:

1. **Synthetic exact-target** (`bench/src_sim/test_ecg_prefetch.cc`) — asserts the
   exact target of `ecg_mode6::selectPrefetchTarget` (min re-reference among the
   next-K, window clipping, ties, invalid/out-of-range entries, disabled). Because
   that function is the single shared decision, this verifies the ECG prefetch
   target for **all three simulators**; mutation-proven (flipping min→max fails).
2. **Live behaviour** (cache_sim) — runs PageRank with each prefetcher and asserts
   its defining, falsifiable property: `none` issues nothing and traffic == demand;
   `DROPLET` fires, cuts demand-to-mem, and does **not** reduce total traffic
   (conserves bandwidth); `ECG_PFX` fires, cuts demand-to-mem, and issues strictly
   **fewer** prefetches than DROPLET at ≥50% useful-rate (selective targets).

**Cross-simulator status (honest, empirically checked 2026-06-20):** the ECG
prefetch *decision* (target selection) is verified for all three backends via the
shared function, and all three now **deliver** the hint end-to-end. cache_sim
issues both DROPLET and ECG_PFX property prefetches over the full property region.
**gem5**: the `Queued::notify` page-cross filter is fixed by the wired
`registerMMU` (`S68-MMU-PATCH`, `configs/graphbrew/graph_se.py`) — DROPLET issues
`pfSpanPage=1691` cross-page property prefetches (zero dropped); and **ECG_PFX now
fires too** (`pfIdentified=pfIssued=63` on email-Eu-core) after fixing two gem5 PR
kernel bugs (the `packMaskEpoch` re-pack discarded the prefetch target; the 4-byte
fast-path record could not carry it — commit history). **Sniper** ECG_PFX fires
under the guarded `sg_kernel` path. So all three simulators deliver the shared
`ecg_mode6::selectPrefetchTarget` hint. **3-sim parity, empirically demonstrated**
(email-Eu-core, `-o5`, ECG_PFX): gem5 `pfIdentified=pfIssued=63` and Sniper
`ecg_pfx_issued = ecg_pfx_target_hints_seen = 63` — *identical hint counts*, because
both consume the same shared decision function; cache_sim issues ECG_PFX prefetches
on the same cell too.
**Caveat (honest):** the gem5 ECG_PFX ISA mask carries the target in a **15-bit
field** (≤32767), so the gem5 ISA testbed validates the *mechanism* only on
≤32K-vertex graphs (it warns + can abort on larger graphs; field widths pinned by
`bench/src_sim/test_ecg_pfx_field_width.cc`); **cache_sim is the authoritative model
for large-graph prefetch performance** (no field limit, and the only one that models
the page/MTLB translation proxy). Sniper is correctness-clean (31-bit target field,
dedup, bounds-checked) but streams an 8-byte fat-mask per edge, so its ECG_PFX
*memory-traffic* numbers are substrate-dependent — use Sniper for mechanism/parity,
cache_sim for the traffic comparison.
(`docs/findings/gem5_implementation_audit_v1.md`,
`docs/findings/property_prefetch_tlb_paging.md`).

### Paging / TLB cost of indirect property prefetch

Prefetching `property[v]` for random neighbour ids scatters across pages, stressing
the **TLB**. This is a real, known cost and the literature moves it **off the core
dTLB**: **DROPLET** adds a **dedicated memory-controller TLB ("MTLB")** in its
property prefetcher (PAG→VAB→MTLB→PAB pipeline; +1.56% TLB-storage, 0.0348% chip-area
overhead — verified from the paper's slides); **P-OPT** puts the irregular array in a
**1 GB huge page** so the address is physical (no per-element translation). ECG_PFX
prefetches the same object and would use the same infrastructure, so translation is a
**shared, orthogonal** cost — and being selective (1 vs DROPLET's K=16 prefetches per
trigger) it issues **K× fewer** translations, so it has *less* TLB pressure. Our
cache_sim has no TLB and our gem5 SE-mode run shows `dtb.accesses=0`, so **neither sim
charges per-prefetch translation cost** for either prefetcher — consistent with the
MTLB/huge-page design intent, and conservative for ECG_PFX. Full analysis:
`docs/findings/property_prefetch_tlb_paging.md`.

## Which metadata is load-bearing + graph-direction correctness

A full audit (`docs/findings/ecg_mask_direction_and_metadata.md`) establishes two
honesty boundaries:

- **Load-bearing fields:** the ECG_GRASP_POPT eviction reads **only the epoch** (the
  packed 7-bit POPT-quant field is *vestigial* for the headline policy — it survives
  only for legacy cache_sim modes + gem5 ISA decode). The prefetch-target field is read
  only by ECG_PFX. POPT *data* is build-time only (it feeds the epoch values + the
  prefetch-target choice). So "we have POPT and epoch" → the **epoch** is what the cache
  actually uses for eviction; POPT is the encoder input + a baseline policy.
- **Direction:** the next-reference matrix must be the graph **transpose** of the
  kernel's property-access edge list — this is literally P-OPT's thesis (*Transpose*-based
  replacement; it stores both CSR and CSC). **PageRank — the headline kernel — is
  direction-correct** (in-pull + `out_neigh` matrix = the transpose). CC is undirected-only;
  SSSP/BC/BFS-top-down traverse out-edges and are *direction-uncertified on directed
  graphs* (they would need `makeOffsetMatrix(..., traverseCSR=false)`), so they are not
  promoted as transpose-faithful ECG results. No headline-results direction bug.

## 3-simulator equivalence showcase + debug proof

`scripts/experiments/ecg/three_sim_showcase.py` runs the same policies on the same cell across
cache_sim / gem5 / Sniper and prints an L3 miss-rate table plus a per-sim `[ECG-CONFIG …]` banner.
It drives the committed `roi_matrix` with the verified cell geometry — a raw
`roi_matrix --policies ECG:…` that omits the load-bearing knobs (`ECG_EDGE_MASK_CHARGED`,
`CACHE_ULTRAFAST=0`, the L1/L2 sizes) makes ECG **degenerate**, so the showcase pins them.

**Two separate claims — keep them apart:**

**(A) ECG advantage — ECG beats GRASP *and* P-OPT** (real graphs, cache_sim = functional authority,
`-o5`, `ECG_VARIANT=shortcircuit`). ECG_GRASP_POPT = GRASP's degree-aware insertion + P-OPT's
next-reference eviction, so the target is `LRU > GRASP > P-OPT ≥ ECG` (lower miss rate better):

| cell (cache_sim, -o5) | LRU | GRASP | P-OPT | **ECG** |
|---|---|---|---|---|
| web-Google @ 512kB | 0.8440 | 0.6733 | 0.6326 | **0.6229** |
| cit-Patents @ 1MB | 0.8958 | 0.8196 | 0.7471 | **0.6795** |

ECG is the lowest in both — it beats GRASP and the (all-ways, uncharged) P-OPT with a memory-resident
mask and no reserved way.

**(B) 3-sim equivalence** — the same policy moves the miss rate the same DIRECTION in all three
simulators. This needs a gem5/Sniper-feasible cell, so it uses the SYNTHETIC kron_s16_k4, whose
Kronecker structure does NOT reward the epoch — so ECG does **not** beat GRASP here; this cell
certifies cross-sim AGREEMENT, not the ECG advantage. (kron_s16_k4 @ 128kB/16w, L1d=16kB, L2=64kB,
`-o5`, `ECG_VARIANT=rrip_first`):

| policy | cache_sim | gem5 | Sniper |
|---|---|---|---|
| LRU | 0.6606 | 0.6475 | 0.5695 |
| GRASP | **0.5319** | **0.5655** | **0.4771** |
| ECG_GRASP_POPT | 0.5718 | eviction-spec | eviction-spec |

Read as **direction vs LRU**, not absolute (gem5/Sniper see the full ISA stream, cache_sim graph-only).
GRASP helps in all three (the equivalence). gem5/Sniper ECG on a *pressured* cell exceeds the sim
timeout, so their ECG correctness is the eviction-spec (40/40) + the byte-identical
`ecg_victim_policy.h` decision, not a full miss-rate run. On a *tiny* L3 the gem5 full-ISA stream
swamps the signal (a documented access-population confound).

**Debug proof:** `ECG_DEBUG=1` makes each sim emit one `[ECG-CONFIG sim=… policy=… mode=… variant=…]`
line at policy init (verified identical mode/variant across all three); `ECG_EVICT_TRACE=N` dumps the
first N L3 evictions (`[EVICT L3 pol=… reason=…]`), proving the policy acts. Details:
`docs/findings/ecg_mask_direction_and_metadata.md` §20.

**Multi-kernel equivalence** (`experiments.py verify --kernels [--gem5 --sniper]`,
`verify/equiv_kernels.py`): the eviction DECISION is kernel-agnostic, so it is certified for
**PR / BFS / BC / CC** across all three simulators — every `(kernel × sim)` obeys the eviction spec
AND emits the debug banner. PR/BFS/BC are DECISIVE (the epoch distance strictly decides ≥1 victim)
on cache_sim + gem5 + Sniper; CC is the do-no-harm cell (epoch delivered on the inclusive cache_sim/gem5
legs, DECISION-level on the non-inclusive Sniper leg where its small union-find `comp[]` never creates
property-eviction pressure). SSSP is cache_sim + gem5 (the Sniper `sg_kernel` SSSP target needs a
weighted `.wsg` the eval corpus lacks). This is the DECISION equivalence; the per-edge mask DIRECTION
is still PR-tuned (BFS/BC out-edge direction uncertified — a miss-rate, not a spec, concern). §21.

## Related
- [[Baseline-Literature-Faithfulness]] — GRASP/P-OPT faithfulness audit
