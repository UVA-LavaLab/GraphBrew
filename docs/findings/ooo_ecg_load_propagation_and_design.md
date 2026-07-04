# OoO ecg.load: propagation proof + de-risked design (rubber-duck reviewed)

**Date:** 2026-07-03
**Prompted by:** "we have an aim to add the inst to OoO pipeline" → "rubber duck and proceed."
**Method:** source audit of the gem5 request/MSHR/fill path (no rebuild) + rubber-duck review.
This is the **proof phase** the review endorsed BEFORE investing in the ISA/core wiring.

## The problem
`ecg.load` must deliver a per-vertex next-ref epoch to the L3 replacement policy, bound to
the CORRECT demand load. The two existing mechanisms break under OoO:
- **Single-slot mailbox** (`setDecodedEcgExtractHint`): global "last ecg.load's epoch". Correct
  ONLY in-order (load of property[v] immediately follows ecg.load(v)). Under OoO, interleaved
  in-flight loads race → last-writer-wins → a load sees the wrong vertex's epoch; the vertex
  guard turns this into an unstamped fill → epoch signal **degrades toward LRU**.
- **4K per-vertex table**: capacity collisions corrupt epochs; also the O(V) storage ECG avoids.

## The designed fix
`EcgEpochExtension : gem5::Request::Extension` (ecg_epoch_request_ext.hh) — tag the ecg.load's
OWN demand Request with `{dest, epoch}` so the hint rides the specific in-flight request
(race-free, no shared state). Consumer `readEcgEpoch(pkt->req,…)` is already wired FIRST in
`ecg_rp.cc:248` (falls back to mailbox). Only PRODUCTION is missing: `attachEcgEpoch` has zero
callers.

## PROVEN (source audit): the extension propagates CPU → L1 → L2 → L3
The load-bearing question was whether a `Request::Extension` survives to the fill packet the L3
replacement policy sees. gem5 copy-constructs requests along the miss/fill path
(`base.cc:499,252,789`: `std::make_shared<Request>(*pkt->req)`), AND the outgoing miss packet is
built with the SAME `RequestPtr` as the first demand target
(`cache.cc createMissPacket`: `new Packet(cpu_pkt->req, cmd, blkSize)`; `sendMSHRQueuePacket`
uses `mshr->getTarget()->pkt`). The `Extensible` copy constructor **clones every extension**:
```cpp
// base/extensible.hh:124
Extensible(const Extensible& other) {
    for (auto& ext : other.extensions)
        extensions.emplace_back(ext->clone());   // EcgEpochExtension::clone() is defined
}
```
So the extension is preserved (same object or clone) → it reaches `handleFill` →
`allocateBlock(pkt)` → `insertBlock`/replacement `reset(pkt)`.

## CONFIRMED (empirical, gem5 micro-kernel): arrived = 100%, matched = 100%
A synthetic harness (env `GEM5_ECG_SYNTH_PROP=1`, inert otherwise) stood in for the ecg.load
producer: it tagged every read demand at cache entry (`BaseCache::recvTimingReq`) with an
address-derived `EcgEpochExtension`, and the L3 replacement `reset()` verified arrival + value.
Result on gem5 PR (‑g12, L3=8kB to force fills, TimingSimpleCPU):
```
[ECG SYNTH-PROP] fills=4000 arrived=4000 (100.0%) matched=4000 (100.0%)
```
- **arrived=100%** — the extension set at L1 entry reaches the L3 fill on EVERY property fill:
  propagation through L1→L2→L3 + MSHR/coalescing is confirmed end-to-end.
- **matched=100%** — the arrived epoch always corresponds to a word WITHIN the filled cache
  line: no corruption, correct block association, coalescing picks a word in the line.

**The #1 load-bearing risk (does the extension propagate + survive coalescing) is retired both
by source audit AND empirically.** (Harness reproduction: 1 include + a `maybeSynthAttachEcgEpoch`
call at the top of `BaseCache::recvTimingReq`, plus a `ecgSynthVerify` call at the `readEcgEpoch`
site in `ecg_rp.cc reset()`; helpers live in `ecg_epoch_request_ext.hh`. Kept out of the tree —
it is a local gem5-tree diagnostic, env-gated. Note: attach and verify must use the SAME address
basis — vaddr-preferring — or the value check spuriously fails on the vaddr/paddr difference.)

## OPEN: MSHR coalescing semantics (a design decision, not a blocker)
`recvTimingResp` uses `initial_tgt = mshr->getTarget()` (the FIRST demand that opened the MSHR),
and the outgoing miss request is built from that first target. So if `ecg.load(v1)` and
`ecg.load(v2)` (same cache line) coalesce, the fill carries **v1's** epoch (first-touched),
not v2's. Consequences:
- The epoch is NOT lost (good) — it degrades to "first vertex on the line to miss."
- Because a line holds 16 vertices, v1 and v2 are on the SAME line; stamping with the first
  toucher's epoch is a defensible line-level policy, but is NOT the `ECG_EDGE_MASK_LINEMIN`
  (per-line-min) epoch cache_sim uses. **Decision:** deliver the line-min epoch in the ecg.load
  record (precompute per line, as cache_sim already does), so whichever vertex opens the MSHR
  carries the same line-min value → coalescing becomes order-independent and matches cache_sim.

## CORRECTED design (rubber-duck fixes)
1. **Production mechanism — NOT a StaticInst member.** `{dest, epoch}` are dynamic (from `Rs2`);
   `StaticInst` is shared across dynamic executions → storing them there recreates the race.
   Use per-dynamic-request delivery: pass the extension into the request-creation path (an
   `initiateMemRead` overload / decorator), OR a tightly-scoped, ASSERTED staging value set
   immediately before issue and cleared immediately after (bridge only). Avoid a dedicated
   hand-rolled `initiateAcc` (replicating translation/splits/faults is a correctness swamp).
2. **Consumer must validate `dest`.** `readEcgEpoch` currently returns epoch only; expose
   `dest()` and have `ecg_rp` accept the epoch only if `dest` maps into the filled line
   (the mailbox path had a vertex guard; the extension path must too).
3. **Speculation:** the current `ea_code` writes global mailbox/table state before the access
   completes → squashed speculative ecg.loads perturb global state under O3. Request-local
   delivery fixes this; do not trust the mailbox/table writes for OoO.

## Constraint that shapes the whole effort
**DerivO3CPU is prohibitively slow** (measured: 1024-vtx PR ‑i1 did NOT finish the ROI in
~5 min; timing does the full run in ~100s; 4096 vtx timed out at 550s). So O3 validation is
only feasible on MICRO kernels (~64–256 vertices), and the RISC-V rebuild loop is ~20–40 min.
gem5 stays a small-graph MECHANISM/correctness artifact; the scale story is cache_sim
(validated <0.1pp vs gem5 on web-Google @2MB) + Sniper.

## Validation plan (when the wiring is done)
On a 64–256-vertex micro-kernel, O3:
1. **Demonstrate the race:** DONE ahead of the producer (see the MEASURED section) — the mailbox-only
   O3 mismatch already exceeds in-order (real but ~3% of source loads at gem5 scale; understated by
   edge-serialization + confounded by destination fills).
2. **Prove the fix:** ECG-O3 with the extension should recover those raced source loads (O3 source-match
   → ~in-order). Because the gem5 gap is modest, report the **guarded-match delta** (raced source loads
   eliminated) + the correctness argument, NOT a headline miss-rate figure. Set expectations: the A/B is
   a *correctness* figure (naive shared-mailbox delivery races on real cores; the request-local extension
   does not), not a large Δmiss-rate — that story lives in cache_sim/Sniper scale + the traversal win.

## MEASURED #1 (PR kernel — CONFOUNDED, SUPERSEDED by the clean synthetic gather below)
> NOTE: the "small race" reading here is a PR-kernel ARTIFACT (destination-fill confound +
> edge-record serialization). The clean synthetic-gather measurement (next section) isolates the
> race and finds it is **~100%** under O3. Read this section only for the confound analysis.

Rubber-duck RE-SCOPE verdict (2026-07-04): before wiring the producer, prove the O3 mailbox race
is real and material. Added an env-gated (`GEM5_ECG_RACE_COUNT=1`) counter at the ecg_rp fill
site: at each property-data L3 fill, compare the filled vertex `v_fill` (addr→vertex) to the
single-slot mailbox's last-decoded vertex; split demand vs prefetch fills. Ran the SAME config on
DerivO3CPU vs TimingSimpleCPU. Config: PR `-g8` (256 vtx) `-k8 -i2`, L1d=L2=L3=512B (force property
thrash), **DDR4_2400 memory (realistic ~100+ cyc latency), DerivO3CPU default window (192 ROB /
32 LQ)**, ECG:ECG_GRASP_POPT mode-6.

| CPU | mbvalid | match | MISMATCH | mismatch-rate |
|-----|--------:|------:|---------:|--------------:|
| TimingSimpleCPU (in-order) | 1984 | 1221 | 763 | 38.5% |
| DerivO3CPU (OoO)           | 1984 | 1183 | **801** | **40.4%** |

- **Race is REAL and directional:** O3 mismatch > in-order at every checkpoint (fills 1000/1500/2000;
  delta grew 8→38). O3 has **38 fewer matches** → ~**3.1% of the ~1221 source (ecg-stamped) fills are
  raced** under O3 (the mailbox was overwritten by a later ecg.load before the earlier fill landed).
- **The ~38% baseline is a STRUCTURAL confound, NOT a race:** PR's destination `score[u]` writes are
  property-classified but are *not* preceded by an `ecg.load` for `u`, so the mailbox (holding a
  neighbor `v`) mismatches `v_fill=u`. This is present IN-ORDER too and is correctly rejected by the
  production vertex-guard (`lookupDecodedEcgHint` returns false → falls to degree-based dbg). `pf=0`
  (no prefetch fills in this config), so the baseline is purely non-ecg-stamped property fills.
- **Why the race is SMALL at gem5 scale (mechanism):** the fused `ecg.load(v)` computes its address
  from `Rs2` (the fat edge record loaded by a *prior* instruction). With thrashing caches the
  edge-record loads MISS, so each `ecg.load` waits on its record → consecutive `ecg.load`s
  **serialize**, limiting the in-flight overlap that overwrites the single-slot mailbox. A real core
  (sequential hardware-prefetched edge stream + deep window + DRAM latency) would overlap far more →
  a much larger race. gem5 micro-kernels **cannot** stress the "edges cached / property missing"
  regime with a shared cache and a CSR where edges ≥ vtx, so they **structurally understate** the race.

**Decision-relevant conclusion (CONFOUNDED — see clean result below):** in the PR kernel the O3-vs-
in-order gap is only ~3% of source loads, but this is an ARTIFACT: (a) ~38% of "mismatches" are
non-ecg-stamped destination `score[u]` fills (present in-order too), and (b) the fused ecg.load's
address depends on `Rs2` (edge record from a prior load), so thrashing caches SERIALIZE consecutive
ecg.loads and throttle the O3 overlap. Both confounds vanish in the clean synthetic gather below.

## MEASURED #2 (clean synthetic gather — DECISIVE): the isolated O3 race is ~100%
To remove both confounds, added an env-gated (`GEM5_ECG_SYNTH_GATHER=1`, guest env forwarded via
graph_se.py) synthetic gather in pr.cc (runs right after the property regions are exported, before the
O(N) mask build, and `return`s): a tight loop of `gem5_ecg_load_evict(contrib, fat, wc=3)` where
`fat = dest | epoch<<32` and `dest = (i * 2654435761) % N` is a **register-computed multiplicative
hash** — NO destination writes (kills confound a) and NO edge-record load (kills confound b, so the
ecg.loads are independent and issue back-to-back). Run on the **faithful RISC-V `ecg.load` path**
(build/RISCV/gem5.opt + `pr_riscv_m5ops`, `GEM5_KERNEL_SUFFIX=_riscv_m5ops`) with `contrib` sized ≫ the
tiny L3 so every hashed access misses. `pf=0`, `ext=0` (no producer yet — mailbox-only delivery).

| CPU | fills | match | MISMATCH | mismatch-rate |
|-----|------:|------:|---------:|--------------:|
| TimingSimpleCPU (in-order), -g12 iters=50000 | 46500 | 46500 | 0 | **0.000%** |
| TimingSimpleCPU (in-order), -g11 iters=40000 | 38500 | 38499 | 1 | **0.003%** |
| **DerivO3CPU (OoO), -g11 iters=40000** | 38000 | **1** | **37999** | **99.997%** |

**The single-slot mailbox delivery is CATASTROPHIC under O3 for independent ecg.loads: ~100% of source
loads get the WRONG epoch** (the mailbox is overwritten by later in-flight ecg.loads before each fill
lands — with a ~32-deep load queue and no address dependency, essentially every fill is stale). In-order
it is PERFECT (0%). This is the un-confounded truth the PR kernel hid.

**Why the PR kernel looked benign (~3%) but the truth is ~100%:** PR's ecg.load address depends on the
edge record, so a real gather's ecg.loads partly serialize (edge-record misses gate them) → limited
overlap → the mailbox is only sometimes stale. The synthetic gather removes that dependency, exposing
the worst case a deep OoO core reaches when the address stream is independent/prefetched. Real
aggressive cores (sequential HW-prefetched edge streams, deep windows, multicore) approach the
synthetic regime, not the throttled PR one.

**Decision-relevant conclusion (CORRECTED):** the single-slot mailbox is an in-order-only model; under
O3 it degrades to ~LRU (≈100% wrong epoch). The request-extension producer is therefore not just
"architecturally cleaner" — it is REQUIRED for correct O3/multicore delivery, and the A/B is a clean,
dramatic figure: mailbox-only O3 ≈100% wrong → extension O3 ≈0% wrong (race-free, request-local). This
strongly motivates wiring the producer. (Throwaway diagnostics: race counter in gem5/src ecg_rp.cc
[gitignored], synthetic branch in pr.cc + env passthrough in graph_se.py [tracked — reverted after the
producer A/B]. RISC-V `ecg.load` path used, not the X86 `workbegin` m5op which serializes.)

## Verdict
Approach is viable and de-risked: propagation is proven by source audit **and confirmed
empirically** (arrived=100%, matched=100%); the O3 race is **confirmed CATASTROPHIC when isolated**
(clean synthetic gather: in-order 0% wrong vs O3 ~100% wrong epoch — see MEASURED #2). The single-slot
mailbox is an in-order-only model; under O3 it degrades to ~LRU. The producer is therefore REQUIRED,
and its A/B is a clean, dramatic figure. The remaining work is the PRODUCER — bind the extension to the
ecg.load's demand request:
- **Producer mechanism:** the O3 `LSQRequest` holds the per-dynamic `DynInstPtr _inst`
  (`cpu/o3/lsq.hh:247`, `instruction()` at 334) — the OoO-safe carrier (NOT StaticInst). ea_code
  stores `{dest, epoch}` on per-dynamic-instruction state via an ExecContext hook; the LSQ
  request setup reads it and calls `attachEcgEpoch(req, dest, line_min_epoch)`.
- **Consumer:** add the `dest` guard to `readEcgEpoch`/`ecg_rp` (accept the epoch only if `dest`
  is within the filled line) — the harness confirmed the value is a within-line word, so the
  guard is a cheap correctness assertion.
- **Record:** deliver the per-LINE-min epoch (cache_sim already computes `ECG_EDGE_MASK_LINEMIN`)
  so coalescing (first-target wins) is order-independent.
- **Validate:** micro-kernel (64–256 vtx) O3 A/B — mailbox-only shows expected-vs-delivered
  mismatch > 0 (the race); extension shows ECG-O3 == ECG-timing (hint survives OoO). This A/B is
  the paper figure that MOTIVATES the instruction.

Enablers landed: `roi_matrix --gem5-cpu-type {timing,O3,minor}`; incremental gem5 rebuild
measured at ~80s (cache objects), so the producer iterate-loop is faster than feared (the O3
RUN, not the build, is the slow part → keep validation to micro-kernels).
