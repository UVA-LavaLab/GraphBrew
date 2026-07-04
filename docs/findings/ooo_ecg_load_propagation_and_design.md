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
(`base.cc:499,252,789`: `std::make_shared<Request>(*pkt->req)`). The `Extensible` copy
constructor **clones every extension**:
```cpp
// base/extensible.hh:124
Extensible(const Extensible& other) {
    for (auto& ext : other.extensions)
        extensions.emplace_back(ext->clone());   // EcgEpochExtension::clone() is defined
}
```
So the extension is preserved through every request copy → it reaches `handleFill` →
`allocateBlock(pkt)` → `insertBlock`/replacement `reset(pkt)`. **Propagation is sound** (the
#1 risk is retired without a rebuild).

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
1. **Demonstrate the race:** with the mailbox only, log expected-vertex vs mailbox-vertex
   mismatch count > 0 (proves in-order-only delivery fails under OoO).
2. **Prove the fix:** ECG-O3 with the extension should match ECG-timing (hint survives OoO);
   without it, ECG-O3 degrades toward LRU. This A/B is also the paper figure that MOTIVATES
   the instruction (naive hint delivery races on real cores).

## Verdict
Approach is viable (propagation proven). Do NOT hand-wire the ISA/core blindly this session
(rebuild + O3 both too slow to get a same-session signal). Next concrete step: implement the
per-request delivery (option: `initiateMemRead` extension overload) + consumer `dest` guard +
line-min epoch in the record, then the micro-kernel O3 A/B. Enabler already landed:
`roi_matrix --gem5-cpu-type O3`.
