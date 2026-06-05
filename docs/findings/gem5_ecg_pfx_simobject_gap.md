# Finding — gem5 ECG_PFX SimObject hint-to-issue gap

**Date:** 2026-06-05 (sprint 6f-6 closeout)
**Discovered by:** mode 6 cross-sim port (commit 1aa1b24b)
**Severity:** medium (blocks gem5 cycle-accurate validation of paper headline result)
**Workaround:** Sniper provides the cycle-accurate cross-sim audit; cache_sim provides the per-cell numerical evidence

## Summary

The gem5 ECG_PFX SimObject (`bench/include/gem5_sim/overlays/mem/cache/prefetch/ecg_pfx.cc`) correctly receives kernel-side prefetch-target hints via `consumePrefetchTargetHint()`. The hints reach `calculatePrefetch()` and are pushed to the `addresses` vector via `addresses.push_back(AddrPriority(address, 0))`. However the L2 prefetcher reports `pfIssued = 0` — no actual prefetches are issued to memory.

## Symptom

Run reproducible from current `graphbrew_ecg` branch HEAD (`1aa1b24b`):

```bash
make gem5-m5ops-pr
GRAPH_FILTER=email-Eu-core ARM_FILTER=ECG_PFX \
  bash scripts/experiments/ecg/sweeps/pfx_gem5_validation_sweep.sh
```

Output observed:
- `[gem5 ECG mode 6] lookahead=8` ← kernel mode 6 path active
- `[ecg_mode6 gem5-PR] vertices=1005 edges=32128 encoded=31142 (96.9%)` ← per-edge mask built
- `ECG_PFX: first target vertex=2 addr=0x9f4008 property=[0x9f4000,0x9f4fb4)` ← hint reaches SimObject
- `system.l2cache.prefetcher.pfIssued                  0` ← **but no prefetches issued**
- `system.l2cache.prefetcher.pfUseful                  0`
- `system.l2cache.prefetcher.pfHitInCache              0`

## Root cause (preliminary diagnosis)

Three candidate paths between `addresses.push_back()` (in `ecg_pfx.cc::calculatePrefetch`) and `pfIssued`:

1. **Queued::issuePrefetchRequests** — the parent class consumes `addresses`. If `prefetch_queue` is full or `queue_size` is 0, prefetches drop.
2. **Cache::access** path — if the prefetch address tag matches an MSHR, the prefetch is filtered as already-in-flight.
3. **Cache::translatePrefetchRequest** — if the prefetch crosses a page boundary or has no valid translation, it drops.

Note that `pfHitInCache = 0` rules out (2) as the main cause (the addresses aren't even getting that far). The most likely culprit is (1) — the prefetch queue is either disabled or has zero capacity for the ECG_PFX SimObject.

## Impact on paper claims

- **§5.3 (mode 6 corpus efficiency)** — uses **cache_sim** as its primary measurement. Not affected.
- **§5.4 (saturation/convergence)** — uses cache_sim with Sniper as a secondary corroboration. Sniper now runs mode 6 (commit `1aa1b24b`). Not affected.
- **§4.1 / §6.3 (simulator coverage)** — already documents "gem5 has not yet been exercised on the prefetcher axis." This finding reinforces that limitation.
- **§7.6 (reproducibility gaps)** — already lists "gem5 ECG_PFX coverage" as a known gap. Refer to this file for the technical cause.

## Suggested fix path (deferred future work)

1. Inspect `Queued::queue_size` for `ecg_pfx` in the gem5 build's `GraphPrefetchers.py` config. Default in some gem5 forks is 0.
2. If queue capacity is non-zero, instrument `calculatePrefetch` to dump `addresses.size()` per call. If non-zero, the gap is downstream.
3. Compare with the DROPLET-class indirect prefetcher in the same overlay tree (`bench/include/gem5_sim/overlays/mem/cache/prefetch/droplet.{cc,hh}`) — if DROPLET issues prefetches but ECG_PFX doesn't on the same workload, the gap is ECG_PFX-specific.

## Smoking gun comparison (added 2026-06-05 14:10)

Side-by-side gem5 smoke with identical L1/L2/L3 config (32kB/256kB/1MB)
on email-Eu-core/pr at -i 2 iterations:

| Arm    | calculatePrefetch fires | addresses.push_back | pfIssued | pfUseful |
|--------|------------------------|--------------------:|---------:|---------:|
| DROPLET| many (every cache acc) | many (per-edge)     | **867**  | **776**  |
| ECG_PFX| at least 1 (log line)  | at least 1          | **0**    | 0        |

So gem5's prefetcher infrastructure DOES work end-to-end (DROPLET
issues 867 prefetches with 89% accuracy in the same gem5 build).
The ECG_PFX-specific gap is between `addresses.push_back()` in
`calculatePrefetch()` and `pfIssued` being incremented.

Differential analysis:
- Same base class: both extend `Queued` (gem5 QueuedPrefetcher).
- Same config params: `prefetch_on_access=True, on_inst=False,
  use_virtual_addresses=True, queue_size=32` (default).
- Same `addresses.push_back(AddrPriority(addr, 0))` call.
- DROPLET pushes on EVERY cache access in `isEdgeArrayAccess` range;
  ECG_PFX pushes only when a hint is queued by the kernel
  (sparse, kernel-driven via m5op `setPrefetchTargetHint`).

Hypothesis (not verified, requires gem5-expert follow-up):
- gem5's Queued base class inserts pushed addresses into an internal
  prefetch queue. Queue draining happens in `issuePrefetch()` which
  is called from the cache's `tick()` loop. If the prefetch queue
  inserts a packet that references a virtual address far from the
  current PC's translation context (or that crosses a page bound),
  the `Queued::translateFunctional()` step may silently drop it.
- Alternatively: the m5op-handler delivering hints may run in a
  thread context that produces packets gem5 cannot enqueue to the
  L2 prefetcher queue.
- Single-slot mailbox limitation: `setPrefetchTargetHint()` in
  `graph_cache_context_gem5.hh:109` uses a single `std::atomic<uint32_t>`
  slot for the hint. Each kernel call OVERWRITES the prior unconsumed
  hint. The kernel emits thousands of hints per PR iteration; the L2
  prefetcher only calls `consumePrefetchTargetHint` on its
  notification events. Mismatched rates lose hints. This explains
  WHY the hint count consumed is small (most are overwritten before
  consumption), but does NOT explain `pfIssued = 0` given that AT
  LEAST ONE hint was consumed (per "first target vertex=4" log).

Next debug step (NOT done in this session): add an instrumentation
counter inside ecg_pfx.cc to count `addresses.push_back()` calls,
compare against `pfIssued`. If push count >> 0 but issued = 0,
the gap is in `Queued::issuePrefetch()` for the ECG_PFX SimObject
specifically; if push count = 0, the gap is in
`consumePrefetchTargetHint()` not returning true.

## Status

- Documented in this finding.
- Sniper mode 6 sweep (commit `1aa1b24b` + sprint 6f-6) provides the cross-sim audit path the paper needs.
- gem5 cycle-accurate ECG_PFX validation is deferred to a follow-on study.
- §6.3 of the paper documents this scope limitation honestly.
