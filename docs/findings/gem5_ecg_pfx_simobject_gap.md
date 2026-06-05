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

## Status

- Documented in this finding.
- Sniper mode 6 sweep (commit `1aa1b24b` + sprint 6f-6) provides the cross-sim audit path the paper needs.
- gem5 cycle-accurate ECG_PFX validation is deferred to a follow-on hardware-synthesis study (already framed this way in §6.3).
