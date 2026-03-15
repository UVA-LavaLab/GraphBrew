# SRRIP → gem5 Integration

## Overview

SRRIP (Static Re-Reference Interval Prediction) is the baseline RRIP policy that
forms the foundation for GRASP and ECG. **No new gem5 code is needed** — gem5's
built-in BRRIP (Bimodal RRIP) with `btp=0` is functionally equivalent to SRRIP.

## Mapping

| SRRIP Concept | Standalone (cache_sim.h) | gem5 Equivalent |
|---------------|--------------------------|-----------------|
| RRPV width | 3-bit (0-7) | `BRRIPRP::numRRPVBits` |
| Insert RRPV | 2 (long re-reference) | `btp=0` forces all inserts at M-1 |
| Hit RRPV | 0 (near-future) | BRRIP default touch behavior |
| Eviction | Find RRPV=3, age all | BRRIP `getVictim()` |

## gem5 Configuration

```python
from m5.objects import BRRIPRP

# SRRIP = BRRIP with bimodal throttle parameter = 0
# (never insert at M, always at M-1 → pure SRRIP)
l3_replacement = BRRIPRP(btp=0)
```

## Standalone Implementation Reference

From `bench/include/cache_sim/cache_sim.h`:

```cpp
size_t findVictimSRRIP(std::vector<CacheLine>& set) {
    while (true) {
        for (size_t i = 0; i < associativity_; i++) {
            if (set[i].rrpv == 3) return i;  // Found victim at distant re-ref
        }
        for (size_t i = 0; i < associativity_; i++) {
            if (set[i].rrpv < 3) set[i].rrpv++;  // Age all lines
        }
    }
}
```

## DRRIP Extension

gem5's BRRIP handles both SRRIP and DRRIP (Dynamic RRIP).  For pure SRRIP, use:
- `btp=0` → Always insert at RRPV = M-1 (long re-reference prediction)

For DRRIP (set-dueling between SRRIP and BRRIP), use gem5's default BRRIP
parameters which include set sampling.

## Validation

SRRIP is used as baseline for validating GRASP/ECG invariants:
1. GRASP with DBG ordering must beat SRRIP (strictly better miss rate)
2. ECG(DBG_ONLY) ≈ GRASP ≈ SRRIP + degree-aware insertion
3. All graph-aware policies must be ≥ SRRIP (never worse)
