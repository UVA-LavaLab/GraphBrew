# L3-sweep monotonicity universality (cache-sim)

**Verdict:** PASS  
**Source:** `wiki/data/oracle_gap.json`  
**L3 axis:** 4kB, 16kB, 64kB, 256kB, 1MB, 4MB, 8MB  
**Cells (>=2 L3 points):** 136  
**Total steps:** 320  
**Bumps:** 0 (0.00%) ceiling 10%  
**Largest bump:** 0.0000 pp (noise tolerance 0.50 pp)  
**Hard violations (>= 0.5 pp):** 0

## Verdict checks

| check | result |
|---|---|
| no_hard_violations | ✅ |
| bump_pct_under_ceiling | ✅ |
| largest_bump_within_noise | ✅ |

## Largest bump (worst-case cell)

_None — every step is monotone non-increasing._

## All bumps (sorted by magnitude)

_None._

## Interpretation

Cache monotonicity (more cache cannot hurt) is a foundational soundness check that downstream slope/distance/sensitivity gates rely on. Small noise-level bumps (<0.5 pp) are expected from sampling and warmup; any larger bump would indicate a simulator bug, corrupted sweep, or pathological policy.
