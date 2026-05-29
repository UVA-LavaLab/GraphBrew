# Cross-tool slope-sign universality

**Verdict:** PASS  
**Slope band:** [-25.0, -0.5] pp/oct  
**Steepness span ceiling:** 5.0 pp/oct  
**(tool, policy) medians in band:** 10/10  

## Median slope per (tool, policy)

| tool | policy | median pp/oct |
|---|---|---:|
| cache-sim | GRASP | -14.6513 |
| cache-sim | LRU | -15.6181 |
| cache-sim | POPT | -14.7621 |
| cache-sim | SRRIP | -15.5966 |
| gem5 | GRASP | -5.9607 |
| gem5 | LRU | -5.1229 |
| gem5 | SRRIP | -7.2113 |
| sniper | GRASP | -7.6445 |
| sniper | LRU | -7.4015 |
| sniper | SRRIP | -7.9640 |

## Per-tool steepness span (max - min across policies)

| tool | span pp/oct |
|---|---:|
| cache-sim | 0.9668 |
| gem5 | 2.0884 |
| sniper | 0.5625 |

## Verdict checks

| check | result |
|---|---|
| all_tool_policy_medians_negative | ✅ |
| all_tool_policy_medians_in_band | ✅ |
| no_tool_exceeds_steepness_span_ceiling | ✅ |

## Violations

_None._

## Interpretation

This is a roll-up invariant ensuring no (tool, policy) slope median ever turns positive (extra cache must not hurt on average) or collapses to near-zero (a policy that stops responding to cache scaling is suspect). The cross-tool span check catches partial regressions where one policy on one tool stops scaling while siblings stay healthy.
