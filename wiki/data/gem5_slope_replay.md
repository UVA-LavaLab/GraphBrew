# Gem5 anchor — capacity-sensitivity slope replay

**Verdict:** PASS  
**Cells (app, graph):** 2  
**(cell, policy) records:** 6

## Method

log2(kB) axis: 4kB → 2.0, 32kB → 5.0, 256kB → 8.0, 2MB → 11.0.
OLS slope of miss_rate (pp) versus log2(L3 kB) across the four anchor sizes per (app, graph, policy). Anchor sizes are smaller than the cache-sim sweep, so absolute slope magnitudes are not comparable to gates 66/67/68 — only the ordering is.

## Per-policy median slope (pp / octave)

| policy | n | median | mean |
|---|---:|---:|---:|
| GRASP | 2 | -5.9607 | -5.9607 |
| LRU | 2 | -5.1229 | -5.1229 |
| SRRIP | 2 | -7.2113 | -7.2113 |

**median(LRU)  − median(GRASP) =** +0.8378 pp/octave (INFORMATIONAL; sub-WSS regime can invert this — not gated)
**median(SRRIP) − median(GRASP) =** -1.2506 pp/octave (want <= 0)
**help-floor for median(GRASP):** -1.0 pp/octave (want median(GRASP) below it)
**cache-monotonicity violations:** 0 (want 0)

## Verdict checks

| check | result |
|---|---|
| cache_monotonic_every_cell | ✅ |
| all_per_policy_medians_negative | ✅ |
| srrip_at_least_as_steep_as_grasp | ✅ |
| grasp_below_help_floor | ✅ |

## Per-cell slopes

| app | graph | policy | slope (pp/oct) | miss@4kB | miss@32kB | miss@256kB | miss@2MB |
|---|---|---|---:|---:|---:|---:|---:|
| bc | email-Eu-core | GRASP | -6.8640 | 65.951 | 11.150 | 2.530 | 0.184 |
| bc | email-Eu-core | LRU | -7.3692 | 71.155 | 11.039 | 2.859 | 0.190 |
| bc | email-Eu-core | SRRIP | -8.2885 | 80.259 | 10.941 | 2.436 | 0.209 |
| pr | email-Eu-core | GRASP | -5.0573 | 49.970 | 14.447 | 11.056 | 0.527 |
| pr | email-Eu-core | LRU | -2.8766 | 29.156 | 13.521 | 13.198 | 0.498 |
| pr | email-Eu-core | SRRIP | -6.1341 | 59.963 | 15.156 | 9.402 | 0.539 |
