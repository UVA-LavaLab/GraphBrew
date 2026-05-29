# Sniper anchor — capacity-sensitivity slope replay

**Verdict:** PASS  
**Cells (app, graph):** 6  
**(cell, policy) records:** 18

## Method

log2(kB) axis: 4kB → 2.0, 32kB → 5.0, 256kB → 8.0, 2MB → 11.0.
OLS slope of miss_rate (pp) versus log2(L3 kB) across the four anchor sizes per (app, graph, policy). Anchor sizes are smaller than the cache-sim sweep, so absolute slope magnitudes are not comparable to gates 66/67/68 — only the ordering is.

## Per-policy median slope (pp / octave)

| policy | n | median | mean |
|---|---:|---:|---:|
| GRASP | 6 | -7.6445 | -6.8227 |
| LRU | 6 | -7.4015 | -6.9539 |
| SRRIP | 6 | -7.9640 | -7.0059 |

**median(LRU)  − median(GRASP) =** +0.2430 pp/octave (INFORMATIONAL; sub-WSS regime can invert this — not gated)
**median(SRRIP) − median(GRASP) =** -0.3195 pp/octave (want <= 0)
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
| bfs | cit-Patents | GRASP | -4.7671 | 94.283 | 46.568 | 46.779 | 46.541 |
| bfs | cit-Patents | LRU | -4.9317 | 96.076 | 46.632 | 46.444 | 46.822 |
| bfs | cit-Patents | SRRIP | -4.9075 | 95.368 | 46.042 | 45.343 | 46.526 |
| bfs | email-Eu-core | GRASP | -4.9111 | 95.053 | 47.579 | 46.242 | 46.387 |
| bfs | email-Eu-core | LRU | -5.0706 | 96.832 | 46.090 | 46.723 | 45.915 |
| bfs | email-Eu-core | SRRIP | -4.9120 | 95.772 | 46.306 | 46.387 | 46.624 |
| pr | cit-Patents | GRASP | -7.6747 | 88.649 | 11.979 | 11.765 | 11.973 |
| pr | cit-Patents | LRU | -6.8004 | 80.142 | 11.957 | 12.083 | 12.096 |
| pr | cit-Patents | SRRIP | -7.9302 | 91.528 | 11.443 | 11.844 | 12.092 |
| pr | email-Eu-core | GRASP | -7.6143 | 87.931 | 12.394 | 12.092 | 11.889 |
| pr | email-Eu-core | LRU | -8.0027 | 91.690 | 12.000 | 11.765 | 11.741 |
| pr | email-Eu-core | SRRIP | -7.9979 | 91.495 | 11.973 | 12.192 | 11.443 |
| sssp | cit-Patents | GRASP | -7.8399 | 86.421 | 8.258 | 8.050 | 8.091 |
| sssp | cit-Patents | LRU | -8.4675 | 92.778 | 8.158 | 8.054 | 8.137 |
| sssp | cit-Patents | SRRIP | -8.1147 | 89.109 | 8.086 | 8.036 | 7.979 |
| sssp | email-Eu-core | GRASP | -8.1293 | 89.302 | 8.077 | 8.028 | 8.025 |
| sssp | email-Eu-core | LRU | -8.4508 | 92.643 | 8.014 | 8.057 | 8.120 |
| sssp | email-Eu-core | SRRIP | -8.1733 | 89.721 | 7.823 | 7.860 | 7.976 |
