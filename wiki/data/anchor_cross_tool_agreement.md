# Cross-tool shared-anchor slope-sign agreement

**Verdict:** PASS (shared=3, sign_agree=3/3)

## Shared cells (gem5 ∩ sniper)

| graph | app | policy | gem5_slope | sniper_slope | sign_match | both_neg | sniper_steeper | |Δ| |
|---|---|---|---:|---:|:---:|:---:|:---:|---:|
| email-Eu-core | pr | GRASP | -5.0573 | -7.6143 | OK | OK | OK | 2.5570 |
| email-Eu-core | pr | LRU | -2.8766 | -8.0027 | OK | OK | OK | 5.1261 |
| email-Eu-core | pr | SRRIP | -6.1341 | -7.9979 | OK | OK | OK | 1.8638 |

## Checks

| check | ok |
|---|:---:|
| shared_floor | OK |
| sign_agreement | OK |
| both_negative | OK |
| sniper_steeper | OK |
| abs_diff_ceiling | OK |

## Thresholds (locked)

- shared_cells_floor = 3
- sign_agreement_floor = 1.0
- sniper_steeper_floor = 1.0
- max_abs_slope_diff_pp = 8.0
