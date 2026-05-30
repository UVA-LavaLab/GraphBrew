# Literature-faithfulness sign-mass audit

For each (expected_sign × policy) bucket: how often the observed delta_pct has the literature's claimed sign, with binomial sign-test p-value and Wilson 95 % lower bound on the correctly-signed fraction.

## Summary

- Total claims (with computable delta_pct): **330**
- ok-status rows across all buckets: **298**
- Buckets (sign × policy): **6**
- Buckets with at least one ok row: **6**

## Per-bucket sign-mass

Sign-classes: correct (delta has the expected sign), tie (delta == 0 or expected_sign is `~`), wrong (delta has the opposite sign). The binomial p-value uses only strict-sign counts (ties excluded).

| sign | policy | n_ok | correct | ties | wrong | frac | wilson 95 LB | binom p | median Δpp | mean Δpp |
|---|---|---|---|---|---|---|---|---|---|---|
| - | GRASP | 13 | 13 | 0 | 0 | 1.000 | 0.772 | 0.0002 | -4.8254 | -6.7092 |
| - | POPT | 7 | 7 | 0 | 0 | 1.000 | 0.646 | 0.0156 | -10.7415 | -10.3969 |
| - | POPT_GE_GRASP | 88 | 55 | 2 | 31 | 0.636 | 0.532 | 0.0127 | -0.3346 | -3.2296 |
| ~ | GRASP | 5 | 0 | 5 | 0 | 0.500 | 0.118 | — | -1.2533 | -1.3515 |
| ~ | POPT_NEAR_GRASP_IF_BIG_GAP | 110 | 0 | 110 | 0 | 0.500 | 0.408 | — | -0.0001 | -2.0141 |
| ~ | SRRIP | 75 | 0 | 75 | 0 | 0.500 | 0.396 | — | -1.5592 | -1.9685 |

## Interpretation

* `-` × GRASP / `-` × POPT — these are the *load-bearing* sign claims (the policy beats LRU). They should show strongly negative median delta_pct and a binomial sign-test p-value below 0.05 (often well below). The Wilson 95 % lower bound floor is sample-size dependent: 13 cells caps the lower bound around 0.77; 7 cells around 0.65.
* `-` × POPT_GE_GRASP — claim that POPT ≥ GRASP (delta is POPT − GRASP). Many cells legitimately tie at 0 (both policies saturate at the same value); the gate treats ties as half-credit. We still expect Wilson LB ≥ 0.50 (better than coin-flip).
* `~` × SRRIP / `~` × POPT_NEAR_GRASP_IF_BIG_GAP — magnitude claims; sign is not asserted and these rows are not locked by the LIT-Sig gate (only reported for reference).
