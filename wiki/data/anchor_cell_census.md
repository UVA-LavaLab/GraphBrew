# Anchor cell-pair census

**Verdict:** PASS  
**Expected L3 axis:** 4kB, 32kB, 256kB, 2MB  
**Expected policies:** GRASP, LRU, SRRIP  
**Shared L3 axis:** ✅  
**Shared policy set:** ✅  
**Shared (graph, app) cells:** 1  

## Anchor coverage

| anchor | n_cells | min | records | l3_axis | policies |
|---|---:|---:|---:|---|---|
| gem5   | 2 | 2 | 6 | 4kB, 32kB, 256kB, 2MB | GRASP, LRU, SRRIP |
| sniper | 6 | 6 | 18 | 4kB, 32kB, 256kB, 2MB | GRASP, LRU, SRRIP |

## gem5 cells

| graph | app |
|---|---|
| email-Eu-core | bc |
| email-Eu-core | pr |

## Sniper cells

| graph | app |
|---|---|
| cit-Patents | bfs |
| cit-Patents | pr |
| cit-Patents | sssp |
| email-Eu-core | bfs |
| email-Eu-core | pr |
| email-Eu-core | sssp |

## Shared cells (both anchors)

| graph | app |
|---|---|
| email-Eu-core | pr |

## Verdict checks

| check | result |
|---|---|
| gem5_cell_count_at_or_above_baseline | ✅ |
| sniper_cell_count_at_or_above_baseline | ✅ |
| gem5_has_expected_cells | ✅ |
| sniper_has_expected_cells | ✅ |
| gem5_l3_axis_matches | ✅ |
| sniper_l3_axis_matches | ✅ |
| gem5_policy_set_matches | ✅ |
| sniper_policy_set_matches | ✅ |
| anchors_share_l3_axis | ✅ |
| anchors_share_policy_set | ✅ |
| anchors_share_at_least_one_cell | ✅ |
| gem5_cell_policy_records_match | ✅ |
| sniper_cell_policy_records_match | ✅ |

## Interpretation

Pins gem5/Sniper anchor coverage so that downstream cross-tool gates (70, 71, 72, 74, 76) cannot silently lose explanatory power if anchor sweeps shrink. Shared L3 axis and shared policy set guarantee apples-to-apples cross-tool slope comparisons; at least one shared (graph, app) cell ensures per-cell parity spot-checks have a foothold.
