# Phase 5 — Ablation Toggles

**Commit:** `c28b9c9`  
**Date:** 2025-07-19

## Overview

Runtime ablation toggles let us isolate the contribution of each
AdaptiveOrder component without recompiling.  All toggles use
**environment variables** read once at first use (singleton
`AblationConfig::Get()`).  When verbose is on, the active ablations are
printed before community processing begins.

## Environment Variables

| Variable | Values | Effect |
|---|---|---|
| `ADAPTIVE_NO_TYPES`   | `1` | Force `type_0` with distance 0 for every community — disables the type system |
| `ADAPTIVE_NO_OOD`     | `1` | Skip the out-of-distribution guardrail; always use the perceptron's top pick |
| `ADAPTIVE_NO_MARGIN`  | `1` | Skip the low-margin fallback to ORIGINAL; always apply the chosen reordering |
| `ADAPTIVE_NO_LEIDEN`  | `1` | Skip Leiden entirely; assign every node to community 0 (one giant community) |
| `ADAPTIVE_FORCE_ALGO` | `0`–`15` | Bypass perceptron; use this algorithm ID for every large community |
| `ADAPTIVE_ZERO_FEATURES` | comma list | Zero the weights of named feature groups before scoring |

### `ADAPTIVE_ZERO_FEATURES` tokens

| Token | Zeroed weights |
|---|---|
| `packing` | `w_packing_factor` |
| `fef`     | `w_forward_edge_fraction` + `w_fef_convergence` |
| `wsr`     | `w_working_set_ratio` |
| `quadratic` | `w_fef_packing`, `w_packing_wsr`, `w_fef_wsr` |

## Wiring Points

| Toggle | Function | File | Effect |
|---|---|---|---|
| `zero_*` features | `scoreBase()` | `reorder_types.h` | Zeroes the corresponding weight before multiply |
| `zero_fef` | `score()` | `reorder_types.h` | Zeroes `w_fef_convergence` bonus |
| `no_types` | `FindBestTypeWithDistance()` | `reorder_types.h` | Returns `"type_0"` immediately |
| `no_ood` | `SelectReorderingWithMode()` | `reorder_types.h` | Skips OOD distance check |
| `no_margin` | `SelectReorderingFromWeights()` | `reorder_types.h` | Skips low-margin ORIGINAL fallback |
| `force_algo` | `SelectBestReorderingForCommunity()` | `reorder_types.h` | Returns forced algo before scoring |
| `no_leiden` | `GenerateAdaptiveMappingRecursiveStandalone()` | `reorder_adaptive.h` | Assigns all nodes to community 0 |

## Smoke Test Results (RMAT-16)

```
# Default — no ablation
./bench/bin/bfs -f rmat_16.sg -a 14 -v 2
→ normal community selections (HubClusterDBG for small group, GOrder for large communities)

# No Leiden — single community
ADAPTIVE_NO_LEIDEN=1 ./bench/bin/bfs -f rmat_16.sg -a 14 -v 2
→ "ABLATION: Leiden skipped, single community"
→ Community 0: 65536 nodes → GOrder

# Force algorithm 5 (DBG)
ADAPTIVE_FORCE_ALGO=5 ./bench/bin/bfs -f rmat_16.sg -a 14 -v 2
→ All large communities → DBG

# Zero packing + fef features
ADAPTIVE_ZERO_FEATURES=packing,fef ./bench/bin/bfs -f rmat_16.sg -a 14 -v 2
→ Prints zeroed features, community selections unchanged (low weight currently)
```

## Note: Small-Group Selection

`ADAPTIVE_FORCE_ALGO` bypasses `SelectBestReorderingForCommunity()` but does
**not** affect `SelectAlgorithmForSmallGroup()`, which handles the merged
small-community group.  This is intentional — the small-group path uses a
separate simpler heuristic.

## Recommended Ablation Experiments

```bash
# Full ablation sweep example (use graphbrew_experiment.py for proper measurements)
for toggle in "" "ADAPTIVE_NO_TYPES=1" "ADAPTIVE_NO_OOD=1" \
              "ADAPTIVE_NO_MARGIN=1" "ADAPTIVE_NO_LEIDEN=1"; do
  env $toggle python3 scripts/graphbrew_experiment.py \
    --benchmarks pr bfs --algorithms 14 \
    --graphs results/graphs/web-Google/web-Google.sg \
    --output "results/ablation_$(echo $toggle | tr '=' '_').json"
done
```
