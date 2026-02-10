# Phase 6 — SOTA-Inspired Improvements

**Commit:** (pending)  
**Date:** 2025-07-19

## Overview

Phase 6 implements **Tier 1 P0** improvements from `06_SOTA_IDEAS.md`,
both gated behind environment-variable toggles for safe A/B testing.

---

## P0 1.1 — Cost-Aware Dynamic Margin (IISWC'18)

**Paper:** Balaji & Lucia, IISWC'18 — reordering only helps a subset
of graph×algorithm combinations; lightweight metrics can predict when
reordering is beneficial.

**What it does:** Enhances the existing fixed-threshold margin check
(`ORIGINAL_MARGIN_THRESHOLD = 0.05`) with a dynamic component that
scales with the expected reorder cost. Higher `avg_reorder_time` for
the chosen algorithm → larger margin required to justify reordering.

**Implementation:**
- File: [reorder_types.h](../bench/include/graphbrew/reorder/reorder_types.h)
- Function: `SelectReorderingFromWeights()`
- Logic:
  ```
  threshold = max(ORIGINAL_MARGIN_THRESHOLD,
                  COST_MODEL_ALPHA × avg_reorder_time)
  if margin < threshold → return ORIGINAL
  ```
- `COST_MODEL_ALPHA = 0.01` (seconds → score units, needs calibration)

**Toggle:** `ADAPTIVE_COST_MODEL=1`

**Test criteria:**
- A/B on MEDIUM tier: compare end-to-end time with/without cost model
- Success = fewer graphs where reorder cost exceeds benefit
- Must not degrade PR geo-mean by more than 5%

---

## P0 1.2 — Packing Factor Short-Circuit (IISWC'18)

**Paper:** IISWC'18 — "packing factor" measures fraction of hub
neighbours already co-located in memory. High packing = current
ordering already has good locality.

**What it does:** Before running the full perceptron, checks if the
community's data is already well-packed and fits in cache. If so,
skips reordering entirely (returns ORIGINAL).

**Implementation:**
- File: [reorder_types.h](../bench/include/graphbrew/reorder/reorder_types.h)
- Function: `SelectBestReorderingForCommunity()`
- Logic:
  ```
  if packing_factor > 0.7 AND working_set_ratio < 2.0:
      return ORIGINAL
  ```

**Toggle:** `ADAPTIVE_PACKING_SKIP=1`

**Test criteria:**
- Count how many communities hit the short-circuit on SMALL/MEDIUM tiers
- For graphs that trigger: verify ORIGINAL time is within 5% of oracle
- Particularly relevant for mesh/rgg graphs with natural locality

---

## Smoke Test Results

```
# P0 1.1 — Cost model toggle recognized
ADAPTIVE_COST_MODEL=1 ./bench/bin/bfs -f graph.sg -o 14 -n 1
→ "COST_MODEL: cost-aware dynamic margin (P0 1.1)"
→ Community selections unchanged (α=0.01 is conservative)

# P0 1.2 — Packing skip toggle recognized
ADAPTIVE_PACKING_SKIP=1 ./bench/bin/bfs -f graph.sg -o 14 -n 1
→ "PACKING_SKIP: packing factor short-circuit (P0 1.2)"
→ No communities triggered (social/web graphs have packing < 0.7)

# Both together
ADAPTIVE_PACKING_SKIP=1 ADAPTIVE_COST_MODEL=1 ./bench/bin/bfs -o 14 -n 1 ...
→ Both toggles active, no interference
```

## Feature Values Observed

| Graph | Packing Factor | WSR | Skip Triggered? |
|-------|---------------|-----|-----------------|
| soc-Slashdot0902 | 0.154 | 0.143 | No (packing < 0.7) |
| web-Google | 0.013 | 1.046 | No (packing < 0.7) |

Social and web graphs have low packing factors — the short-circuit
targets mesh/geometric graphs where spatial locality is inherent.

---

## Calibration Needed

Both P0 improvements require calibration before enabling by default:

1. **COST_MODEL_ALPHA** (currently 0.01): Run full sweep on MEDIUM tier,
   measure correlation between `avg_reorder_time` and actual regret.
   Adjust α to minimize false positives (skipping when reorder helps).

2. **Packing thresholds** (0.7 / 2.0): Run oracle analysis on graphs
   with varying packing factors. Find the threshold where ORIGINAL
   matches oracle within 5%.

---

## Priority Matrix (from 06_SOTA_IDEAS.md)

| Idea | Status | Toggle |
|------|--------|--------|
| **1.1 Cost model** | ✅ Implemented | `ADAPTIVE_COST_MODEL=1` |
| **1.2 Packing skip** | ✅ Implemented | `ADAPTIVE_PACKING_SKIP=1` |
| 1.3 FEF convergence | Not started | — |
| 1.4 Calibrated margins | Not started | — |
| 2.1 Bandit exploration | Not started | — |
| 2.2 Per-type OOD radius | Not started | — |
| 2.3 Overhead filtering | Not started | — |

Proceed to P1 only after P0 shows improvement on MEDIUM tier evaluation.
