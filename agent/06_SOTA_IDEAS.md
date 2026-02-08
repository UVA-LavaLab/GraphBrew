# SOTA-Inspired Improvement Ideas (propose + test)

---

## Tier 1 — Minimal Risk

1) Learn/predict "SKIP reorder" explicitly using end-to-end objective.
2) Calibrate scores to expected gain (logistic regression / Platt scaling).
3) Add cost model: predicted gain − predicted reorder cost.
4) Add missing high-value candidate algorithms (if not already present):
   - Locality-driven ordering (e.g., Gorder-style)
   - Fast hub-focused ordering variants
   - RabbitOrder-like pipeline (parallel, JIT)

   (Only propose if implementable in this repo.)

---

## Tier 2 — Moderate

1) Contextual bandit:
   - Explore only when margin low
   - Constrain exploration budget
   - Fall back to ORIGINAL on uncertainty
2) Better OOD:
   - Distance-based confidence with per-type radius
   - Fallback to nearest type with penalty if far

---

## Tier 3 — Heavy

1) Learned reordering (RL / GNN-based) for large communities only.
2) Hierarchical model: graph-level gating + community-level expert

---

## For Each Idea You Propose

- **Implementation sketch** (files, API changes)
- **Testing plan** (ablations, metrics)
- **Safety plan** (fallbacks, rollback)
