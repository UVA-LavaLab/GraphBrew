# ML Features & Metrics — Academic References

## Overview

AdaptiveOrder uses features derived from several academic works. This documents their provenance.

---

## IISWC'18 — Hub Neighbor Co-Location (Packing Factor)

- **Paper**: Balaji & Lucia, "When is Graph Reordering an Optimization?" IISWC 2018
- **Repo**: [CMUAbstract/Graph-Reordering-IISWC18](https://github.com/CMUAbstract/Graph-Reordering-IISWC18)
- **Features**: `packing_factor` (hub neighbor co-location), `packing_factor_cl` (cache-line variant)
- **How**: For each hub, count neighbors on same cache line. Higher = better existing locality = less reorder benefit.

## DON-RL — Pairwise Locality Scoring

- **Features**: `vertex_significance_skewness` (per-vertex locality variation), `window_neighbor_overlap` (fraction of neighbors within window)
- **How**: For vertex v in window [v-w/2, v+w/2], count v's neighbors in that window / degree(v). Compute mean and skewness.

## GoGraph — Forward Edge Fraction (FEF)

- **Paper**: Zhou et al., "GoGraph" IEEE TPDS 2024
- **Repo**: [iDC-NEU/GoGraph](https://github.com/iDC-NEU/GoGraph)
- **Feature**: `forward_edge_fraction` with special convergence bonus for PR/SSSP
- **How**: Count edges (u,v) where u < v. FEF ≈ 0.5 for random; >0.5 = forward-biased.

## Chen & Chung — Workload Balancing

```bibtex
@article{chen-chung-tpds21,
  title   = {Workload Balancing via Graph Reordering on Multicore Systems},
  author  = {Chen, Yulin and Chung, Yeh-Ching},
  journal = {IEEE TPDS},
  year    = {2021}
}
```

Insight: degree-based reordering can cause load imbalance by concentrating high-degree vertices in one thread's partition.

## Working Set Ratio (WSR)

- `working_set_ratio` = log₂(graph_data_size / LLC_size)
- `wsr_l1`, `wsr_l2` = L1/L2-specific variants
- WSR > 1 means graph doesn't fit in cache. Higher = more pressure = more reorder benefit.

## Integration

- **Python**: `scripts/lib/ml/features.py`
- **C++**: `bench/include/graphbrew/reorder/reorder_types.h`
- **Scoring**: `scripts/lib/ml/weights.py` — `PerceptronWeight.compute_score()`
