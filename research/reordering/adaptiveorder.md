# AdaptiveOrder — Algorithm ID: 14

## Citation

```bibtex
@inproceedings{graphbrew-vldb26,
  title     = {GraphBrew: Adaptive Graph Reordering via Community-Aware Optimization},
  author    = {Mughrabi, Abdullah and others},
  booktitle = {Proceedings of the VLDB Endowment},
  year      = {2026}
}
```

## Why Faithful Implementation Matters

AdaptiveOrder is **our ML-powered algorithm selector**. Its correctness depends on:
- **All candidate algorithms** being faithfully implemented (it selects among them)
- **Feature extraction** matching between Python training and C++ runtime
- **Scoring formula** being identical in `weights.py` (Python SSOT) and `reorder_types.h` (C++ runtime)
- **Training data integrity** — if any algorithm produces wrong benchmark times, the model trains on lies

## Key Contributions

1. **Automatic algorithm selection**: ML picks the best reordering for a given graph
2. **Multi-model**: Perceptron, decision tree, hybrid, kNN database lookup
3. **Type-based clustering**: Separate models per graph family
4. **OOD safety**: Falls back to ORIGINAL for out-of-distribution graphs

## ML Performance (from wiki/AdaptiveOrder-ML.md)

| Model | Accuracy | ≤5% Regret | Avg Regret |
|-------|----------|-----------|-----------|
| XGBoost Fam+Orig | **66.3%** | 56.8% | 280% |
| XGBoost Family | 64.1% | 55.5% | 289% |
| Perceptron (LOGO) | ~52% | — | — |
| Decision Tree | ~45% | — | — |

**Key insight**: ORIGINAL wins **64% of E2E tasks** — strong bias toward "don't reorder". Penalty for wrong reordering (+208%) vastly outweighs gains (+22%).

## Feature Vector (24D)

**18 Linear**: modularity, log_nodes, log_edges, density, avg_degree, degree_variance, hub_concentration, clustering_coeff, packing_factor (IISWC'18), forward_edge_fraction (GoGraph), working_set_ratio, avg_path_length, diameter, community_count, vertex_significance_skewness (DON-RL), window_neighbor_overlap (DON-RL), packing_factor_cl (IISWC'18), wsr_l1/l2

**5 Quadratic**: degree_variance×hub_concentration, modularity×log_nodes, packing_factor×log₂(wsr), forward_edge_fraction×convergence (PR/SSSP), cache impact terms

## Selection Modes (6)

| Mode | Name | Criterion |
|------|------|-----------|
| 0 | fastest-reorder | Minimize reorder time |
| 1 | fastest-execution | Minimize kernel time (default) |
| 2 | best-endtoend | Minimize reorder + kernel |
| 3 | best-amortization | Minimize break-even iterations |
| 4 | decision-tree | sklearn DT per-benchmark |
| 5 | hybrid | DT + perceptron tiebreaker |

## Safety Guards

- **OOD detection**: Type distance > 1.5 → fall back to ORIGINAL
- **Margin check**: Best - ORIGINAL < 5% → keep ORIGINAL

## GraphBrew Integration

- **Algorithm ID**: 14 (ADAPTIVEORDER)
- **C++ Runtime**: `bench/include/graphbrew/reorder/reorder_adaptive.h`
- **C++ Database**: `bench/include/graphbrew/reorder/reorder_database.h`
- **Python ML**: `scripts/lib/ml/weights.py` (SSOT scoring), `training.py`, `features.py`, `model_tree.py`, `eval_weights.py`, `adaptive_emulator.py`, `oracle.py`
- **CLI**: `-o 14`
- **Models**: `results/data/adaptive_models.json`
