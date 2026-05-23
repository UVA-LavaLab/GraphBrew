# PR — PageRank

## Citation

GAP Benchmark Suite — Beamer, Asanović, Patterson (arXiv:1508.03619, 2015).

## Algorithm

```
PR(v) = (1-d)/|V| + d × Σ_{u ∈ in-neighbors(v)} PR(u) / out_degree(u)
```

**Pull-based** (GAP suite): each vertex reads from in-neighbors. Gauss-Seidel & Jacobi variants. Converges when max |ΔPR| < ε (default 10⁻⁶).

**PR_SPMV**: SpMV formulation `PR = (1-d)/|V| × 1 + d × Aᵀ × PR`. Row-by-row traversal.

| Property | PR (Standard) | PR_SPMV |
|----------|--------------|---------|
| Access | Random neighbor reads | Sequential matrix rows |
| Iterations | 10-30 typical | 10-30 typical |
| Cache sensitivity | Very high | Moderate |

**Why reordering matters**: PR accesses ALL in-neighbors every iteration. Scattered neighbors → cache misses. Co-locating neighbors (RabbitOrder, GOrder, GraphBrewOrder) dramatically reduces miss rates.

**Convergence bonus**: Forward-edge orderings (GoGraph) let Gauss-Seidel read already-updated values → faster convergence.

## GraphBrew Integration

- **PR Source**: `bench/src/pr.cc` — Author: Scott Beamer
- **PR_SPMV Source**: `bench/src/pr_spmv.cc`
- **ECG classification**: Iterative-type (Section B5)
- **Both included** in ECG paper's 7 benchmarks
