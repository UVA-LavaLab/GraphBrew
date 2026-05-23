# GoGraphOrder — Algorithm ID: 16

## Citation

```bibtex
@article{gograph-tpds24,
  title   = {GoGraph: Fast Iterative Graph Computing with Updated Neighbor States},
  author  = {Zhou, Xianghao and others},
  journal = {IEEE Transactions on Parallel and Distributed Systems (TPDS)},
  year    = {2024}
}
```

## Official Repository

- **GitHub**: [iDC-NEU/GoGraph](https://github.com/iDC-NEU/GoGraph) — no explicit license
- **Author**: zhouyijie9 ([@zhouyijie9](https://github.com/zhouyijie9))
- **Dependencies**: g++ 4.9.2+, Boost 1.58.0, libnuma, libtcmalloc
- **Usage**: `./run_all.sh [GRAPH_NAME]` (e.g., `./run_all.sh indochina.mtx web-google.mtx`)
- **Input**: Edge list (`.mtx`), output: `.GoGraph` suffix
- **Test included**: `test/pagerank-async.cpp` for validating reordering effectiveness
- **Includes**: `relative_works/` directory with comparison implementations, preprocessing for isolated/high-degree vertices

## Why Faithful Implementation Matters

GoGraph's forward-edge-fraction (FEF) metric is one of the **18 features** in AdaptiveOrder's perceptron scoring, with a **special convergence bonus** for PageRank and SSSP. If our GoGraph implementation computes M(σ) incorrectly:
- The `forward_edge_fraction` feature is wrong → perceptron model trained on bad data
- Convergence benefits for iterative algorithms are misattributed

## Key Contributions

1. **M(σ) optimization**: Maximizes "positive" (forward) edges — edges (u,v) where σ(u) < σ(v)
2. **Convergence acceleration**: Iterative algorithms use more up-to-date neighbor values → fewer iterations
3. **Complementary to cache optimization**: Optimizes temporal locality (data flow) not spatial (cache)

## Algorithm Description

- **FEF** = |{(u,v) ∈ E : σ(u) < σ(v)}| / |E|
- FEF = 0.5 for random ordering; FEF > 0.5 = forward-biased
- For DAGs: topological sort → FEF = 1.0
- For general graphs: heuristic approaches achieve FEF > 0.5

**Complexity**: O(|E|) typical.
**Best for**: Iterative algorithms (PageRank, SSSP, label propagation).

## GraphBrew Integration

- **Algorithm ID**: 16 (GOGRAPHORDER)
- **C++ Implementation**: `bench/include/graphbrew/reorder/reorder_gograph.h`
- **CLI**: `-o 16`
- **ML Feature**: `forward_edge_fraction` with convergence bonus for PR/SSSP
