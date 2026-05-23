# RabbitOrder — Algorithm ID: 8

## Citation

```bibtex
@inproceedings{rabbitorder-ipdps16,
  title     = {Rabbit Order: Just-in-time Parallel Reordering for Fast Graph Analysis},
  author    = {Arai, Junya and Shiokawa, Hiroaki and Yamamuro, Takeshi and Onizuka, Makoto and Iwamura, Sotetsu},
  booktitle = {2016 IEEE International Parallel and Distributed Processing Symposium (IPDPS)},
  year      = {2016}
}
```

## Official Repository

- **GitHub**: [araij/rabbit_order](https://github.com/araij/rabbit_order) — Custom license
- **Format**: Single-header C++ (`rabbit_order.hpp`)
- **Dependencies**: g++ 4.9.2+, Boost 1.58.0, libnuma, libtcmalloc_minimal
- **Note from authors**: "Some graph datasets are already reordered, and so Rabbit Order will not show significant performance improvement on those graphs" (e.g., LAW graphs reordered by Layered Label Propagation, web graphs ordered by URL)

## Why Faithful Implementation Matters

RabbitOrder is the **default intra-community orderer** in GraphBrewOrder (Algorithm 12). If our implementation deviates:
- GraphBrewOrder results change because per-community reordering quality changes
- VLDB paper's ablation studies (rabbit preset vs leiden preset) become inconsistent
- AdaptiveOrder's training data includes RabbitOrder as a candidate — wrong perf skews ML model

## Key Contributions

1. **Just-in-time reordering**: Reorders vertices during graph loading, overlapping I/O with computation
2. **Hierarchical community detection**: Graph bisection via adjacency similarity discovers nested communities
3. **DFS-based merge**: DFS traversal of the community hierarchy tree places related communities adjacently
4. **Parallel execution**: Task parallelism in hierarchical decomposition

## Algorithm Description

1. **Community Detection via Hierarchical Bisection**: Repeatedly bisect graph using adjacency similarity (Jaccard-like). Build binary dendrogram.
2. **DFS Ordering**: Traverse dendrogram in DFS order — vertices in DFS order form the new permutation.
3. **Label Map Output**: `new_id → old_id` mapping.

**Complexity**: O(|E| log |V|) typical, O(|E| × |V|) worst case.
**Best for**: Social networks, web graphs, graphs with strong community structure.

## GraphBrew Integration

- **Algorithm ID**: 8 (RABBITORDER)
- **C++ Implementation**: `bench/include/graphbrew/reorder/reorder_rabbit.h`
- **External Library**: `bench/include/external/rabbit/` (bundled)
- **CLI**: `-o 8`
- **Variants**: `default`, `csr` (no Boost), `boost` (original)
- **Used by**: GraphBrewOrder (Algo 12) as default intra-community orderer with `rabbit` preset
