# SSSP — Single-Source Shortest Paths (Delta-Stepping)

## Citations

```bibtex
@article{delta-stepping-2003,
  title   = {\delta-stepping: a parallelizable shortest path algorithm},
  author  = {Meyer, Ulrich and Sanders, Peter},
  journal = {Journal of Algorithms},
  volume  = {49},
  number  = {1},
  pages   = {114--152},
  year    = {2003}
}

@inproceedings{graphit-cgo20,
  title     = {Optimizing ordered graph algorithms with GraphIt},
  author    = {Zhang, Yunming and Brahmakshatriya, Ajay and Chen, Xinyi and Dhulipala, Laxman and Kamil, Shoaib and Amarasinghe, Saman and Shun, Julian},
  booktitle = {18th International Symposium on Code Generation and Optimization (CGO)},
  pages     = {158--170},
  year      = {2020}
}
```

## Algorithm

Group vertices into buckets by distance (width = δ). Process in bucket order:
1. **Light edges** (weight ≤ δ): relax within current bucket, repeat until stable
2. **Heavy edges** (weight > δ): relax to later buckets

δ = 0 → Dijkstra; δ = ∞ → Bellman-Ford. Bucket fusion (Zhang et al. 2020) reduces synchronization.

**Convergence**: Benefits from forward-edge orderings (GoGraph) — vertices read already-updated distances.

## GraphBrew Integration

- **Source**: `bench/src/sssp.cc` — Authors: Scott Beamer, Yunming Zhang
- **ECG classification**: Traversal-type
- **AdaptiveOrder**: Gets `forward_edge_fraction × convergence` bonus in perceptron
