# LeidenOrder — Algorithm ID: 15

## Citations

```bibtex
@article{leiden-2019,
  title   = {From Louvain to Leiden: guaranteeing well-connected communities},
  author  = {Traag, Vincent Antoine and Waltman, Ludo and van Eck, Nees Jan},
  journal = {Scientific Reports},
  volume  = {9},
  pages   = {5233},
  year    = {2019},
  doi     = {10.1038/s41598-019-41695-z}
}

@article{gve-leiden-2024,
  title   = {GVE-Leiden: Fast Leiden Algorithm for Community Detection in Shared Memory Setting},
  author  = {Sahu, Subhajit},
  journal = {arXiv preprint arXiv:2312.13936},
  year    = {2024}
}
```

## Official Repositories

- **Original Leiden**: [vtraag/libleidenalg](https://github.com/vtraag/libleidenalg) — reference implementation
- **GVE-Leiden (used by GraphBrew)**: [puzzlef/leiden-communities-openmp](https://github.com/puzzlef/leiden-communities-openmp) — MIT
  - Author: Subhajit Sahu ([@wolfram77](https://github.com/wolfram77))

## Why Faithful Implementation Matters

Leiden is the **community detection backbone** for GraphBrewOrder. If our Leiden implementation deviates:
- Community boundaries change → per-community orderings change → GraphBrewOrder results change
- Resolution parameter behavior must match the paper — different resolution ≠ different communities
- **Guarantee of well-connected communities** (no disconnected sub-communities) must hold

## GVE-Leiden Performance (from official repo, 13 graphs)

| vs. | Speedup |
|-----|---------|
| Original Leiden | **436×** |
| igraph Leiden | **104×** |
| NetworKit Leiden | **8.2×** |
| cuGraph Leiden | **3.0×** |

- **Throughput**: 403M edges/s on 3.8B-edge graph
- **Modularity**: 0.3% lower than original Leiden (negligible)
- **Zero disconnected communities** (unlike NetworKit: 1.5×10⁻², cuGraph: 6.6×10⁻⁵)

## Algorithm Description

1. **Local Move Phase**: Each vertex considers moving to neighboring community if it increases modularity
2. **Refinement Phase** (Leiden's key innovation over Louvain): Split communities to ensure internal connectivity
3. **Aggregation Phase**: Collapse communities → super-vertices, recurse
4. **Termination**: No further modularity-improving moves

**Resolution parameter (γ)**: 0.5 (fewer, larger) → 1.0 (default) → 1.5+ (more, smaller)

**Complexity**: O(|E|) per iteration, typically 5-15 iterations.

## GraphBrew Integration

- **Algorithm ID**: 15 (LEIDENORDER)
- **External Library**: `bench/include/external/leiden/` (GVE-Leiden bundled)
- **Used by**: GraphBrewOrder (Algo 12) as default community detection
- **CLI**: `-o 15`
- **Resolution**: `--resolution 0.5 0.75 1.0 1.5`
