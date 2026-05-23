# GraphBrewOrder — Algorithm ID: 12

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

GraphBrewOrder is **our algorithm**. It depends on the correctness of:
- **Leiden community detection** — wrong communities → wrong per-community orderings
- **Intra-community orderers** — must match their respective papers (RabbitOrder, HubCluster, etc.)
- **Size-sorted merge** — largest-first concatenation is a design choice; changing order affects cache behavior
- **VLDB ablation studies** compare presets (leiden vs rabbit vs hubcluster) — all must be faithful

## Key Contributions

1. **Per-community optimization**: Detects communities and applies ordering within each independently
2. **Community-preserving locality**: Keeps same-community vertices contiguous (global orderings destroy this)
3. **Size-sorted merge**: Largest communities first → most frequently accessed at lower addresses
4. **Configurable pipeline**: Any community detector × any intra-community orderer

## Algorithm Description

1. **Community Detection**: Default Leiden (configurable resolution). Alternative: RabbitOrder clustering.
2. **Community Classification**: Small vs large dynamic threshold. Small → merged + lightweight heuristic.
3. **Per-Community Reordering**: Each large community independently ordered. Default: RabbitOrder.
4. **Size-Sorted Concatenation**: Communities sorted by descending |V|. Assign IDs: C₀=[0,|C₀|), C₁=[|C₀|,|C₀|+|C₁|), etc.

**Complexity**: O(|E|) Leiden + O(Σ cost(ordering_i) per community).

### Presets

| Preset | Community Detection | Intra-Community Ordering |
|--------|-------------------|-------------------------|
| `leiden` | Leiden | Leiden internal |
| `rabbit` | RabbitOrder | RabbitOrder |
| `hubcluster` | Leiden | HubCluster |

### Intra-Community Strategies (14)

`hrab`, `dfs`, `bfs`, `conn`, `dbg`, `hsort`, `hclust`, `dsort`, `rand`, `nat`, `rcm`, `gorder`, `minla`, `maxla`

## GraphBrew Integration

- **Algorithm ID**: 12 (GRAPHBREWORDER)
- **C++ Implementation**: `bench/include/graphbrew/reorder/reorder_graphbrew.h` (~7,359 lines)
- **Leiden dependency**: `bench/include/external/leiden/` (GVE-Leiden bundled)
- **CLI**: `-o 12`
- **Python experiments**: `scripts/experiments/vldb_paper_experiments.py`
