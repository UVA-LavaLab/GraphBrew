# GOrder — Algorithm ID: 9

## Citation

```bibtex
@inproceedings{gorder-sigmod16,
  title     = {Speedup Graph Processing by Graph Ordering},
  author    = {Wei, Hao and Yu, Jeffrey Xu and Lu, Can and Lin, Xuemin},
  booktitle = {Proceedings of the 2016 International Conference on Management of Data (SIGMOD)},
  series    = {SIGMOD '16},
  year      = {2016},
  keywords  = {CPU performance, graph algorithms, graph ordering}
}
```

## Official Repository

- **GitHub**: [datourat/Gorder](https://github.com/datourat/Gorder) — MIT
- **Includes**: `paper.pdf` in repo
- **Usage**: `./Gorder LiveJournal.txt -w 5` (default window w=5)
- **Input**: SNAP-format edge list (0-based contiguous vertex IDs, no duplicates)
- **Contact**: [weihaohal@gmail.com](mailto:weihaohal@gmail.com)

## Why Faithful Implementation Matters

GOrder is a comparison baseline in VLDB experiments. The window parameter `w` critically affects quality — the default `w=5` from the original repo must be preserved. Using the wrong `w` produces orderings that don't match published GOrder results.

## Key Contributions

1. **Window-based scoring**: Locality score = edges (u,v) where |σ(u) - σ(v)| ≤ w (neighbors within cache distance)
2. **Greedy vertex placement**: At each step, choose the unplaced vertex maximizing in-window neighbors
3. **NP-hardness proof**: Graph ordering problem is NP-hard with approximation guarantees
4. **2-5× speedups**: Demonstrated on real-world graphs for BFS, PageRank

## Algorithm Description

1. Start with arbitrary vertex, maintain priority queue
2. At each step: place vertex with most already-placed neighbors within window `w`
3. Tie-breaking: prefer higher-degree vertices
4. Prune search: only consider vertices adjacent to recently placed ones

**Complexity**: O(|E| × w) where w = window size.

## GraphBrew Integration

- **Algorithm ID**: 9 (GORDER)
- **C++ Implementation**: `bench/include/graphbrew/reorder/reorder_classic.h`
- **External Library**: `bench/include/external/gorder/` (bundled)
- **CLI**: `-o 9`
- **Variants**: `default`, `csr`, `fast`
