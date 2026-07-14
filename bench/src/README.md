# bench/src/ — Benchmark Drivers

Each file is a standalone GAPBS benchmark that links against the header-only library.

| File | Algorithm | Description |
|------|-----------|-------------|
| `bfs.cc` | Breadth-First Search | Direction-optimizing BFS |
| `sssp.cc` | Single-Source Shortest Path | Delta-stepping SSSP |
| `pr.cc` | PageRank | Pull-based iterative PageRank |
| `pr_spmv.cc` | PageRank (SpMV) | SpMV-style PageRank variant |
| `bc.cc` | Betweenness Centrality | Brandes' algorithm |
| `cc.cc` | Connected Components | Afforest algorithm |
| `cc_sv.cc` | Connected Components (SV) | Shiloach-Vishkin algorithm |
| `tc.cc` | Triangle Counting | Set-intersection based |
| `tc_p.cc` | Triangle Counting (parallel) | Parallel triangle counting |
| `converter.cc` | Graph Converter | Converts between graph formats |

All benchmarks share the same CLI flags (`-f`, `-o`, `-n`, etc.) via GAPBS `benchmark.h`.
