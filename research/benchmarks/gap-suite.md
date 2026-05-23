# GAP Benchmark Suite

## Citation

```bibtex
@article{gap-2017,
  title   = {The GAP Benchmark Suite},
  author  = {Beamer, Scott and Asanovi\'{c}, Krste and Patterson, David},
  journal = {arXiv preprint arXiv:1508.03619},
  year    = {2015}
}
```

## Official Repository

- **GitHub**: [sbeamer/gapbs](https://github.com/sbeamer/gapbs) — BSD-3, 387 stars, 165 forks
- **Website**: [gap.cs.berkeley.edu/benchmark.html](http://gap.cs.berkeley.edu/benchmark.html)
- **Spec**: [arXiv:1508.03619](http://arxiv.org/abs/1508.03619)
- **Contributors**: Scott Beamer, Yunming Zhang, Michael Sutton + 6 others
- **Note**: "These baseline implementations are representative of state-of-the-art performance, and thus new contributions should outperform them to demonstrate an improvement."

## Why Faithful Implementation Matters

GraphBrew's benchmark kernels are **forked from GAPBS**. If our modifications break kernel behavior:
- Benchmark times are wrong → ML training data is wrong → AdaptiveOrder makes bad selections
- Self-recording JSON schema must match → `test_self_recording.py` (V1-V6) validates this
- Cache simulation variants (`bench/src_sim/`) must produce identical access patterns to standard variants

## Kernels

| Kernel | Algorithm | File |
|--------|----------|------|
| BFS | Direction-Optimizing | [bfs.md](bfs.md) |
| PR | PageRank (pull, Gauss-Seidel & Jacobi) | [pr.md](pr.md) |
| PR_SPMV | PageRank (SpMV variant) | [pr.md](pr.md) |
| CC | Afforest (Connected Components) | [cc.md](cc.md) |
| CC_SV | Shiloach-Vishkin | [cc.md](cc.md) |
| BC | Brandes (Betweenness Centrality) | [bc.md](bc.md) |
| SSSP | Delta-Stepping | [sssp.md](sssp.md) |
| TC | Triangle Counting | [tc.md](tc.md) |

## Graph Formats

`.el` (edge list), `.wel` (weighted), `.sg` (serialized CSR), `.wsg` (weighted serialized), `.mtx` (Matrix Market), `.gr` (DIMACS), `.graph` (Metis)

## Full Benchmark Execution Warning (from repo)

> A full run requires ~275 GB disk + 64 GB RAM. Building input graphs can take up to 8 hours.

## GraphBrew Integration

- **Source**: `bench/src/` — all kernels, modified with `-o` reordering flag and JSON self-recording
- **Build**: `make all` (standard), `make all-sim` (cache simulation)
- **Output**: `bench/bin/` (standard), `bench/bin_sim/` (simulation)
