# Edge-Centric and GAS CPU Baselines

GraphBrew keeps the existing GAPBS-derived programs as the canonical
vertex/frontier baselines and adds edge-centric schedules as separate CPU
implementations. Gather-Apply-Scatter (GAS) is added only where its phases match
the algorithm rather than being forced onto every graph problem.

The executable contract is:

```text
bench/contracts/edge_gas_algorithms.json
```

Validate it with:

```bash
make check-edge-contracts
make check-edge-contract-profiles
```

## Canonical algorithms

| Algorithm | Edge-centric | GAS | Required structure |
|---|---:|---:|---|
| BFS | yes | no | sparse push plus dense pull |
| BC | yes | no | forward BFS plus ordered backward gather |
| CC | yes | yes | edge linking; optional active min-label propagation |
| CC-SV | yes | no separate GAS | atomic hooking plus pointer jumping |
| PR | yes | yes | incoming gather and residual convergence |
| PR-SPMV | yes | no separate GAS | deterministic synchronous gather |
| SSSP | yes | yes | Delta-Stepping edge baseline; active-set GAS reference |
| TC | yes | no | oriented edges plus sorted adjacency intersection |

`bfs_p` and `tc_p` are specialized consumers, not separate semantic
algorithms. Existing `src_sim` files are legacy instrumented forks and are not
the semantic contract authority; some have already drifted from canonical
source/verifier behavior. New cache-simulation binaries must instantiate shared
kernels through access hooks instead of copying algorithm bodies.

## Correctness rules

- Existing verifier functions remain authoritative.
- BFS parent identity may vary, but it must be a valid shortest-path tree.
- CC labels are representative-invariant; compare partitions, not label bytes.
- BC backward accumulation preserves stored CSR successor order because the
  verifier compares normalized floats at `FLT_EPSILON` scale.
- SSSP distances and TC counts are exact.
- PR variants pass the existing residual threshold.
- Kernel and verifier `SourcePicker` instances must select the same source.

## Data and scheduling

- CSR/source order serves push and scatter.
- CSC/destination order serves pull and Gather.
- Undirected graphs expose both all directed CSR entries and an oriented
  one-entry-per-edge stream.
- Edge-stream construction is outside timed trials unless conversion itself is
  the benchmark.
- Dense gathers use destination ownership and deterministic segmented
  reduction.
- Sparse traversals use thread-local frontier queues and CAS/min updates.
- Work counters are informational; verifier-defined output is the cross-thread
  correctness gate.

## Shared edge primitives

`bench/include/graphbrew/edge/` provides the common CPU schedule layer:

- source-major and destination-major flat views preserve logical
  `(source, destination)` edge identity;
- non-owning `EdgeStream` views reject temporary flat graphs at compile time;
- ordinal partitions cover every directed entry exactly once, while oriented
  undirected iteration retains only `source < destination`;
- `Frontier` combines sorted sparse IDs with a dense bit map;
  `FrontierBuilder` atomically deduplicates parallel producers into
  thread-local queues;
- integer min/max/CAS helpers report whether an update won;
- edge-map access policies are invoked concurrently and must be thread-safe.

Run the primitive and thread-count checks with:

```bash
make check-edge-primitives
```

View construction is intentionally measured separately from algorithm trials:

```bash
make edge_view_benchmark
OMP_NUM_THREADS=4 bench/bin/edge_view_benchmark -g 18
```

## Dense iterative edge baselines

The first four edge binaries share algorithm headers under
`bench/include/graphbrew/algorithms/`:

- `pr_spmv_edge` is synchronous Jacobi PageRank. Each iteration freezes
  outgoing contributions, then performs destination-owned incoming gathers in
  stored edge order.
- `pr_edge` is an explicitly asynchronous destination-owned PageRank schedule.
  Cross-owner contributions use atomic floats, so immediate visibility is
  data-race-free. Iteration counts may differ by thread count; the existing
  residual verifier is the semantic gate.
- `cc_edge` preserves Afforest neighbor sampling, skips proven sampled prefixes,
  then performs edge-balanced atomic union. Directed edges connect endpoints
  for weak connectivity; symmetric graphs use one oriented edge.
- `cc_sv_edge` performs CAS-safe monotone root hooking followed by atomic
  shortcutting.

All flat views are built before `BenchmarkKernel`, so trial timing excludes
representation conversion. The paired matrix runs every declared profile,
including directed, disconnected, synthetic, and dangling-vertex cases,
against the canonical binary and the edge binary at OMP 1/2/4/8:

```bash
make check-edge-dense
```

The matrix gates verifier-defined output. It does not require equal PageRank
iterations or bit-identical asynchronous scores.

## Frontier and weighted edge baselines

- `bfs_edge` keeps one sparse-plus-bitmap frontier. Sparse phases push outgoing
  edges with CAS parent claims; dense phases own destinations and pull from the
  incoming flat view, so directed traversal uses true predecessors. The
  canonical alpha/beta switching and scout/awake work metrics are preserved.
- `sssp_edge` retains Delta-Stepping rather than scanning every edge. One
  persistent OpenMP team processes the shared current bin, relaxes only active
  vertices' weighted outgoing ranges, fuses small same-bin thread-local work,
  and selects the globally smallest remaining bin.
- weighted outgoing flattening preserves `(source, destination, weight)` and is
  built before timed trials.

Both kernel and verifier use separately constructed, identically seeded
`SourcePicker` instances. Run all registered graph/thread profiles plus paired
multi-trial source checks with:

```bash
make check-edge-frontier
```

The current matrix passes 36/36 verifier-backed edge trials at OMP 1/2/4/8.

## Literature

- PowerGraph/GAS:
  https://www.usenix.org/conference/osdi12/technical-sessions/presentation/gonzalez
- X-Stream: https://dl.acm.org/doi/10.1145/2517349.2522740
- Ligra: https://jshun.csail.mit.edu/ligra.shtml
- Direction-Optimizing BFS:
  https://people.eecs.berkeley.edu/~krste/papers/beamer-sc2012.pdf
- GraphIt: https://graphit-lang.org/
- GAPBS: https://arxiv.org/abs/1508.03619
