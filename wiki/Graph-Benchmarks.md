# Graph Benchmarks

GraphBrew ships eight canonical benchmark binaries. The frozen reordering
matrix uses seven; triangle counting remains an explicit secondary workload.

| Kernel | Binary | Primary access pattern | Default matrix |
|---|---|---|---|
| Pull PageRank | `pr` | repeated neighbor-property gathers | yes |
| SpMV PageRank | `pr_spmv` | Jacobi sparse matrix-vector traversal | yes |
| Direction-optimizing BFS | `bfs` | frontier-dependent top-down/bottom-up traversal | yes |
| Afforest CC | `cc` | sampled linking and component propagation | yes |
| Shiloach-Vishkin CC | `cc_sv` | repeated parent hooking and compression | yes |
| Delta-stepping SSSP | `sssp` | weighted bucketed frontier traversal | yes |
| Betweenness Centrality | `bc` | forward and backward BFS phases | yes |
| Triangle Counting | `tc` | sorted-neighbor intersections | no |

## Common command

```bash
./bench/bin/<kernel> -f graph.sg -s -o <ordering> -n <trials>
```

Start with ORIGINAL and both Rabbit variants before evaluating a GraphBrew
composition:

```bash
./bench/bin/pr -f graph.sg -s -o 0 -n 5
./bench/bin/pr -f graph.sg -s -o 8:csr -n 5
./bench/bin/pr -f graph.sg -s -o 8:boost -n 5
```

See [Command-Line Reference](Command-Line-Reference) for shared and
kernel-specific flags.

## PageRank

```bash
./bench/bin/pr -f graph.sg -s -o 12:leiden -n 5
./bench/bin/pr_spmv -f graph.sg -s -o 12:leiden -n 5
```

Pull PR and PR-SpMV have different update semantics. Report iteration count,
executed work, and kernel time separately; a convergence change must not be
presented as a pure locality gain.

## BFS

```bash
./bench/bin/bfs -f graph.sg -s -o 12:leiden -r 0 -n 5
```

Source selection changes traversed work. Frozen comparisons bind the source
set and verify answer fingerprints.

## Connected Components

```bash
./bench/bin/cc -f graph.sg -s -o 12:leiden -n 5
./bench/bin/cc_sv -f graph.sg -s -o 12:leiden -n 5
```

Afforest and CC-SV exercise different memory and work-propagation patterns.
CC-SV is a weak point for several community layouts and must remain visible in
aggregate reporting rather than being folded into Afforest CC.

## SSSP

```bash
./bench/bin/sssp -f graph.wsg -s -o 12:leiden -r 0 -d 2 -n 5
```

SSSP requires weighted input. Final campaigns bind the source IDs, weight
scheme and checksum, delta, conversion policy, and answer fingerprint.

## Betweenness Centrality

```bash
./bench/bin/bc -f graph.sg -s -o 12:leiden -r 0 -i 4 -n 5
```

BC repeats forward and backward traversals. Bind source and iteration policy
when comparing orderings.

## Triangle Counting

```bash
./bench/bin/tc -f graph.sg -s -o 12:leiden -n 5
```

TC has a different access pattern from the traversal and propagation kernels.
Evaluate it independently rather than transferring conclusions from another
kernel.

## Reporting

For every kernel record:

- graph and labeling identity;
- exact ordered `-o` specification;
- mapping fingerprint;
- source/iteration/delta policy where applicable;
- threads and affinity;
- trial count and verification state;
- mapping and kernel time separately.
