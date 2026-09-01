<p align="center"><img src="./docs/figures/logo.png" width="180"></p>

# GraphBrew

GraphBrew is a C++17/OpenMP framework for **composable and explainable vertex
layouts**. It separates three decisions that monolithic reorderers usually
couple:

| Decision | Purpose | Examples |
|---|---|---|
| Partitioner | discover vertex groups | GVE-Leiden, Rabbit |
| Block layout | place groups globally | identity, SizeDesc, supergraph order |
| Vertex layout | order vertices inside each block | BFS, RCM, local Gorder |

![GraphBrew architecture](./docs/figures/graphbrew-architecture.svg)

## Layout model

GraphBrew represents a layout as `<P,B,L>`: partition vertices, place the
resulting blocks, and order vertices inside each block. It compiles that
expression into one persistent permutation and records requested/realized
semantics, mapping fingerprints, construction cost, executed work, and
amortized time.

For partition `P`, block permutation `B`, and local permutation `L`, the final
ID is:

```text
pi(v) = block_offset(B(P(v))) + L[P(v)](v)
```

`-o 12:<configuration>` executes the exact composition supplied by the user.
GraphBrew does not silently search for or replace a requested layout.

## Build

```bash
python3 scripts/graphbrew_experiment.py --check-deps
make -j"$(nproc)" all
make check
```

The original `-o 8:boost` path requires Boost 1.58. The native
`-o 8:csr` implementation does not.

## Run

```bash
GRAPH=scripts/test/graphs/tiny/tiny.el

# Current-layout baseline
./bench/bin/pr -f "$GRAPH" -s -o 0 -n 3

# Rabbit baselines
./bench/bin/pr -f "$GRAPH" -s -o 8:csr -n 3
./bench/bin/pr -f "$GRAPH" -s -o 8:boost -n 3

# Explicit GraphBrew composition
./bench/bin/pr -f "$GRAPH" -s \
  -o '12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8' \
  -n 3
```

`-o 12:<configuration>` always runs the exact composition supplied by the
user. GraphBrew does not search compositions at runtime.

## Run experiments

Use the public orchestrator:

```bash
python3 scripts/graphbrew_experiment.py \
  --full --quick --size small --trials 1 --skip-cache
```

Use `--dry-run` before broad collection. Large graphs and result artifacts
belong on an external data partition, not in the repository.

## Repository map

```text
bench/src/                         graph kernels
bench/include/graphbrew/reorder/   reordering implementations
bench/include/external/gapbs/      graph builder and benchmark lifecycle
scripts/graphbrew_experiment.py    public experiment orchestrator
scripts/experiments/               specialized restartable campaigns
scripts/test/                      regression checks
docs/                              public documentation and figures
wiki/                              documentation source
```

## Documentation

- [GraphBrew Running Example](https://github.com/UVA-LavaLab/GraphBrew/wiki/GraphBrew-Running-Example)
- [GraphBrewOrder](https://github.com/UVA-LavaLab/GraphBrew/wiki/GraphBrewOrder)
- [Reordering Algorithms](https://github.com/UVA-LavaLab/GraphBrew/wiki/Reordering-Algorithms)
- [Reproducible Experiments](https://github.com/UVA-LavaLab/GraphBrew/wiki/Reproducible-Experiments)
- [Command-Line Reference](https://github.com/UVA-LavaLab/GraphBrew/wiki/Command-Line-Reference)

## Attribution and license

GraphBrew includes or compares against GAPBS, Rabbit Order, Gorder,
GVE-Leiden, RCM, Corder, and related locality methods. Attributions are
recorded in source headers and the wiki.

See [LICENSE](LICENSE).
