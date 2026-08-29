<p align="center"><img src="./docs/figures/logo.png" width="180"></p>

# GraphBrew

GraphBrew is a C++17/OpenMP framework for **composable vertex reordering**.
It separates three decisions that monolithic reorderers usually couple:

| Decision | Purpose | Examples |
|---|---|---|
| Partitioner | discover vertex groups | GVE-Leiden, Rabbit |
| Block layout | place groups globally | identity, SizeDesc, supergraph order |
| Vertex layout | order vertices inside each block | BFS, RCM, local Gorder |

![GraphBrew architecture](./docs/figures/graphbrew-architecture.svg)

## Paper story

GraphBrew makes each composition explicit, records the realized mapping, and
measures mapping cost, kernel behavior, executed work, and amortized
end-to-end time separately.

The current paper supports two contributions:

1. **Fixed quality composition.**
   `LeidenGVE-SizeDesc-LocalGorder8`

   ```text
   12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8
   ```

   On the fresh 11-graph, seven-kernel confirmation, comparator/GraphBrew
   kernel-time geometric means are:

   | Comparator | Kernel GM | GraphBrew/comparator mapping GM |
   |---|---:|---:|
   | Rabbit CSR | 1.042x | 18.52x |
   | Rabbit Boost | 1.044x | 17.35x |
   | GORDER_csr | 1.052x | 0.752x |

   This is a **quality point**, not universal dominance. Rabbit break-even is
   roughly 37,000–40,000 reuses, and summed Rabbit seconds never cross.

2. **Compact-and-Emit.**
   A one-pass construction optimization that compacts active community IDs
   and writes final IDs during intra-community BFS.

   ```text
   12:leiden:compose:sg_none:comm_identity:
   intra_bfs_compact_direct:cd_parallel:sgmb4096:norefine:1:1
   ```

   It preserves the BFS permutation and reduces five-graph mapping cost to
   0.479x the faster Rabbit implementation. A mandatory ORIGINAL audit found
   no low-reuse region that beats both doing nothing and Rabbit, so this is a
   **mapping-construction contribution**, not a balanced ordering claim.

![GraphBrew evidence boundary](./docs/figures/graphbrew-evidence-boundary.svg)

The corrected composition atlas reinforces the boundary: the in-sample oracle
reaches 1.062x over the fastest comparator, but the best fixed arm reaches
0.946x and graph-held-out selection reaches 0.907x. GraphBrew therefore makes
no automated-generator or universal-selector claim.

Machine-readable claim values and source hashes are in
[`docs/recommendation-evidence.json`](docs/recommendation-evidence.json).

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

# Confirmed fixed GraphBrew quality composition
./bench/bin/pr -f "$GRAPH" -s \
  -o '12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8' \
  -n 3
```

`-o 12:<configuration>` always runs the exact composition supplied by the
user. GraphBrew does not search compositions at runtime.

## Reproduce experiments

Use the public orchestrator:

```bash
python3 scripts/graphbrew_experiment.py --vldb 2 --paper-preview \
  --paper-graph-dir /media/NVMeData/00_GraphDatasets/GraphBrew \
  --paper-artifact-root /media/NVMeData/00_GraphDatasets/GraphBrew/artifacts \
  --paper-threads 4 --paper-cpu-list 24-27
```

Final timing uses frozen protocols, pre-generated mappings, semantic
verification, fixed affinity, repeated trials, and fail-closed scheduler
checks. Large graphs and result artifacts belong on the external NVMe
partition, not in the repository.

## Repository map

```text
bench/src/                         graph kernels
bench/include/graphbrew/reorder/   reordering implementations
bench/include/external/gapbs/      graph builder and benchmark lifecycle
scripts/graphbrew_experiment.py    public experiment orchestrator
scripts/experiments/               frozen and restartable campaigns
scripts/test/                      regression and evidence checks
docs/                              public figures and evidence manifest
wiki/                              documentation source
```

## Documentation

- [Evidence and Claims](https://github.com/UVA-LavaLab/GraphBrew/wiki/Evidence-and-Claims)
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
