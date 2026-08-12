# Partitioning and Shards

The compact-CSR partition path is separate from GraphBrew's canonical kernel
and reordering experiments. It provides deterministic ownership, ghost
metadata, runtime-traffic accounting, and the versioned `graph.shard.v1`
package.

## Build and smoke test

```bash
make bfs_p graph_shard_export
./bench/bin/bfs_p -f scripts/test/data/tiny.el -s \
  -n 1 -r 0 -v -P 4 -B total
make check-partition
```

`-P` selects the shard count. `-B` accepts `vertices`, `out`, or `total`.

## Export a package

```bash
./bench/bin/graph_shard_export \
  -f /path/to/graph.sg \
  -P 16 -B total \
  -E /path/to/output-package
```

`graph.shard.v1` contains a manifest, source/internal mapping sidecars, and
per-shard little-endian CSR/ghost arrays. The streamed exporter builds and
writes one shard at a time instead of materializing every shard concurrently.

## Required invariants

Partition changes must preserve:

- deterministic fingerprints across supported thread counts;
- exact source-ID mappings;
- ownership and ghost coverage;
- valid compact CSR offsets and local slots;
- byte-identical streamed and in-memory package output;
- complete runtime-traffic and capacity accounting.

`make check-partition` is the authoritative integration gate for these
contracts. Partition experiments and evidence are not part of the primary
kernel-speedup evaluation unless explicitly selected.
