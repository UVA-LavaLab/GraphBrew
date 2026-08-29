# GraphBrew Running Example

One nine-vertex graph is carried through the complete composition pipeline:

1. load graph and CSR;
2. attach community membership;
3. place contiguous community blocks;
4. order vertices inside each block;
5. validate and emit the relabeled CSR; and
6. inspect the resulting property-access locality.

The `C0`/`C1` partition is fixed pedagogically so every mapping and CSR value
is checkable. The example illustrates composition; it is not a performance
result or the detector’s measured output on this tiny graph.

## 1. Input graph and CSR

[![Input graph and CSR](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-graph-to-csr.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-graph-to-csr.svg?v=graphbrew-public-v4)

Tracked vertex `v2` reads neighbors `[1,4,6,8]`. No labels have moved.

## 2. Community membership

[![Community membership](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-leiden-transform.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-leiden-transform.svg?v=graphbrew-public-v4)

The example fixes:

```text
C0 = {1,2,4,6,7}
C1 = {0,3,5,8}
```

Topology and vertex IDs remain unchanged.

## 3. Block layout

[![SizeDesc blocks](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-sizedesc-transform.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-sizedesc-transform.svg?v=graphbrew-public-v4)

`comm_size_desc` assigns `C0` to IDs `0..4` and `C1` to IDs `5..8`.

## 4. Vertex layout inside each block

The small fixture uses `gordf4` only to place its two communities on opposite
sides of one local-layout decision:

[![Per-community dispatch](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-gordf5000.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-gordf5000.svg?v=graphbrew-public-v4)

### Relaxed local Gorder

[![Relaxed local Gorder](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-gorder-transform.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-gorder-transform.svg?v=graphbrew-public-v4)

`intra_gorder` is GraphBrew’s historical direct-neighbor local heuristic. It
is distinct from faithful standalone `GORDER_csr`.

### Hub-rooted BFS

[![Hub-rooted BFS](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-bfs-transform.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-bfs-transform.svg?v=graphbrew-public-v4)

For `C0`, `v2` is the highest-degree root and receives local ID zero.

## 5. Relabel and validate CSR

[![Relabeled CSR](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-relabel-emit.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-relabel-emit.svg?v=graphbrew-public-v4)

The final order is:

```text
[v2,v1,v4,v6,v7 | v8,v5,v0,v3]
```

The permutation is bijective and preserves all 24 directed arcs.

### Compact-and-Emit

[![Compact-and-Emit](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-compact-emit.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-compact-emit.svg?v=graphbrew-public-v4)

For one-pass BFS compositions, Compact-and-Emit can build the same selected
permutation while scheduling active community IDs only and writing final IDs
during traversal.

## 6. Locality consequence

[![Locality consequence](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-locality-outcome.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-locality-outcome.svg?v=graphbrew-public-v4)

The tracked neighbor IDs become `[1,2,3,5]`, reducing the illustrative
four-property access from three cache lines to two.

## Exact fixture

| Quantity | Value |
|---|---|
| input order | `[v0,v1,v2,v3,v4,v5,v6,v7,v8]` |
| block-only SizeDesc order | `[v1,v2,v4,v6,v7,v0,v3,v5,v8]` |
| `C0` local BFS order | `[v2,v1,v4,v6,v7]` |
| `C1` relaxed-Gorder order | `[v8,v5,v0,v3]` |
| final order | `[v2,v1,v4,v6,v7,v8,v5,v0,v3]` |
| old-to-new map | `[7,1,0,8,2,6,3,4,5]` |
| relabeled row 0 | `[1,2,3,5]` |

Regenerate every public figure with:

```bash
python3 scripts/generate_public_figures.py
python3 scripts/generate_public_figures.py --check
```
