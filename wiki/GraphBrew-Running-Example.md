# GraphBrew Running Example

These figures share one stage map: **1** load and profile, **2** partition,
**3** block layout, **4** vertex layout, **5** relabeled-CSR emission, and
**6** kernel access pattern. Serialization, validation, and timing are shown
without new stage numbers.

On a narrow screen, select a figure to open its full-resolution SVG.

Every stage uses the same nine vertices, twelve undirected edges, communities
`C0`/`C1`, and tracked vertex `v2`. The example uses `gordf4` only so both
local-layout branches fit in one small figure. The evaluated composition uses
`gordf5000`.

The `C0`/`C1` membership is manually frozen after Stage 2 so every subsequent
mapping and CSR value is arithmetically checkable. It is not the community
detector's measured output for this tiny graph. The Algorithm-12 catalog strip
separately reports the actual end-to-end converter output for the same CLI.

## 1. Load the graph and expose one CSR row

[![Stage 1 graph to CSR](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-graph-to-csr.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-graph-to-csr.svg)

**Figure 1.** The topology, CSR row, and lightweight profile all refer to
`v2`. No labels move and no graph kernel runs.

The next stage adds community membership to this exact graph.

## 2. Partition without relabeling

[![Stage 2 partition](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-leiden-transform.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-leiden-transform.svg)

**Figure 2.** The pedagogical partition fixes
`C0={1,2,4,6,7}` and `C1={0,3,5,8}`. Vertex IDs and all twelve edges remain
unchanged.

The next stage converts those two sets into contiguous ID ranges.

## 3. Place community blocks

[![Stage 3 SizeDesc](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-sizedesc-transform.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-sizedesc-transform.svg)

**Figure 3.** `comm_size_desc` places five-vertex `C0` in IDs `0..4` and
four-vertex `C1` in IDs `5..8`. Tracked `v2` enters the first block.

The block ranges are now fixed. Stage 4 chooses an order independently inside
each block.

## 4. Dispatch and order each block

[![Stage 4 dispatch](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-gordf5000.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-gordf5000.svg)

**Figure 4.** The threshold is per community. In this mechanism-scale example,
`C1` takes relaxed local Gorder while `C0` takes hub-rooted BFS.

### 4A. Small block: relaxed local Gorder

[![Stage 4A relaxed local Gorder](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-gorder-transform.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-gorder-transform.svg)

**Figure 4A.** The active `intra_gorder` implementation is the historical
direct-neighbor UnitHeap heuristic. It is not faithful standalone
`GORDER_csr`.

### 4B. Large block: hub-rooted BFS

[![Stage 4B BFS](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-bfs-transform.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-bfs-transform.svg)

**Figure 4B.** `v2` is the highest-degree root. BFS emits
`[v2,v1,v4,v6,v7]`, so `v2` receives local and global ID `0`.

The two local orders now compose with the block order into one permutation.

## 5. Emit and validate relabeled CSR

[![Stage 5 relabeled CSR](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-relabel-emit.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-relabel-emit.svg)

**Figure 5.** The final memory order is
`[v2,v1,v4,v6,v7 | v8,v5,v0,v3]`. The permutation is bijective and the
relabeled CSR preserves all 24 directed arcs.

The last stage follows the same `v2` property reads after relabeling.

## 6. Show the locality consequence

[![Stage 6 locality](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-locality-outcome.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-locality-outcome.svg)

**Figure 6.** `v2` still reads four neighbor properties. Their IDs change from
`[1,4,6,8]` to `[1,2,3,5]`, reducing four-property cache-line touches from
three lines to two.

This is an illustrative locality mechanism, not a performance result. Measured
kernel and end-to-end claims remain in the evidence pages.

## Exact fixture

| Quantity | Value |
|---|---|
| input order | `[v0,v1,v2,v3,v4,v5,v6,v7,v8]` |
| block-only SizeDesc order | `[v1,v2,v4,v6,v7,v0,v3,v5,v8]` |
| C0 local BFS order | `[v2,v1,v4,v6,v7]` |
| C1 relaxed-Gorder order | `[v8,v5,v0,v3]` |
| final order | `[v2,v1,v4,v6,v7,v8,v5,v0,v3]` |
| old-to-new map | `[7,1,0,8,2,6,3,4,5]` |
| relabeled offsets | `[0,4,7,9,12,14,18,20,22,24]` |
| relabeled row 0 | `[1,2,3,5]` |

The machine-readable source is
[`docs/figures/graphbrew-running-example.json`](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/graphbrew-running-example.json).
Regenerate every public explanatory figure with:

```bash
python3 scripts/generate_public_figures.py
python3 scripts/generate_public_figures.py --check
```

## Source map

| Figure content | Implementation source |
|---|---|
| composition grammar | `bench/include/graphbrew/reorder/reorder_graphbrew_parser.h` |
| community detection and composition | `bench/include/graphbrew/reorder/reorder_graphbrew.h` |
| relaxed local Gorder | `intraGorderGreedy` in `reorder_graphbrew.h` |
| hub-rooted BFS | `intraBFSFromHub` in `reorder_graphbrew.h` |
| mapping load/apply and CSR relocation | `bench/include/external/gapbs/builder.h` |
| work and timing validation | `scripts/experiments/vldb/runner.py` |
