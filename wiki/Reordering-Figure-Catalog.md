# Reordering Figure Catalog

Every strip uses the same measured 9-vertex input and the same converter
binary. The shared input is shown once; each algorithm then shows only its
measured output order and the mechanism that produced it.

[![Shared catalog input](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/example-input.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/example-input.svg)

**Shared-example contract.** All 17 outputs preserve the same nine vertices
and twelve undirected edges. The blue-outlined cell is the vertex with the
largest displacement in that measured output.

Converter SHA256: `70580d71854e222d299c488274feced6cdd8005667bc87b07fc326c3d6fc5d41`.

[Capture receipt](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/catalog-capture.json) with commands, mapping fingerprints, and raw stdout tails.

Re-capture and regenerate:

```bash
python3 scripts/generate_public_figures.py --capture-catalog
python3 scripts/generate_public_figures.py --check
```

[Download the generated 18-page draw.io bundle](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/GraphBrew-reordering-figures.drawio).

On a narrow screen, select any figure to open the full-resolution SVG.

## Baselines

### 0. ORIGINAL

- **CLI:** `0`
- **Mechanism:** Identity permutation.
- **Evidence:** output order captured from the shared example

[![ORIGINAL measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/00-original.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/00-original.svg)

**Figure.** Topology and memory order are identical.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/00-original.drawio).

### 1. RANDOM

- **CLI:** `1`
- **Mechanism:** Fixed seed-0 shuffle.
- **Evidence:** output order captured from the shared example

[![RANDOM measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/01-random.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/01-random.svg)

**Figure.** Only labels change; the topology is fixed.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/01-random.drawio).

## Degree and bucket layouts

### 2. SORT

- **CLI:** `2`
- **Mechanism:** Global degree sort.
- **Evidence:** output order captured from the shared example

[![SORT measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/02-sort.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/02-sort.svg)

**Figure.** High-degree vertices move toward the first memory region.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/02-sort.drawio).

### 3. HUBSORT

- **CLI:** `3`
- **Mechanism:** Sort selected hubs.
- **Evidence:** output order captured from the shared example

[![HUBSORT measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/03-hubsort.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/03-hubsort.svg)

**Figure.** Only the hub region receives a full degree sort.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/03-hubsort.drawio).

### 4. HUBCLUSTER

- **CLI:** `4`
- **Mechanism:** Stable hub clustering.
- **Evidence:** output order captured from the shared example

[![HUBCLUSTER measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/04-hubcluster.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/04-hubcluster.svg)

**Figure.** The hub/non-hub split changes regions without sorting every vertex.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/04-hubcluster.drawio).

### 5. DBG

- **CLI:** `5`
- **Mechanism:** Degree buckets.
- **Evidence:** output order captured from the shared example

[![DBG measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/05-dbg.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/05-dbg.svg)

**Figure.** Bucket boundaries, not one global sort, define the output.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/05-dbg.drawio).

### 6. HUBSORTDBG

- **CLI:** `6`
- **Mechanism:** HubSort + buckets.
- **Evidence:** output order captured from the shared example

[![HUBSORTDBG measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/06-hubsortdbg.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/06-hubsortdbg.svg)

**Figure.** A sorted hub bucket is followed by grouped non-hubs.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/06-hubsortdbg.drawio).

### 7. HUBCLUSTERDBG

- **CLI:** `7`
- **Mechanism:** Stable hub/non-hub buckets.
- **Evidence:** output order captured from the shared example

[![HUBCLUSTERDBG measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/07-hubclusterdbg.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/07-hubclusterdbg.svg)

**Figure.** Both regions preserve encounter order.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/07-hubclusterdbg.drawio).

## Community and locality layouts

### 8. RABBITORDER

- **CLI:** `8:csr`
- **Mechanism:** Community merge + DFS.
- **Evidence:** output order captured from the shared example

[![RABBITORDER measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/08-rabbitorder.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/08-rabbitorder.svg)

**Figure.** Community blocks become contiguous; DFS chooses hierarchy order.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/08-rabbitorder.drawio).

### 9. GORDER

- **CLI:** `9:csr`
- **Mechanism:** Standalone GORDER_csr.
- **Evidence:** output order captured from the shared example

[![GORDER measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/09-gorder.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/09-gorder.svg)

**Figure.** This is distinct from GraphBrew's relaxed local intra_gorder.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/09-gorder.drawio).

### 10. CORDER

- **CLI:** `10:canonical`
- **Mechanism:** Canonical hot/cold.
- **Evidence:** output order captured from the shared example

[![CORDER measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/10-corder.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/10-corder.svg)

**Figure.** The workload segmentation defines the memory regions.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/10-corder.drawio).

### 11. RCM

- **CLI:** `11:bnf`
- **Mechanism:** BNF + RCM.
- **Evidence:** output order captured from the shared example

[![RCM measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/11-rcm.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/11-rcm.svg)

**Figure.** The output targets lower graph bandwidth.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/11-rcm.drawio).

### 12. GraphBrewOrder

- **CLI:** `12:leiden:compose:sg_none:comm_size_desc:intra_gorder:gw8:gordf4:cd_serial:refine_none`
- **Mechanism:** Explicit three-axis compose.
- **Evidence:** output order captured from the shared example

[![GraphBrewOrder measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/12-graphbreworder.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/12-graphbreworder.svg)

**Figure.** GraphBrew emits one explicit composition, not a competitor fallback.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/12-graphbreworder.drawio).

## External and selected layouts

### 13. MAP

- **CLI:** `13:graphbrew-running-example.lo`
- **Mechanism:** External label list.
- **Evidence:** output order captured from the shared example

[![MAP measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/13-map.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/13-map.svg)

**Figure.** MAP materializes a supplied order; it does not discover one.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/13-map.drawio).

### 14. AdaptiveOrder

- **CLI:** `14:<policy>`
- **Mechanism:** Policy-selected arm.
- **Evidence:** selected-arm illustration; AdaptiveOrder has no fixed permutation

[![AdaptiveOrder measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/14-adaptiveorder.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/14-adaptiveorder.svg)

**Figure.** The output shown is the selected GraphBrew arm for this illustration.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/14-adaptiveorder.drawio).

### 15. LeidenOrder

- **CLI:** `15`
- **Mechanism:** Leiden + post-layout.
- **Evidence:** output order captured from the shared example

[![LeidenOrder measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/15-leidenorder.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/15-leidenorder.svg)

**Figure.** Community detection alone is not an ordering.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/15-leidenorder.drawio).

## Directed layout

### 16. GoGraphOrder

- **CLI:** `16`
- **Mechanism:** Directed forward-edge.
- **Evidence:** output order captured from the shared example

[![GoGraphOrder measured output](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/16-gographorder.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/16-gographorder.svg)

**Figure.** The objective is directed; the shown order is measured on this edge list.

[Editable draw.io source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/16-gographorder.drawio).
