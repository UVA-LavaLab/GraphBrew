# Reordering Figure Catalog

Each algorithm ID has a consistent transformation figure and an editable
`.drawio` source that Lucidchart can import as shapes and connectors.

See [Editable Figure Sources](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/README.md) for the Lucidchart workflow.

[Download the 17-page editable bundle](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/GraphBrew-reordering-figures.drawio).

> These are mechanism illustrations on one fixed toy topology, not mapping
> fingerprints from benchmark graphs. Tie-breaking, schedule-sensitive
> algorithms, variants, and external mapping files can produce different
> exact permutations.

## 0. ORIGINAL

![ORIGINAL transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/00-original.svg)

- **CLI:** `-o 0`
- **Mechanism:** Keep the input labels unchanged.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/00-original.drawio)

## 1. RANDOM

![RANDOM transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/01-random.svg)

- **CLI:** `-o 1`
- **Mechanism:** Apply the fixed seed-0 shuffled control.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/01-random.drawio)

## 2. SORT

![SORT transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/02-sort.svg)

- **CLI:** `-o 2`
- **Mechanism:** Sort every vertex by descending degree.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/02-sort.drawio)

## 3. HUBSORT

![HUBSORT transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/03-hubsort.svg)

- **CLI:** `-o 3`
- **Mechanism:** Sort the hub subset; preserve non-hub IDs when possible.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/03-hubsort.drawio)

## 4. HUBCLUSTER

![HUBCLUSTER transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/04-hubcluster.svg)

- **CLI:** `-o 4`
- **Mechanism:** Place stable hubs first and retain non-hub order.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/04-hubcluster.drawio)

## 5. DBG

![DBG transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/05-dbg.svg)

- **CLI:** `-o 5`
- **Mechanism:** Group vertices into logarithmic degree buckets.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/05-dbg.drawio)

## 6. HUBSORTDBG

![HUBSORTDBG transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/06-hubsortdbg.svg)

- **CLI:** `-o 6`
- **Mechanism:** Compact hubs first and sort the hub bucket.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/06-hubsortdbg.drawio)

## 7. HUBCLUSTERDBG

![HUBCLUSTERDBG transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/07-hubclusterdbg.svg)

- **CLI:** `-o 7`
- **Mechanism:** Compact stable hubs first and stable non-hubs second.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/07-hubclusterdbg.drawio)

## 8. RABBITORDER

![RABBITORDER transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/08-rabbitorder.svg)

- **CLI:** `-o 8:csr`
- **Mechanism:** Detect communities and emit dendrogram DFS order.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/08-rabbitorder.drawio)

## 9. GORDER

![GORDER transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/09-gorder.svg)

- **CLI:** `-o 9:csr`
- **Mechanism:** Greedily maximize neighbor overlap in a sliding window.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/09-gorder.drawio)

## 10. CORDER

![CORDER transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/10-corder.svg)

- **CLI:** `-o 10:canonical`
- **Mechanism:** Partition the output into hot and cold workload segments.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/10-corder.drawio)

## 11. RCM

![RCM transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/11-rcm.svg)

- **CLI:** `-o 11:bnf`
- **Mechanism:** Use a peripheral BFS order and reverse it to reduce bandwidth.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/11-rcm.drawio)

## 12. GraphBrewOrder

![GraphBrewOrder transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/12-graphbreworder.svg)

- **CLI:** `-o 12:<recipe>`
- **Mechanism:** Compose partitioner, block layout, and local vertex layout.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/12-graphbreworder.drawio)

## 13. MAP

![MAP transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/13-map.svg)

- **CLI:** `-o 13:<file>`
- **Mechanism:** Load and apply an external .lo or .so permutation.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/13-map.drawio)

## 14. AdaptiveOrder

![AdaptiveOrder transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/14-adaptiveorder.svg)

- **CLI:** `-o 14:<policy>`
- **Mechanism:** Select one validated reordering arm, then execute that arm.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/14-adaptiveorder.drawio)

## 15. LeidenOrder

![LeidenOrder transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/15-leidenorder.svg)

- **CLI:** `-o 15:<layout>`
- **Mechanism:** Detect Leiden communities and apply an explicit post-layout.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/15-leidenorder.drawio)

## 16. GoGraphOrder

![GoGraphOrder transformation](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/reordering/16-gographorder.svg)

- **CLI:** `-o 16`
- **Mechanism:** Reassign IDs to increase directed edges with src < dst.
- [Editable `.drawio` source](https://github.com/UVA-LavaLab/GraphBrew/blob/main/docs/figures/editable/16-gographorder.drawio)
