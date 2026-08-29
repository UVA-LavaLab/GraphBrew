# GraphBrew

GraphBrew is a framework for composing vertex reorderings from three explicit
decisions: **partitioner**, **block layout**, and **vertex layout**.

[![GraphBrew architecture](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg?v=graphbrew-public-v4)

## What the paper establishes

| Contribution | Result |
|---|---|
| GORDER-quality replacement point | LeidenGVE–SizeDesc–LocalGorder8 is 1.052x faster in kernel GM, maps at 0.752x GORDER_csr cost, and wins end to end from reuse one |
| Rabbit Pareto boundary | The 4% per-cell GM requires 17–19x mapping cost; without Afforest CC the margin disappears, and summed seconds never win |
| Compositional diversity | Eight of nine arms win cells; the cell oracle is 1.122x faster than the best fixed GraphBrew arm, but graph-held-out family/kernel selection reaches only 0.911x |
| Compact-and-Emit | Preserves the BFS permutation while removing sparse community scheduling and final-emission work |

These are bounded claims. GraphBrew does **not** claim a universal ordering,
a Rabbit-cost-balanced arm, or a graph-held-out automatic generator.

See [Evidence and Claims](Evidence-and-Claims) for the exact scope.

## Read in this order

1. [Evidence and Claims](Evidence-and-Claims)
2. [GraphBrew Running Example](GraphBrew-Running-Example)
3. [GraphBrewOrder](GraphBrewOrder)
4. [Reordering Algorithms](Reordering-Algorithms)
5. [Reproducible Experiments](Reproducible-Experiments)

## Interfaces

| Interface | Role |
|---|---|
| `-o 12:<configuration>` | run one explicit composition |
| `-o 13:<mapping>` | apply a pre-generated permutation |
| `-o 14` | experimental compatibility selector; not a headline result |

Repository: https://github.com/UVA-LavaLab/GraphBrew
