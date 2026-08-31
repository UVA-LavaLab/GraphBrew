# GraphBrew

GraphBrew is a framework for **composable and explainable vertex layouts**.

> Vertex reordering is not one monolithic algorithm choice; it is a
> kernel-dependent layout-composition problem.

[![GraphBrew architecture](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg?v=graphbrew-public-v4)

## What the paper establishes

| Contribution | Result |
|---|---|
| Compositional layout model | An executable `<P,B,L>` expression partitions vertices, places blocks, and orders vertices inside each block before producing one persistent permutation |
| Kernel-specific mechanism evidence | All seven sealed layouts win cells; fixed membership attributes LocalGorder8’s BFS gain to per-edge locality and its Afforest CC gain to fewer compression steps |
| Practical layout and construction | LeidenGVE–SizeDesc–LocalGorder8 is 1.052x faster and 0.752x as expensive to map as GORDER_csr; Compact-and-Emit preserves a selected permutation while reducing construction work |

These are bounded claims. GraphBrew does **not** claim a universal ordering,
a Rabbit-cost-balanced arm, or a graph-held-out automatic generator. Rabbit
remains the low-overhead Pareto boundary.

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
