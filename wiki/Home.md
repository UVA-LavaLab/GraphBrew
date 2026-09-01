# GraphBrew

GraphBrew is a framework for **composable and explainable vertex layouts**.
It separates vertex grouping, block placement, and within-block ordering,
then compiles those choices into one persistent permutation.

[![GraphBrew architecture](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg?v=graphbrew-public-v4)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg?v=graphbrew-public-v4)

## Core model

| Stage | Question |
|---|---|
| Partitioner `P` | Which vertices belong in the same block? |
| Block layout `B` | In what order should the blocks appear? |
| Vertex layout `L` | How should vertices be ordered inside each block? |

The resulting ID is:

```text
pi(v) = block_offset(B(P(v))) + L[P(v)](v)
```

GraphBrew records requested and realized configurations, mapping
fingerprints, construction cost, CSR relocation cost, kernel time, and
verification state.

## Read in this order

1. [Getting Started](Getting-Started)
2. [GraphBrew Running Example](GraphBrew-Running-Example)
3. [GraphBrewOrder](GraphBrewOrder)
4. [Reordering Algorithms](Reordering-Algorithms)
5. [Running Benchmarks](Running-Benchmarks)
6. [Reproducible Experiments](Reproducible-Experiments)

## Interfaces

| Interface | Role |
|---|---|
| `-o 12:<configuration>` | run one explicit composition |
| `-o 13:<mapping>` | validate and apply a pre-generated permutation |
| `-o 14:<policy>` | use the experimental policy-dispatch interface |

Repository: https://github.com/UVA-LavaLab/GraphBrew
