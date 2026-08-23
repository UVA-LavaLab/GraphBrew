# GraphBrew

GraphBrew is a framework for **composable vertex reordering**. It separates
the partitioner, community-block layout, and within-block vertex layout so
their cost and locality effects can be measured independently.

[![GraphBrew infrastructure and paper direction](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg?v=graphbrew-public-v3)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg?v=graphbrew-public-v3)

## Infrastructure, validated policy, and paper direction

| Layer | Role |
|---|---|
| **GraphBrew infrastructure** | explicit composition grammar, mapping generation, provenance, kernel evaluation, and reusable experiment orchestration |
| **Validated policy today** | the frozen reuse-1/2 selector; its fallback branch runs Boost Rabbit |
| **Paper research direction** | a deterministic Rabbit-free composition generator selected from graph, kernel, reuse, and cost semantics |

The infrastructure is the reusable system. Rabbit-free automatic composition
is the paper direction being evaluated on top of it.

## Read the project in this order

| Step | Page | Question answered |
|---:|---|---|
| 1 | [GraphBrew Running Example](GraphBrew-Running-Example) | How does one graph move from CSR through partition, layout, relabeling, and locality? |
| 2 | [GraphBrewOrder](GraphBrewOrder) | What does each production token do? |
| 3 | [All-Kernel Low-Reuse Selector](All-Kernel-Low-Reuse-Selector) | How is a new graph classified without a previous kernel run? |
| 4 | [Reordering Figure Catalog](Reordering-Figure-Catalog) | How do all 17 measured output orders compare on one input? |
| 5 | [Reproducible Experiments](Reproducible-Experiments) | How are mappings, kernels, reuse, and evidence measured? |

## Two interfaces

| Interface | Use |
|---|---|
| `-o 12:<configuration>` | Run one exact, hand-configured composition. No runtime search occurs. |
| `-o 14:_:_:_:allkernel-lowreuse-rule:best-endtoend:<reuse>` | Apply the frozen reuse-1/2 rule and choose GraphBrew or Boost Rabbit. |

## Claim boundary

- The promoted GraphBrew arm is non-Rabbit and beat Boost Rabbit on all seven
  final holdouts where the frozen rule selected it.
- The fallback branch runs Boost Rabbit and therefore ties, rather than beats,
  the always-Boost baseline on those graphs.
- The complete policy is a winning portfolio, but it is not Rabbit-free.
- Fully deployed timing must include algorithm-14 feature extraction in
  addition to chosen mapping and kernel time.

## Quick links

- [AdaptiveOrder](AdaptiveOrder)
- [Getting Started](Getting-Started)
- [Running Benchmarks](Running-Benchmarks)
- [Command-Line Reference](Command-Line-Reference)
- [Supported Graph Formats](Supported-Graph-Formats)
- [Graph Benchmarks](Graph-Benchmarks)
- [Reordering Figure Catalog](Reordering-Figure-Catalog)
- [Cache Simulation](Cache-Simulation)
- [Code Architecture](Code-Architecture)
- [Troubleshooting](Troubleshooting)
- [FAQ](FAQ)
- [Contributing](Contributing)

Repository: https://github.com/UVA-LavaLab/GraphBrew
