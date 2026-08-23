# GraphBrew

GraphBrew is a framework for **composable vertex reordering**. It separates
the partitioner, community-block layout, and within-block vertex layout so
their cost and locality effects can be measured independently.

[![GraphBrew current and proposed architecture](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg)](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg)

## Current versus proposed

| Status | Path |
|---|---|
| **Validated today** | explicit compositions and the frozen reuse-1/2 selector; its fallback branch runs Boost Rabbit |
| **Research direction** | a deterministic Rabbit-free composition generator selected from graph, kernel, reuse, and cost semantics |

The architecture figure keeps those claims separate while carrying one tracked
vertex through the same six-stage ordering pipeline.

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
