# GraphBrew

GraphBrew is a research and deployment framework for **composable graph
reordering**. It separates three decisions that monolithic ordering names
usually combine:

1. **Partitioner** — discover vertex groups.
2. **Block layout** — place groups in the global ID space.
3. **Vertex layout** — order vertices inside each block.

![GraphBrew architecture](https://raw.githubusercontent.com/UVA-LavaLab/GraphBrew/main/docs/figures/graphbrew-architecture.svg)

## Two ways to use GraphBrew

| Interface | Meaning |
|---|---|
| `-o 12:<configuration>` | Explicit, hand-configured composition for controlled experiments or known workloads. |
| `-o 14:_:_:_:allkernel-lowreuse-rule:best-endtoend:<reuse>` | Frozen deterministic reuse-1/2 policy. This is the validated automatic path and is **not ML**. |

The runtime policy uses cheap graph statistics and declared reuse to choose
the promoted FastLeiden-Gorder8 composition or Boost Rabbit. It does not train
at runtime, use graph names, query an oracle database, or trial multiple
orderings.

## Start here

- [Getting Started](Getting-Started)
- [GraphBrewOrder](GraphBrewOrder) — composition axes and exact configuration
- [AdaptiveOrder](AdaptiveOrder) — deterministic runtime policy and scope
- [All-Kernel Low-Reuse Selector](All-Kernel-Low-Reuse-Selector) — mechanism,
  figure, frozen rule, and evidence
- [Reordering Algorithms](Reordering-Algorithms) — baselines and measured
  guidance
- [Running Benchmarks](Running-Benchmarks)
- [Reproducible Experiments](Reproducible-Experiments)

## Reference

- [Command-Line Reference](Command-Line-Reference)
- [Supported Graph Formats](Supported-Graph-Formats)
- [Graph Benchmarks](Graph-Benchmarks)
- [Cache Simulation](Cache-Simulation)
- [Code Architecture](Code-Architecture)
- [Troubleshooting](Troubleshooting)
- [FAQ](FAQ)

## Development

- [Contributing](Contributing)
- [Python Scripts](Python-Scripts)
- [Partitioning and Shards](Partitioning-and-Shards)

Repository: https://github.com/UVA-LavaLab/GraphBrew
