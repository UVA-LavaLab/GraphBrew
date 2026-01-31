# Symbols → Files Map

## Partitioning
| Symbol | File | Notes |
|--------|------|-------|
| `MakeCagraPartitionedGraph` | `bench/include/gapbs/cache/popt.h` | Cagra/GraphIT CSR slicing, uses `graphSlicer`, honors `outDegree` (CLI `-z`) |
| `graphSlicer` | `bench/include/gapbs/cache/popt.h` | Slice CSR by vertex range |
| `TrustPartitioner::MakeTrustPartitionedGraph` | `bench/include/gapbs/partition/trust.h` | TRUST partitioning for triangle counting |

## Reordering
| Symbol | File | Notes |
|--------|------|-------|
| `GenerateGraphBrewMappingUnified` | `bench/include/gapbs/reorder/reorder_graphbrew.h` | GraphBrew clustering + final reorder |
| `GenerateAdaptiveMapping` | `bench/include/gapbs/reorder/reorder_adaptive.h` | AdaptiveOrder selector |
| `GenerateRabbitOrderCSRMapping` | `bench/include/gapbs/reorder/reorder_rabbit.h` | Native CSR RabbitOrder |
| `GenerateLeidenCSRMapping` | `bench/include/gapbs/reorder/reorder_leiden.h` | GVE-Leiden CSR variants |

## Builder Entrypoints
| Symbol | File | Notes |
|--------|------|-------|
| `BuilderBase::MakePartitionedGraph` | `bench/include/gapbs/builder.h` | Dispatches `-j` to Cagra or TRUST |
| `BuilderBase::GenerateMapping` | `bench/include/gapbs/builder.h` | Reordering dispatcher (IDs 0–17) |

## Cache Simulation
| Symbol | File | Notes |
|--------|------|-------|
| `cache_sim.h` | `bench/include/cache/cache_sim.h` | Cache simulation core |
| `graph_sim.h` | `bench/include/cache/graph_sim.h` | Graph wrappers for cache sim |

---
*See `docs/INDEX.md` for directory map and `wiki/Command-Line-Reference.md` for CLI.*
