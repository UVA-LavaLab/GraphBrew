# Symbols → Files Map

## Partitioning
| Symbol | File | Notes |
|--------|------|-------|
| `MakeCagraPartitionedGraph` | `bench/include/graphbrew/partition/cagra/popt.h` | Cagra/GraphIT CSR slicing, uses `graphSlicer`, honors `outDegree` (CLI `-z`) |
| `graphSlicer` | `bench/include/graphbrew/partition/cagra/popt.h` | Slice CSR by vertex range |
| `TrustPartitioner::MakeTrustPartitionedGraph` | `bench/include/graphbrew/partition/trust.h` | TRUST partitioning for triangle counting |

## Reordering
| Symbol | File | Notes |
|--------|------|-------|
| `GenerateGraphBrewMappingUnified` | `bench/include/graphbrew/reorder/reorder_graphbrew.h` | GraphBrew clustering + final reorder |
| `GenerateAdaptiveMapping` | `bench/include/graphbrew/reorder/reorder_adaptive.h` | AdaptiveOrder selector |
| `GenerateRabbitOrderCSRMapping` | `bench/include/graphbrew/reorder/reorder_rabbit.h` | Native CSR RabbitOrder |
| `GenerateLeidenCSRMapping` | `bench/include/graphbrew/reorder/reorder_leiden.h` | GVE-Leiden CSR variants |

## Builder Entrypoints
| Symbol | File | Notes |
|--------|------|-------|
| `BuilderBase::MakePartitionedGraph` | `bench/include/graphbrew/builder.h` | Dispatches `-j` to Cagra or TRUST |
## Umbrella
| Symbol | File | Notes |
|--------|------|-------|
| `graphbrew.h` | `bench/include/graphbrew/graphbrew.h` | Umbrella: builder, reorder, partitioning, cache_sim |
| `BuilderBase::GenerateMapping` | `bench/include/graphbrew/builder.h` | Reordering dispatcher (IDs 0–17) |

## Cache Simulation
| Symbol | File | Notes |
|--------|------|-------|
| `cache_sim.h` | `bench/include/cache_sim/cache_sim.h` | Cache simulation core |
| `graph_sim.h` | `bench/include/cache_sim/graph_sim.h` | Graph wrappers for cache sim |

---
*See `docs/INDEX.md` for directory map and `wiki/Command-Line-Reference.md` for CLI.*
