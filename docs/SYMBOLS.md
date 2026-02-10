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
| `GenerateGraphBrewMappingUnified` | `bench/include/external/gapbs/builder.h` | GraphBrew clustering + final reorder (GraphBrew-powered) |
| `GenerateAdaptiveMapping` | `bench/include/graphbrew/reorder/reorder_adaptive.h` | AdaptiveOrder selector |
| `GenerateRabbitOrderCSRMapping` | `bench/include/graphbrew/reorder/reorder_rabbit.h` | Native CSR RabbitOrder |
| ~~`GenerateLeidenCSRMapping`~~ | ~~`reorder_leiden.h`~~ | _Removed — LeidenCSR deprecated; use GraphBrewOrder (12)_ |

## Builder Entrypoints
| Symbol | File | Notes |
|--------|------|-------|
| `BuilderBase::MakePartitionedGraph` | `bench/include/external/gapbs/builder.h` | Dispatches `-j` to Cagra or TRUST |
| `BuilderBase::GenerateMapping` | `bench/include/external/gapbs/builder.h` | Reordering dispatcher (IDs 0–16; 16 deprecated) |

## Umbrella
| Symbol | File | Notes |
|--------|------|-------|
| `graphbrew.h` | `bench/include/graphbrew/graphbrew.h` | Umbrella: builder, reorder, partitioning, cache_sim |

## Cache Simulation
| Symbol | File | Notes |
|--------|------|-------|
| `cache_sim.h` | `bench/include/cache_sim/cache_sim.h` | Cache simulation core |
| `graph_sim.h` | `bench/include/cache_sim/graph_sim.h` | Graph wrappers for cache sim |

---
*See `docs/INDEX.md` for directory map and `wiki/Command-Line-Reference.md` for CLI.*
