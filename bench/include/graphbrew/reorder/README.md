# reorder/ — Reordering Algorithms

All graph reordering implementations and dispatch infrastructure.

## Headers

| Header | Algo IDs | Description |
|--------|----------|-------------|
| `reorder.h` | — | Main dispatcher, `resolveVariant()`, `warnUnknownVariant()`, `hasVariants()` |
| `reorder_types.h` | — | Enums, shared features, variant resolution, and retained legacy model types |
| `reorder_basic.h` | 0–2 | ORIGINAL, Random, Sort |
| `reorder_hub.h` | 3–7 | HubSort, HubCluster, DBG, HubSortDBG, HubClusterDBG |
| `reorder_classic.h` | 10 | COrder |
| `reorder_rabbit.h` | 8 | RabbitOrder CSR (`8:csr`, auto-adaptive resolution) + Boost (`8:boost`, reference) |
| `reorder_gorder.h` | 9 | GOrder CSR (`9:csr`) + parallel batch (`9:fast`) |
| `reorder_rcm.h` | 11 | RCM default + BNF variant (`11:bnf`) |
| `reorder_graphbrew.h` | 12 | GraphBrewOrder — Leiden + per-community reordering pipeline |
| `reorder_graphbrew_diagnostics.h` | 12 | Callable diagnostic ordering families used by GraphBrew |
| `reorder_graphbrew_parser.h` | 12 | GraphBrew token parser |
| `reorder_adaptive.h` | 14 | Frozen deterministic selector plus retained offline-model modes |
| `reorder_gograph.h` | 16 | GoGraphOrder directed forward-edge diagnostic |

## Variant Dispatch Flow

```
CLI -o 9:fast
  → command_line.h splits on ':'  →  algo=9, params=["fast"]
  → builder.h::GenerateMapping()  →  resolveVariant(params) → "fast"
  → GenerateGOrderFastMapping()
```

Unknown variants print a warning and fall back to the default implementation.
