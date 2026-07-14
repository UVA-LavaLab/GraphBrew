# reorder/ — Reordering Algorithms

All graph reordering implementations and dispatch infrastructure.

## Headers

| Header | Algo IDs | Description |
|--------|----------|-------------|
| `reorder.h` | — | Main dispatcher, `resolveVariant()`, `warnUnknownVariant()`, `hasVariants()` |
| `reorder_types.h` | — | Enums, `PerceptronWeights`, `ResolveVariantSelection()`, weight tables |
| `reorder_basic.h` | 0, 1, 7 | ORIGINAL, Sort, Random |
| `reorder_hub.h` | 2–6 | HubSort, HubCluster, DBG, HubSortDBG, HubClusterDBG |
| `reorder_classic.h` | 10 | COrder |
| `reorder_rabbit.h` | 8 | RabbitOrder CSR (`8:csr`, auto-adaptive resolution) + Boost (`8:boost`, reference) |
| `reorder_gorder.h` | 9 | GOrder CSR (`9:csr`) + parallel batch (`9:fast`) |
| `reorder_rcm.h` | 11 | RCM default + BNF variant (`11:bnf`) |
| `reorder_graphbrew.h` | 12 | GraphBrewOrder — Leiden + per-community reordering pipeline |
| `reorder_adaptive.h` | 14 | AdaptiveOrder — perceptron-based algorithm selection |

## Variant Dispatch Flow

```
CLI -o 9:fast
  → command_line.h splits on ':'  →  algo=9, params=["fast"]
  → builder.h::GenerateMapping()  →  resolveVariant(params) → "fast"
  → GenerateGOrderFastMapping()
```

Unknown variants print a warning and fall back to the default implementation.
