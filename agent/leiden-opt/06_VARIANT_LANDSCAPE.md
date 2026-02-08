# Variant Landscape — Every Leiden/VIBE Variant

Complete map of the variant space. Know this before exploring.

---

## Algorithm IDs (CLI: `-o <ID>`)

| ID | Name | Relevance |
|:---:|------|-----------|
| 8 | RabbitOrder | **The competitor** — Louvain + parallel aggregation |
| 12 | GraphBrewOrder | Cluster + per-community reorder (uses Leiden or Rabbit for clustering) |
| 15 | LeidenOrder | Classic Leiden via igraph (slow, reference only) |
| 16 | LeidenDendrogram | Hierarchical Leiden with tree traversal |
| 17 | **LeidenCSR** | **Fast GVE-Leiden on native CSR — main variant family** |

---

## LeidenCSR Variants (Algorithm 17)

CLI format: `-o 17:variant:resolution:iterations:passes`

| Variant | CLI | Description | Speed | Quality |
|---------|-----|-------------|:---:|:---:|
| GVE | `gve` | Standard GVE-Leiden | Fast | Good |
| GVE2 | `gve2` | Double-buffered super-graph | Fast | Good |
| GVEOpt | `gveopt` | Cache-optimized + prefetching | Fast | Good |
| **GVEOpt2** | **`gveopt2`** | **CSR aggregation (default)** | **Fast** | **Good** |
| GVERabbit | `gverabbit` | GVE + RabbitOrder within communities | Moderate | High |
| DFS | `dfs` | DFS ordering of community tree | Fast | Medium |
| BFS | `bfs` | BFS ordering of community tree | Fast | Medium |
| HubSort | `hubsort` | Hub-first within communities | Fast | Medium |
| Fast | `fast` | Speed-optimised (fewer iterations) | Very Fast | Lower |
| Faithful | `faithful` | 1:1 leiden.hxx reference | Slow | Reference |

---

## VIBE Variants (Modular Framework)

CLI format: `--csr-variants vibe:<sub-variant>`

### Core Variants
| Variant | CLI | Algorithm | Aggregation | Ordering |
|---------|-----|-----------|-------------|----------|
| VIBE (default) | `vibe` | Leiden | Leiden CSR | BFS connectivity |
| VIBE DFS | `vibe:dfs` | Leiden | Leiden CSR | Dendrogram DFS |
| VIBE BFS | `vibe:bfs` | Leiden | Leiden CSR | Dendrogram BFS |
| VIBE Rabbit | `vibe:rabbit` | RabbitOrder | Lazy incremental | Hierarchical |
| VIBE Streaming | `vibe:streaming` | Leiden | Lazy (RabbitOrder-style) | Hierarchical |
| VIBE LazyUpdate | `vibe:lazyupdate` | Leiden | Leiden CSR | BFS (batched weight updates) |

### Hybrid Variants (Best Performers)
| Variant | CLI | Description |
|---------|-----|-------------|
| **HRAB** | **`vibe:hrab`** | **Leiden communities → RabbitOrder super-graph → BFS intra** |
| **HRAB+Gordi** | **`vibe:hrab:gordi`** | **Same + Gorder-greedy intra-community (highest locality)** |
| TQR | `vibe:tqr` | Tile-quantized RabbitOrder: cache-line-aligned macro-ordering |

### Intra-Community Ordering Variants
| Variant | CLI | Description |
|---------|-----|-------------|
| DBG | `vibe:dbg` | Degree-Based Grouping within communities |
| CORDER | `vibe:corder` | Hot/cold partitioning within communities |
| DBG Global | `vibe:dbg-global` | DBG across all vertices post-clustering |
| CORDER Global | `vibe:corder-global` | Hot/cold across all vertices post-clustering |
| Conn | `vibe:conn` | Connectivity BFS within communities |

### Sub-Variants of vibe:rabbit
| Variant | CLI | Description |
|---------|-----|-------------|
| Rabbit+DFS | `vibe:rabbit:dfs` | RabbitOrder + DFS post-ordering |
| Rabbit+BFS | `vibe:rabbit:bfs` | RabbitOrder + BFS post-ordering |
| Rabbit+DBG | `vibe:rabbit:dbg` | RabbitOrder + DBG post-ordering |
| Rabbit+CORDER | `vibe:rabbit:corder` | RabbitOrder + Hot/cold post-ordering |

---

## Tunable Parameters (VibeConfig)

| Parameter | Default | Range | CLI | What It Controls |
|-----------|---------|-------|-----|-----------------|
| `resolution` | 0.75 (auto) | 0.1–2.0 | 3rd field of `-o 17:v:R` | Community granularity: lower = larger, higher = smaller |
| `tolerance` | 1e-2 | 1e-6–1e-1 | — | When to stop local-moving iterations |
| `aggregationTolerance` | 0.8 | 0.5–0.99 | — | Progress threshold to stop aggregation |
| `toleranceDrop` | 10.0 | 2–100 | — | Tolerance tightening per pass |
| `maxIterations` | 10 | 1–100 | 4th field of `-o 17:v:R:I` | Max local-moving iterations per pass |
| `maxPasses` | 10 | 1–50 | 5th field of `-o 17:v:R:I:P` | Max aggregation passes |
| `useRefinement` | true | bool | — | Enable Leiden refinement (the key difference from Louvain) |
| `useDynamicResolution` | false | bool | `dynamic` | Per-pass resolution adjustment |
| `useGorderIntra` | false | bool | `gordi` suffix | Gorder-greedy within communities |
| `gorderWindow` | 5 | 3–20 | — | Sliding window size for Gorder intra |
| `useHubExtraction` | false | bool | `hubx` | Extract hubs before community ordering |
| `hubExtractionPct` | 0.001 | 0.0001–0.01 | `hubx0.5` | % of vertices to extract as hubs |
| `tileSize` | 4096 | 1024–16384 | — | Cache blocking tile size |
| `prefetchDistance` | 8 | 4–32 | — | Prefetch lookahead |
| `useDegreeSorting` | false | bool | — | Process by ascending degree (helps vibe:rabbit) |
| `useCommunityMerging` | false | bool | `merge` | Merge small communities |
| `useLazyUpdates` | false | bool | — | Batch community weight updates |

---

## Resolution Modes

| Mode | CLI Value | Description |
|------|-----------|-------------|
| Auto | `0` or omitted | Computed from graph density/degree CV |
| Fixed | `1.5` | Use exact value, all passes |
| Dynamic | `dynamic` | Auto initial, adjust per pass based on community metrics |
| Dynamic+Init | `dynamic_2.0` | Start at 2.0, then adjust |

---

## LeidenDendrogram Variants (Algorithm 16)

| Variant | CLI | Description |
|---------|-----|-------------|
| DFS | `-o 16:R:dfs` | Depth-first traversal |
| DFSHub | `-o 16:R:dfshub` | DFS with hub vertices first |
| DFSSize | `-o 16:R:dfssize` | DFS with larger subtrees first |
| BFS | `-o 16:R:bfs` | Breadth-first traversal |
| Hybrid | `-o 16:R:hybrid` | Combined DFS-Hub + BFS (default) |

---

## GraphBrewOrder Cluster Variants (Algorithm 12)

| Cluster | CLI | Description |
|---------|-----|-------------|
| Leiden | `-o 12:leiden` | Standard igraph Leiden |
| GVE | `-o 12:gve` | GVE-Leiden (native CSR) |
| GVEOpt | `-o 12:gveopt` | Cache-optimised GVE |
| Rabbit | `-o 12:rabbit` | RabbitOrder's Louvain |
| HubCluster | `-o 12:hubcluster` | HubCluster partitioning |

Fast variants: append `fast` (e.g., `gvefast`, `gveoptfast`) for fewer iterations.

---

## Variant Comparison Quick Reference

### Speed (reorder time, fastest → slowest):
```
vibe:rabbit > leiden:fast > leiden:gveopt2 > vibe:hrab > rabbit:csr > vibe:hrab:gordi > gorder
```

### Locality Quality (cache miss reduction, best → worst):
```
vibe:hrab:gordi > vibe:hrab > rabbit:csr > leiden:gveopt2 > vibe:rabbit > vibe > original
```

### Recommended Starting Points:
- **Speed-constrained:** `vibe:rabbit` (fast, decent quality)
- **Quality-focused:** `vibe:hrab:gordi` (best locality, slower)
- **Balanced:** `vibe:hrab` (good quality, moderate speed)
- **Baseline comparison:** `rabbit:csr` (the competitor to beat)
