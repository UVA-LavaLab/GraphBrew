# Include Layout

```
bench/include/
├── graphbrew/            # Core GraphBrew runtime (builder, reorder, partition, util)
│   ├── builder.h
│   ├── graphbrew.h       # Umbrella
│   ├── reorder/          # Reordering algorithms
│   ├── partition/
│   │   ├── trust.h       # TRUST partitioning
│   │   └── cagra/popt.h  # Cagra/GraphIT partition helpers
│   └── ... (core headers)
├── cache_sim/            # Cache simulation headers
│   ├── cache_sim.h
│   └── graph_sim.h       # includes ../graphbrew/graph.h
└── external/             # External modules bundled
    ├── rabbit/
    ├── gorder/
    ├── corder/
    └── leiden/
```

## Notes
- Include path variables (Makefile): `INCLUDE_GRAPHBREW`, `INCLUDE_EXTERNAL`, `INCLUDE_*` for each external module.
- Legacy `gapbs/` and `partitioning/` are removed; use `graphbrew/` and `partition/cagra/`.
- Lint: `make lint-includes` to catch legacy include paths.
