# Cache Simulation (bench/include/cache)

Purpose: L1/L2/L3 cache simulation core used by `bench/bin_sim/*`.

Key headers:
- `cache_sim.h` — Cache simulator core, eviction policies
- `graph_sim.h` — Graph wrappers/instrumentation for cache simulation

Not partitioning: For Cagra/GraphIT partitioning helpers, see `bench/include/gapbs/cache/README.md`.
