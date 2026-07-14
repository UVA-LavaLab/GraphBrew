# scripts/lib/core/ — Constants, Logging, Data Stores

Single source of truth for algorithm IDs, paths, graph types, and runtime data.

| File | Purpose |
|------|---------|
| `utils.py` | Algorithm ID map (0–16), path constants, `Logger`, `run_command`, variant lists |
| `graph_types.py` | `GraphInfo` dataclass, `GraphType` enum (SOCIAL, WEB, ROAD, etc.) |
| `datastore.py` | `BenchmarkStore` (streaming JSON), `GraphPropsStore` (feature vectors) |
| `graph_data.py` | Graph download/conversion helpers, SuiteSparse catalog integration |

Key constants in `utils.py`:
- `ALGORITHMS` — dict mapping ID → name for all 17 algorithms
- `BIN_DIR`, `BIN_SIM_DIR` — compiled binary paths
- `GRAPHBREW_VARIANTS` — tuple of GraphBrew ordering variant names
- `SLOW_ALGORITHMS` — set of IDs that may be slow (Gorder, Corder, RCM)
