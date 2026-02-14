# scripts/lib/ — Shared Python Modules

Core library used by `graphbrew_experiment.py` and other scripts.

| Module | Purpose |
|--------|---------|
| `utils.py` | **SSOT** — algorithm IDs, variant registry, constants, logging |
| `reorder.py` | Reordering execution, mapping I/O, algorithm configs |
| `benchmark.py` | Benchmark runner, result parsing |
| `benchmark_runner.py` | Low-level benchmark process management |
| `features.py` | Graph topology feature extraction |
| `graph_types.py` | Graph/algorithm dataclasses |
| `graph_data.py` | Graph metadata and dataset management |
| `download.py` | Graph dataset downloading |
| `cache.py` | Cache simulation orchestration |
| `cache_compare.py` | Cache miss comparison analysis |
| `training.py` | Perceptron training pipeline |
| `perceptron.py` | Perceptron model implementation |
| `weights.py` | Weight file I/O and management |
| `weight_merger.py` | Multi-run weight merging |
| `eval_weights.py` | Weight evaluation and scoring |
| `analysis.py` | Result analysis and statistics |
| `figures.py` | Plot generation |
| `phases.py` | Experiment phase orchestration |
| `oracle.py` | Oracle (best-possible) analysis |
| `metrics.py` | Performance metrics computation |
| `progress.py` | Progress bar utilities |
| `build.py` | C++ build helpers |
| `dependencies.py` | Dependency checking |
| `ab_test.py` | A/B testing utilities |
| `adaptive_emulator.py` | AdaptiveOrder emulation |
| `leiden_compare.py` | Leiden variant comparison |
| `check_includes.py` | C++ include lint checking |
