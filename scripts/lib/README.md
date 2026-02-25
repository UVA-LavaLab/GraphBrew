# scripts/lib/ — Shared Python Modules

Core library used by `graphbrew_experiment.py` and other scripts.

| Module | Purpose |
|--------|---------|
| `utils.py` | **SSOT** — algorithm IDs, variant registry, constants, logging |
| `reorder.py` | Reordering execution, mapping I/O, algorithm configs |
| `benchmark.py` | Benchmark runner, result parsing, fresh benchmark runner |
| `features.py` | Graph topology feature extraction |
| `graph_types.py` | Graph/algorithm dataclasses |
| `graph_data.py` | Graph metadata and dataset management |
| `download.py` | Graph dataset downloading |
| `cache.py` | Cache simulation orchestration, quick cache comparison |
| `training.py` | Perceptron training pipeline |
| `perceptron.py` | Perceptron model implementation |
| `weights.py` | Weight file I/O and management |
| `weight_merger.py` | Multi-run weight merging |
| `eval_weights.py` | Weight evaluation and scoring |
| `analysis.py` | Result analysis, A/B testing, Leiden variant comparison |
| `figures.py` | Plot generation |
| `phases.py` | Experiment phase orchestration |
| `oracle.py` | Oracle (best-possible) analysis |
| `metrics.py` | Performance metrics computation |
| `progress.py` | Progress bar utilities |
| `build.py` | C++ build helpers |
| `dependencies.py` | Dependency checking |
| `adaptive_emulator.py` | AdaptiveOrder emulation |
| `check_includes.py` | C++ include lint checking |
| `datastore.py` | Unified adaptive_models.json database |
| `regen_features.py` | Feature regeneration utility |
