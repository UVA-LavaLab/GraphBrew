# scripts/lib/ — Shared Python Library

Core library powering `graphbrew_experiment.py` and related tooling.
Organised into **five sub-packages** for clarity.

## Package Map

```
lib/
├── __init__.py          # Re-exports every public name (backward-compatible)
├── README.md            # ← you are here
│
├── core/                # Constants, logging, data stores
│   ├── utils.py         # SSOT — algorithm IDs, variant registry, paths, logging
│   ├── graph_types.py   # Graph / algorithm dataclasses
│   ├── datastore.py     # BenchmarkStore, GraphPropsStore (append-only JSON DBs)
│   └── graph_data.py    # Graph metadata & dataset catalog
│
├── pipeline/            # Experiment execution stages
│   ├── dependencies.py  # External-tool checks (cmake, numactl, …)
│   ├── build.py         # C++ build helpers
│   ├── download.py      # Graph dataset downloading
│   ├── suitesparse_catalog.py  # SuiteSparse auto-discovery (ssgetpy)
│   ├── reorder.py       # Reordering execution, mapping I/O
│   ├── benchmark.py     # Benchmark runner & result parsing
│   ├── cache.py         # Cache-simulation orchestration
│   ├── phases.py        # Phase orchestration (reorder → benchmark → cache)
│   └── progress.py      # Progress-bar utilities
│
├── ml/                  # ML scoring & training (legacy / fallback)
│   ├── weights.py       # SSO scoring — PerceptronWeight.compute_score()
│   ├── eval_weights.py  # SSO data loading — load_all_results(), etc.
│   ├── training.py      # Iterative / batched perceptron training
│   ├── model_tree.py    # Decision tree & hybrid DT+Perceptron models
│   ├── adaptive_emulator.py  # AdaptiveOrder C++ emulation
│   ├── oracle.py        # Oracle (best-possible) analysis
│   └── features.py      # Graph topology feature extraction
│
├── analysis/            # Post-run analysis & visualisation
│   ├── adaptive.py      # A/B testing, Leiden variant comparison
│   ├── metrics.py       # Performance-metrics computation
│   └── figures.py       # SVG / PNG plot generation
│
└── tools/               # Standalone CLI utilities
    ├── check_includes.py   # C++ header-include linting
    └── regen_features.py   # Feature-vector regeneration
```

## Import Convention

The top-level `lib/__init__.py` re-exports every public symbol, so
existing code keeps working:

```python
from scripts.lib import ALGORITHMS, BenchmarkStore, PerceptronWeight
```

For new code prefer **direct sub-package imports** — they are faster and
make dependencies explicit:

```python
from scripts.lib.core.utils import ALGORITHMS, PROJECT_ROOT
from scripts.lib.ml.weights   import PerceptronWeight
from scripts.lib.pipeline.benchmark import run_benchmarks
```

## Standalone CLI Entry-Points

Several modules double as CLI scripts:

```bash
python -m scripts.lib.ml.weights          --help   # train / export weights
python -m scripts.lib.ml.eval_weights     --help   # evaluate weight files
python -m scripts.lib.analysis.metrics    --help   # compute metrics
python -m scripts.lib.tools.check_includes         # lint C++ headers
python -m scripts.lib.tools.regen_features         # regenerate features
```

## Architecture Notes

| Concept | Detail |
|---------|--------|
| **Streaming Database v2.0** | C++ binaries write to `benchmarks.json` / `graph_properties.json` via `InitSelfRecording()`. C++ trains ML models (perceptron, DT, hybrid) **at runtime** from the database — no Python training step required. |
| **SSO (fallback)** | `ml/weights.py` holds the sole Python scoring formula (`compute_score`). `ml/eval_weights.py` holds the sole data-loading helpers. All other ML modules delegate to these two. Used when the streaming database has < 3 graphs. |
| **Phase pipeline** | Default: `reorder → benchmark → cache`. Weight generation is opt-in (Phase 4/5, deprecated). |
| **Algorithm naming** | Canonical list lives in `core/utils.py::ALGORITHMS`. Every module that needs algorithm IDs imports from there — never hard-codes. |
