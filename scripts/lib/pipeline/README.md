# scripts/lib/pipeline/ — Experiment Pipeline Stages

Modules implementing each phase of the GraphBrew experiment pipeline.

| File | Phase | Purpose |
|------|-------|---------|
| `dependencies.py` | Setup | Check/install system deps (g++, Boost, NUMA, tcmalloc) |
| `build.py` | Setup | Build benchmark binaries (`make all`, `make all-sim`) |
| `download.py` | Download | Download graphs from SuiteSparse catalog |
| `suitesparse_catalog.py` | Download | SuiteSparse Matrix Collection API client |
| `reorder.py` | Reorder | Generate label maps (.lo files) for all algorithm × graph pairs |
| `benchmark.py` | Benchmark | Run benchmark binaries, parse timing output |
| `cache.py` | Cache | Run cache simulation binaries, parse L1/L2/L3 statistics |
| `phases.py` | Orchestration | High-level phase functions called by graphbrew_experiment.py |
| `progress.py` | Utility | Progress tracking and ETA estimation |

Pipeline flow: `dependencies → build → download → reorder → benchmark → cache → weights`
