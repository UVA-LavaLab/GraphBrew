# Python Infrastructure

GraphBrew has one public Python entry point:

```bash
python3 scripts/graphbrew_experiment.py --help
```

Use the orchestrator for dependency management, graph preparation,
reordering, benchmarking, cache simulation, offline model fitting,
verification, and frozen-study reproduction. Do not add one-off runners or
duplicate policy registries in experiment scripts.

## Package layout

```text
scripts/
├── graphbrew_experiment.py       public orchestrator
├── requirements.txt
├── lib/
│   ├── core/
│   │   ├── utils.py              paths, algorithms, variants, result types
│   │   ├── experiment_policy.py  benchmark/cache/evaluation policy
│   │   ├── datastore.py          raw observation and graph-property stores
│   │   ├── graph_types.py        shared graph data types
│   │   └── graph_data.py         graph metadata and run logs
│   ├── pipeline/
│   │   ├── dependencies.py       dependency checks/install
│   │   ├── build.py              binary builds
│   │   ├── download.py           graph acquisition
│   │   ├── reorder.py            mapping generation
│   │   ├── reorder_config.py     effective/realized config validation
│   │   ├── benchmark.py          execution and timing parser
│   │   └── cache.py              cache-simulation execution
│   ├── ml/
│   │   ├── feature_schema.py     shared Tier-0 feature contract
│   │   ├── portfolio.py          deployable arm registry
│   │   ├── weights.py            offline perceptron scoring/fitting
│   │   ├── model_tree.py         offline model implementations
│   │   └── adaptive_emulator.py  Python/C++ parity diagnostics
│   ├── analysis/                 downstream metrics and diagnostics
│   └── tools/                    maintenance utilities
├── experiments/
│   ├── adaptive/                 adaptive planning and dry-run manifests
│   ├── vldb/                     frozen study runner and restartable stages
│   ├── ecg/                      ECG study runner
│   ├── partition_cut/            separate partition research path
│   └── adaptive_ml/              retired legacy ablation entry points
└── test/                         pytest regression suite
```

All imports use the canonical `scripts.*` package identity. Direct file
execution must insert the repository root and still import `scripts.*`; loading
the same module as both `lib.*` and `scripts.lib.*` is prohibited.

## Common orchestrator commands

```bash
# Dependency checks
python3 scripts/graphbrew_experiment.py --check-deps

# Generic rapid path
python3 scripts/graphbrew_experiment.py \
  --full --quick --size small --trials 1 --skip-cache \
  --graphs-dir /media/Data/00_GraphDatasets/GraphBrew

# One phase
python3 scripts/graphbrew_experiment.py --phase benchmark --size small

# Dry-run a broad collection
python3 scripts/graphbrew_experiment.py --target-graphs 50 --dry-run

# Authoritative verification gate
python3 scripts/graphbrew_experiment.py --test
```

The frozen study is also launched through the orchestrator. Use
`scripts/experiments/vldb/stages/` directly only when a long run must be
restartable stage by stage.

## Single sources of truth

- Algorithm IDs and general variants: `scripts/lib/core/utils.py`
- Benchmark and cache subsets: `scripts/lib/core/experiment_policy.py`
- GraphBrew config parsing/validation:
  `scripts/lib/pipeline/reorder_config.py`
- Frozen study graph/algorithm/trial matrix:
  `scripts/experiments/vldb/config.py`
- Deployable adaptive portfolio: `scripts/lib/ml/portfolio.py`
- Tier-0 features: `scripts/lib/ml/feature_schema.py` and the shared C++
  schema definition

Call the shared canonical-name and converter-option helpers instead of
rebuilding names or `-o` strings.

## Result storage

The generic harness is the official writer for
`results/data/benchmarks.json`. Each versioned row is one immutable raw
observation. Labeling, measurement mode, threads, mapping identity, attempt,
success/failure state, and preprocessing components are preserved.

C++ self-recording is explicit only:

```bash
./bench/bin/pr -f graph.sg -s -o 5 -n 1 -D /tmp/graphbrew-db/
```

Do not let Python and C++ concurrently rewrite the same result file.

## Adaptive evaluation

Benchmark binaries load exported models and never train at runtime. Legacy
non-nested LOGO evaluators fail closed. Generalization evidence must use nested
leave-one-topology-out folds with fold-local portfolio selection, model
fitting, and OOD calibration.

## Testing

```bash
make check
python3 -m pytest scripts/test/test_model_tree.py::TestCriterion::test_criterion_values -q
make check-partition
```

The primary gate intentionally excludes edge/GAS and partition integration
suites; those remain explicit secondary paths.
