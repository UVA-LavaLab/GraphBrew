# Python tooling

```
scripts/
├── experiments/               isolated, restartable study runners
│   ├── vldb/                   frozen study reproduction
│   ├── adaptive_ml/            retired offline-model ablation
│   └── legacy/                 archived (no live imports)
├── lib/                       ← reusable Python modules (imported, not run)
│   ├── core/                   ResultsStore, parsing, run helpers
│   ├── pipeline/               download.py (catalog auto-download), build, convert
│   ├── analysis/               metrics and offline diagnostics
│   ├── ml/                     retained offline-model tooling
│   └── tools/                  misc CLIs
├── test/                      ← pytest tests
├── graphbrew_experiment.py    public experiment orchestrator
├── generate_public_figures.py running-example figure/wiki generator
└── requirements.txt
```

## Canonical paths (single source of truth)

| Artifact | Path |
|---|---|
| Large graph corpus                    | `/media/NVMeData/00_GraphDatasets/GraphBrew/` |
| Large mappings/results                | `/media/NVMeData/00_GraphDatasets/GraphBrew/artifacts/` |
| Generic observations                  | `results/data/` |
| Generic logs                          | `results/logs/`, `results/slurm_logs/` |

Auto-download for the VLDB pipeline is driven by
[`experiments/vldb/config.py:VLDB_GRAPH_SOURCES`](experiments/vldb/config.py).

## Frozen-study reproduction

```bash
source .venv/bin/activate

# Smoke (~1 min, 2 tiny graphs)
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 2 --preview
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 2 --preview
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --preview

# Local 6-graph eval
python3 scripts/experiments/vldb/stages/01_prep.py     --exp 2 --local
python3 scripts/experiments/vldb/stages/02_reorder.py  --exp 2 --local
python3 scripts/experiments/vldb/stages/03_cpu_perf.py --exp 2 --local

# Cache stats only (host CPU speed doesn't matter)
python3 scripts/experiments/vldb/stages/04_cache_sim.py --exp 1 --local

# Figures
python3 scripts/experiments/vldb/stages/05_aggregate.py --exp 0
```

SLURM templates: `scripts/experiments/vldb/stages/slurm/*.sbatch`.

## Legacy / all-in-one entry points

- `scripts/experiments/vldb/runner.py --all --local` — monolithic VLDB runner
- `scripts/graphbrew_experiment.py --phase all` — original one-click pipeline
- `scripts/generate_public_figures.py --check` — verify generated public
  SVG/draw.io/catalog artifacts against the running-example fixture
- `scripts/experiments/vldb/slurm/monolithic.sbatch` — monolithic SLURM template

## Tests

```bash
pytest scripts/test/
```
