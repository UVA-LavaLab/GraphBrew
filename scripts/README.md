# Python tooling

```
scripts/
├── experiments/               isolated, restartable study runners
│   ├── vldb/                   frozen study reproduction
│   ├── adaptive_ml/            model diagnostics
│   └── legacy/                 archived (no live imports)
├── lib/                       ← reusable Python modules (imported, not run)
│   ├── core/                   ResultsStore, parsing, run helpers
│   ├── pipeline/               download.py (catalog auto-download), build, convert
│   ├── analysis/               amortise, figures, cold-start sim
│   ├── ml/                     adaptive ordering model
│   └── tools/                  misc CLIs
├── test/                      ← pytest tests
├── graphbrew_experiment.py    public experiment orchestrator
└── requirements.txt
```

## Canonical paths (single source of truth)

| Artifact | Path |
|---|---|
| Large graph corpus                    | `/media/Data/00_GraphDatasets/GraphBrew/` |
| Large mappings/results                | `/media/Data/00_GraphDatasets/GraphBrew/artifacts/` |
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
- `scripts/experiments/vldb/slurm/monolithic.sbatch` — monolithic SLURM template

## Tests

```bash
pytest scripts/test/
```
