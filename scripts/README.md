# Python tooling

```
scripts/
├── experiments/               specialized restartable campaign runners
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

## Entry points

- `scripts/graphbrew_experiment.py --phase all` — original one-click pipeline
- `scripts/generate_public_figures.py --check` — verify generated public
  SVG/draw.io/catalog artifacts against the running-example fixture

Specialized campaign implementations remain under `scripts/experiments/`.
Their release documentation is intentionally separate from the generic
public workflow.

## Tests

```bash
pytest scripts/test/
```
