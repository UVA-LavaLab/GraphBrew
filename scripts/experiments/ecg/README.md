# ECG Experiment Package

## SSOT hierarchy

1. `research/ecg-hpca/` — paper claims and methodology.
2. `final_paper_manifest.json` — final experiment definitions.
3. `flows/paper_run.py` — resumable orchestration.
4. `roi_matrix.py` — simulator cell engine.
5. `flows/paper_pipeline.py` — complete-matrix aggregation.
6. `verify/` — correctness gates.

## Final policy labels

```text
LRU
SRRIP
GRASP
POPT
ECG:K2
ECG:K2_STREAMSHIELD
```

`POPT` is the charged, size-correct practical baseline.
`POPT:UNCHARGED` is an oracle-only diagnostic.

## Run

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile streamshield_sniper_realgraph \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_successor_webgoogle \
  --no-build
```

## Verify

```bash
python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr bfs --schedule-k 2
```

Generated result directories are ignored.
