# Contributing to the ECG Artifact

This branch accepts changes only when they directly support the ECG successor
architecture, its baselines, or its reproducibility.

## Rules

1. Keep paper configuration in
   `scripts/experiments/ecg/final_paper_manifest.json`.
2. Keep current claims and methodology in `research/ecg-hpca/`.
3. Include LRU, SRRIP, GRASP, charged P-OPT, K2, and K2+StreamShield in every
   reported comparison.
4. Preserve policy isolation: ambient ECG variables must not contaminate
   baselines.
5. Do not commit generated `results/`, graph datasets, simulator checkouts, or
   binaries.
6. Do not claim overall detailed-simulator superiority until the real-graph
   Sniper matrix is complete.

## Validation

```bash
pytest -q scripts/test
python3 -m py_compile \
  scripts/experiments/ecg/roi_matrix.py \
  scripts/experiments/ecg/flows/paper_run.py \
  scripts/experiments/ecg/flows/paper_pipeline.py
git diff --check
```

For simulator or policy changes, also run the focused equivalence gates listed
in the root README.
