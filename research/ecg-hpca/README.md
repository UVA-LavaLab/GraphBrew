# ECG Successor HPCA Paper -- Navigation

This directory is the paper-facing source of truth for the ECG cache work.
The public paper name is pending; implementation code keeps the `ECG_*` names.

## Read in this order

1. [`PAPER.md`](PAPER.md) -- the **normative** scientific SSOT: thesis, prior
   publication, mechanism, simulator roles, metrics/claim rules, evaluation
   plan, current STOP interpretation, limitations, and the claim/scope
   ledger.
2. [`RESULTS.md`](RESULTS.md) -- current measured tables only
   (non-normative; interpretation lives in `PAPER.md`).
3. [`ARTIFACT.md`](ARTIFACT.md) -- build, test, and reproduction commands.
4. [`evidence/current/`](evidence/current/) -- compact evidence notes behind
   the numbers in `RESULTS.md`.
5. [`preregistration/pr_screen.json`](preregistration/pr_screen.json) -- the
   one active preregistration.
6. [`evidence/`](evidence/) -- historical/superseded experiment findings,
   retained for audit only.
7. [`physical/README.md`](physical/README.md) -- non-normative physical-input
   generation and remaining synthesis requirements.

## Canonical executable sources

- Manifest profile: `ecg_pr_screen` in
  `scripts/experiments/ecg/final_paper_manifest.json`
- Screen analyzer: `scripts/experiments/ecg/analysis/proposal_sota_gate.py`
- Paper runner: `scripts/experiments/ecg/flows/paper_run.py`
- Matrix engine: `scripts/experiments/ecg/roi_matrix.py`

Do not add compatibility wrappers or paper configuration outside the
manifest and preregistration files above. Generated `results/` artifacts are
never committed.
