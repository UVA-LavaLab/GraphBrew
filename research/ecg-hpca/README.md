# ECG Successor HPCA Paper SSOT

This directory is the official paper-facing source of truth for the ECG cache
work. The working paper name is:

> **Public paper name pending; implementation code name: ECG**

The public name must distinguish this architecture from the archival ECG
workshop paper. The implementation keeps the established `ECG_*` code and ISA
names.

## Read in this order

1. [`PAPER.md`](PAPER.md) — thesis, title, abstract, and contribution outline.
2. [`ARCHITECTURE.md`](ARCHITECTURE.md) — K2, StreamShield, ISA, diagrams, and comparisons.
3. [`CHAIR_QUERY.md`](CHAIR_QUERY.md) — required HPCA prior-publication inquiry.
4. [`CLAIMS.md`](CLAIMS.md) — claims that are proven, pending, or prohibited.
5. [`METHODOLOGY.md`](METHODOLOGY.md) — simulator roles, baselines, and hardware model.
6. [`RESULTS.md`](RESULTS.md) — frozen numerical results and their scope.
7. [`RUNBOOK.md`](RUNBOOK.md) — canonical local and Slurm commands.
8. [`evidence/`](evidence/) — dated audits and historical experiment findings.

## Canonical executable sources

- Manifest: `scripts/experiments/ecg/final_paper_manifest.json`
- Paper runner: `scripts/experiments/ecg/flows/paper_run.py`
- Matrix engine: `scripts/experiments/ecg/roi_matrix.py`
- Pipeline: `scripts/experiments/ecg/flows/paper_pipeline.py`
- Slurm shards: `scripts/experiments/ecg/slurm/make_slurm_shards.py`
- Correctness gates: `scripts/experiments/ecg/verify/`

Do not add compatibility wrappers or paper configuration outside the manifest.

## Documentation policy

- This directory defines the current paper.
- The GitHub Wiki page `ECG-HPCA-Paper` is the public landing page.
- `research/ecg-hpca/evidence/` preserves dated evidence and debugging history.
- No additional wiki pages are normative.
- Generated `results/` artifacts are never committed.
