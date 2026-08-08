# Artifact: Build and Reproduction

Build, test, and reproduction commands only. Scientific claims, interpretation,
and prohibited-claim boundaries live in [`PAPER.md`](PAPER.md); current
measured tables live in [`RESULTS.md`](RESULTS.md). This file replaces the
former `RUNBOOK.md`.

## 1. Required graphs

The canonical screen uses deterministic 65,536-vertex web-Google/soc-pokec
samples and a 262,144-vertex cit-Patents sample. Expected paths:

```text
results/graphs/email-Eu-core/email-Eu-core.sg
results/graphs/web-Google/web-Google.sg
results/graphs/soc-pokec/soc-pokec.sg
results/graphs/cit-Patents/cit-Patents.sg
results/graphs/web-Google-n16/web-Google-n16.sg
results/graphs/soc-pokec-n16/soc-pokec-n16.sg
results/graphs/cit-Patents-n18/cit-Patents-n18-sym.sg
```

Graph datasets and converted `.sg` files are gitignored and must not be
committed. Stage a full graph and convert it with:

```bash
mkdir -p results/graphs/web-Google
curl -L https://snap.stanford.edu/data/web-Google.txt.gz |
  gzip -dc > results/graphs/web-Google/web-Google.el

make converter
bench/bin/converter \
  -f results/graphs/web-Google/web-Google.el \
  -b results/graphs/web-Google/web-Google.sg
```

Repeat for `email-Eu-core`, `soc-pokec-relationships` (as `soc-pokec`), and
`cit-Patents` from `https://snap.stanford.edu/data/`.

Generate the canonical samples with the deterministic sampler:

```bash
python3 scripts/experiments/ecg/flows/sample_realgraph.py \
  --input results/graphs/web-Google/web-Google.el \
  --output results/graphs/web-Google-n16/web-Google-n16.el \
  --vertices results/graphs/web-Google-n16/web-Google-n16.vertices.tsv \
  --metadata results/graphs/web-Google-n16/web-Google-n16.sample.json \
  --target-vertices 65536

python3 scripts/experiments/ecg/flows/sample_realgraph.py \
  --input results/graphs/soc-pokec/soc-pokec.el \
  --output results/graphs/soc-pokec-n16/soc-pokec-n16.el \
  --vertices results/graphs/soc-pokec-n16/soc-pokec-n16.vertices.tsv \
  --metadata results/graphs/soc-pokec-n16/soc-pokec-n16.sample.json \
  --target-vertices 65536

python3 scripts/experiments/ecg/flows/sample_realgraph.py \
  --input results/graphs/cit-Patents/cit-Patents.el \
  --output results/graphs/cit-Patents-n18/cit-Patents-n18.el \
  --vertices results/graphs/cit-Patents-n18/cit-Patents-n18.vertices.tsv \
  --metadata results/graphs/cit-Patents-n18/cit-Patents-n18.sample.json \
  --target-vertices 262144

bench/bin/converter -f results/graphs/web-Google-n16/web-Google-n16.el \
  -b results/graphs/web-Google-n16/web-Google-n16.sg
bench/bin/converter -f results/graphs/soc-pokec-n16/soc-pokec-n16.el \
  -b results/graphs/soc-pokec-n16/soc-pokec-n16.sg
bench/bin/converter -f results/graphs/cit-Patents-n18/cit-Patents-n18.el \
  -s -b results/graphs/cit-Patents-n18/cit-Patents-n18-sym.sg
```

The `-s` conversion is the explicit symmetrization step for the cit-Patents
screen cell. Use `--check-graphs` below to verify presence before running.

## 2. Build

```bash
make setup-gem5
make setup-gem5-guest-tools
make setup-sniper
make all-sim
make gem5-riscv-m5ops-pr gem5-riscv-m5ops-bfs \
  gem5-riscv-m5ops-sssp gem5-riscv-m5ops-bc gem5-riscv-m5ops-cc
make sniper-sg_kernel
```

RISC-V gem5 builds additionally require a RISC-V cross compiler.
`make setup-sniper` builds Sniper and applies the ECG overlays
(`scripts/setup_sniper.py --apply-overlays`); this is required for the
parity checks in Section 5 but does not itself run any simulation.

## 3. Tests

```bash
python3 -m pytest -q scripts/test
```

Targeted subsets used most often during ECG doc/config changes:

```bash
python3 -m pytest -q scripts/test/test_proposal_sota_screen.py
python3 -m pytest -q scripts/test/test_ecg_paper_ssot.py
python3 -m pytest -q scripts/test/test_frozen_metrics.py
```

## 4. Canonical PageRank screen

Config: `research/ecg-hpca/preregistration/pr_screen.json`.
Manifest profile: `ecg_pr_screen`
(`scripts/experiments/ecg/final_paper_manifest.json`).

Inspect the profile without building or running anything:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_pr_screen \
  --run-dir results/ecg_experiments/final_paper_runs/pr_screen_dryrun \
  --list --dry-run --no-build --allow-missing-graphs
```

Expected: `jobs=12` (3 graphs x 4 iterations), with each cell's
seven-policy roster (`LRU GRASP POPT POPT:UNCHARGED
ECG:K2_LRU_STREAMSHIELD ECG:K2_RRIP_STREAMSHIELD
ECG:K2_ONLINE_STREAMSHIELD`). Whole-cell execution only; per-policy sharding
is rejected for this profile.

Launch a fresh canonical execution with the pinned interpreter and a new run
directory:

```bash
/usr/bin/python3.12 -I scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_pr_screen \
  --run-dir results/ecg_experiments/final_paper_runs/pr_screen_rerun \
  --no-build --no-resume
```

After a complete run, evaluate the screen's stop/go decision:

```bash
python3 scripts/experiments/ecg/analysis/proposal_sota_gate.py \
  --input <run-dir>/combined_roi_matrix.csv \
  --config research/ecg-hpca/preregistration/pr_screen.json \
  --output <run-dir>/proposal_sota_decision.json
```

The decision file reports baseline sanity, all candidate ratios, worst cells,
per-graph/per-iteration/leave-one-out guards, transport attribution, oracle
attribution, and the stop/go result. Keep the raw run directory and the git
commit alongside the result; no separate stamping/attestation workflow is
required to interpret it.

The already-completed run this screen's current STOP result is drawn from:

```text
results/ecg_experiments/final_paper_runs/proposal_sota_v2_2db0a04b_20260802
```

## 5. Sniper overlay parity (no simulation required)

These checks validate that the Sniper, gem5, and cache_sim replacement-policy
sources implement the same victim-selector contract; they read source files
and run fast in-process logic, not a simulation:

```bash
python3 -m pytest -q scripts/test/test_grasp_sideband_registration.py
python3 -m pytest -q scripts/test/test_popt_permutation_equivalence.py

python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr bfs sssp bc cc --schedule-k 2
```

`equiv_kernels.py` additionally verifies the Sniper overlay is applied
(`bench/include/sniper_sim/.sniper_overlays.json`) and that
`cache_set_ecg.cc`/`ecg_victim_policy.h` under
`bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/` match
the shared `bench/include/ecg_victim_policy.h` SSOT before any kernel runs.

## 6. Other reproduction profiles

These remain in the manifest for non-screen mechanism/regression coverage and
are unaffected by the screen consolidation in this file:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_replacement_baseline \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_replacement \
  --check-graphs --no-build

python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile ecg_cache_sim_factorial \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_factorial \
  --list --dry-run --no-build
```

Full 3-simulator smoke (120-row data-shape check):

```bash
python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_3sim_allalg_smoke \
  --run-tag ecg_3sim_smoke \
  --out results/ecg_experiments/slurm/ecg_3sim_smoke.tsv

python3 scripts/experiments/ecg/flows/run_local_shards.py \
  --shards results/ecg_experiments/slurm/ecg_3sim_smoke.tsv \
  --run-root results/ecg_experiments/final_paper_runs/local \
  --jobs 8 --cache-sim-jobs 5 --gem5-jobs 1 --sniper-jobs 1

python3 scripts/experiments/ecg/verify/smoke_coverage.py \
  --csv results/ecg_experiments/paper_pipeline/ecg_3sim_smoke/aggregate/roi_matrix_all.csv
```

Acceptance is exactly 120 valid rows (3 simulators x 5 algorithms x 8
policies).

Three-real-graph cross-simulator matrix (360-row no-prefetch comparison of
web-Google, soc-pokec, and cit-Patents across cache_sim/gem5/Sniper):

```bash
python3 scripts/experiments/ecg/slurm/make_slurm_shards.py \
  --profile ecg_3sim_realgraph_allalg \
  --run-tag ecg_3sim_realgraph_allalg \
  --out results/ecg_experiments/slurm/ecg_3sim_realgraph_allalg.tsv
```

Acceptance is exactly 360 valid rows. The full-iteration Sniper headline
matrix (`streamshield_sniper_realgraph`) remains blocked pending a bounded
Sniper prefetch configuration; do not launch it, inspect only:

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile streamshield_sniper_realgraph \
  --run-dir results/ecg_experiments/final_paper_runs/headline_dry \
  --list --dry-run --no-build
```

## 7. Evidence and run directories

- Compact current evidence: [`evidence/current/`](evidence/current/) --
  three short notes plus an index, each pointing at its raw run directory
  under `results/ecg_experiments/`.
- Raw simulator output lives under `results/ecg_experiments/` and
  `results/graphs/`; both are gitignored. Never commit generated `results/`
  content.
- Historical (superseded) evidence remains under `evidence/` and in git
  history; it is not pruned by this consolidation and is not part of the
  active reproduction path above.

## 8. Aggregation

```bash
python3 scripts/experiments/ecg/flows/paper_pipeline.py \
  --skip-run \
  --input-run-dirs \
    results/ecg_experiments/final_paper_runs/ecg_replacement \
    results/ecg_experiments/final_paper_runs/ecg_factorial \
  --run-root results/ecg_experiments/paper_pipeline/ecg_final
```
