# GraphBrew ECG Research Artifact

This repository contains the implementation and reproducibility artifact for
the successor to *ECG: Expressing Locality and Prefetching for Optimal Caching
in Graph Structures* (IEEE IPDPSW 2024). Implementation identifiers retain the
`ECG_*` names while the public paper name remains open.

## Canonical paper documents

| Document | Ownership |
|---|---|
| [`research/ecg-hpca/PAPER.md`](research/ecg-hpca/PAPER.md) | Sole normative scientific SSOT: mechanism, simulator roles, metrics, current interpretation, limitations, and claim boundaries |
| [`research/ecg-hpca/RESULTS.md`](research/ecg-hpca/RESULTS.md) | Current measured tables; interpretation remains in `PAPER.md` |
| [`research/ecg-hpca/ARTIFACT.md`](research/ecg-hpca/ARTIFACT.md) | Dataset staging, build, test, execution, and analysis commands |
| [`research/ecg-hpca/README.md`](research/ecg-hpca/README.md) | Paper-directory navigation |
| [`research/ecg-hpca/preregistration/pr_screen.json`](research/ecg-hpca/preregistration/pr_screen.json) | Active PageRank screen configuration |

This root README is navigation only. It does not independently define the
architecture, active policy, result interpretation, or reproduction procedure.

## Repository map

| Path | Purpose |
|---|---|
| `bench/include/` | Shared cache, metadata, simulator-overlay, and policy support |
| `bench/src_sim/` | cache_sim-instrumented graph kernels |
| `bench/src_gem5/` | gem5 graph kernels |
| `bench/src_sniper/` | Sniper graph workload |
| `scripts/experiments/ecg/` | Experiment runner, manifest, analyzers, and verification tools |
| `scripts/test/` | Focused artifact and scientific-contract tests |
| `research/ecg-hpca/evidence/` | Current compact evidence plus historical audit material |
| `wiki/ECG-HPCA-Paper.md` | Public landing page pointing back to the canonical documents |

Start with [`research/ecg-hpca/ARTIFACT.md`](research/ecg-hpca/ARTIFACT.md)
for all build and reproduction commands. Generated graphs, simulator outputs,
binaries, traces, and `results/` content are gitignored and must not be
committed.

The IPDPSW 2024 paper is archival. Submission-eligibility and contribution-
delta guidance for a successor paper is maintained only in
[`research/ecg-hpca/PAPER.md`](research/ecg-hpca/PAPER.md).
