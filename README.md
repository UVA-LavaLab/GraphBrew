# ECG Cache Architecture Artifact

This branch is the implementation and reproducibility artifact for the successor
to **ECG: Expressing Locality and Prefetching for Optimal Caching in Graph
Structures** (IEEE IPDPSW 2024).

The new architecture adds:

- **K2** two-future-reference edge records;
- traversal-adaptive graph-cache replacement;
- **StreamShield** request-bound LLC placement control;
- RISC-V `ecg.load2` and `ecg.stream.load2`;
- cache_sim, gem5, and Sniper implementations with exact equivalence gates.

The public HPCA paper name remains open. Implementation names remain `ECG_*`.

## Repository map

| Path | Purpose |
|---|---|
| `research/ecg-hpca/` | Paper SSOT, claim ledger, methodology, results, runbook |
| `research/ecg-hpca/evidence/` | Historical ECG experiments and audit evidence |
| `scripts/experiments/ecg/` | Canonical experiment, verification, analysis, and Slurm package |
| `bench/include/cache_sim/` | Functional cache hierarchy and ECG policy |
| `bench/include/gem5_sim/` | gem5 configs, overlays, and ISA support |
| `bench/include/sniper_sim/` | Sniper configs, overlays, and fused K2 model |
| `bench/src_sim/` | cache_sim-instrumented graph kernels |
| `bench/src_gem5/` | gem5 graph kernels |
| `bench/src_sniper/` | Sniper kernels and bounded SIFT workload |
| `wiki/ECG-HPCA-Paper.md` | Minimal public-facing status page |

## Setup

```bash
make setup-gem5
make setup-sniper
make all-sim
make gem5-riscv-m5ops-pr gem5-riscv-m5ops-bfs
make sniper-sg_kernel
```

RISC-V gem5 builds additionally require a RISC-V cross compiler.

## Correctness gates

```bash
pytest -q scripts/test

python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr bfs --schedule-k 2

python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr --schedule-k 2 --stream-bypass
```

## Paper matrix

Every reported comparison includes:

```text
LRU  SRRIP  GRASP  charged P-OPT  K2  K2+StreamShield
```

The cache_sim factorial additionally exposes `ECG:K1` and
`ECG:K1_STREAMSHIELD`.

```bash
python3 scripts/experiments/ecg/flows/paper_run.py \
  --profile streamshield_sniper_realgraph \
  --run-dir results/ecg_experiments/final_paper_runs/ecg_successor_webgoogle \
  --no-build
```

See [`research/ecg-hpca/RUNBOOK.md`](research/ecg-hpca/RUNBOOK.md) for local,
Slurm, and aggregation workflows.

## Prior-publication boundary

The IPDPSW 2024 ECG paper is archival. An HPCA submission must be materially
different, cite the workshop paper, disclose the contribution delta, and receive
PC-chair guidance before registration. See
[`research/ecg-hpca/CHAIR_QUERY.md`](research/ecg-hpca/CHAIR_QUERY.md).

Generated `results/`, simulator checkouts, binaries, traces, and graph files are
ignored and must not be committed.
