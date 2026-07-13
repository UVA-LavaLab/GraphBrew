# ECG Experiment Package

## SSOT hierarchy

1. `research/ecg-hpca/` — paper claims and methodology.
2. `final_paper_manifest.json` — final experiment definitions.
3. `flows/paper_run.py` — resumable orchestration.
4. `roi_matrix.py` — simulator cell engine.
5. `policy_specs.py` — canonical policy labels and K1/K2 transport modes.
6. `flows/paper_pipeline.py` — complete-matrix aggregation.
7. `verify/` — correctness gates.
8. `slurm/` — one-policy shard generation and array execution.

## Directory roles

| Path | Responsibility |
|---|---|
| `roi_matrix.py` | One simulator matrix; environment isolation; metrics; completion marker |
| `policy_specs.py` | One policy parser/output-label SSOT |
| `flows/paper_run.py` | Manifest expansion, fingerprints, resume, run completion |
| `flows/proof_matrix.py` | cache_sim component ablations |
| `flows/paper_pipeline.py` | Reject incomplete groups; generate tables and figures |
| `verify/ecg.py` | Exact policy and live-trace checks |
| `verify/equiv.py` | Self-contained behavioral parity and insertion invariants |
| `verify/equiv_kernels.py` | PR/BFS multi-simulator K2 and StreamShield gates |
| `verify/pfx.py` | Prefetch-path correctness |
| `slurm/make_slurm_shards.py` | Manifest-derived policy shards |

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
