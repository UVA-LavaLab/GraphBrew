# Hardened K2-M Three-Simulator Conformance — 2026-07-24

This supersedes `k2m_equivalence_20260724.md` for the mechanism gate. The older
run is valid but was produced by a weaker verifier; only this run carries the
structured evidence archive, the exact victim oracle, and the trace-correlation
checks described below.

## What this gate does and does not claim

**Claims.** cache_sim, gem5, and Sniper compile the *same* victim selector
(`bench/include/ecg_victim_policy.h`), each independently emits its native
per-way candidate state, and every emitted eviction matches an exact Python
decision oracle. Each backend additionally passes scoped K2 delivery checks.

**Does not claim.** Numerically identical cache statistics or identical victim
sequences across the three simulators. Their hierarchies, inclusion policies,
and frontends differ; cross-simulator results are read as mechanism agreement
and direction relative to each simulator's own LRU.

## Result

- Commit: `3811624b2303962a4e3c0ca9f153038e8e5f198b` (clean worktree).
- Graph: `email-Eu-core.sg`; Schedule-2; computed-address K2-M (`mask`).
- Command:

```bash
python3 scripts/experiments/ecg/verify/equiv_kernels.py \
  --gem5 --sniper --kernels pr bfs sssp bc cc \
  --schedule-k 2 --gem5-isa-variant mask \
  --evidence-dir \
    results/ecg_experiments/final_paper_runs/ecg_3sim_k2m_conformance_20260724
```

- **15/15 kernel x simulator cells conform.**
- Zero K2 distance mismatches in every cell.
- 11 of 15 cells reached *decisive* epoch eviction (epoch distance strictly
  decided at least one victim); 6 were required to.
- All five independent preflight units passed: exact victim unit, field-layout
  parity, epoch-pair builder, unknown-`ECG_MODE` hard-fail, and
  unknown-`ECG_VARIANT` hard-fail.

| kernel | cache_sim | gem5 | Sniper |
|---|---|---|---|
| PR | decisive | conformance | conformance (32 binds) |
| BFS | decisive | decisive | conformance (32 binds) |
| SSSP | decisive | decisive | decisive (32 binds) |
| BC | decisive | decisive | decisive (32 binds) |
| CC | decisive | conformance | decisive (32 binds) |

Sniper cells additionally required exact governed-load binding: all 32
bind-consume records paired one-to-one with 32 fused receipts under a *shared
transaction id*, with line alignment, power-of-two line size, epoch range, and
nonzero context checked per record.

## Evidence archive

Run directory (gitignored):
`results/ecg_experiments/final_paper_runs/ecg_3sim_k2m_conformance_20260724`.

| Artifact | SHA-256 |
|---|---|
| `manifest.json` | `cc509b73cf0eea768d6a723b045b12c725d10c389d69753959850fa7e2487cbf` |
| `summary.json` | `6a74ac1d032da52a3d5be94a8c8bf4b5960e6d4c0ca8dca6561444ad04d6e8ed` |
| `preflight.json` | `dd67c20ec8aa065fa63629dceddf06725d343974ec50013511f4bfa4997379a4` |

The archive contains, per cell, the raw trace plus the copied
`roi_matrix.{csv,json,complete.json}` with hashes, and at the top level a
manifest pinning the graph, both simulator binaries, all guest binaries, the
policy SSOT, the verifier itself, and the git head. The Sniper simulator and
workload binaries are copied *into* the archive so a proof cannot later be
validated against mutable external files. `manifest.summary_sha256` binds the
summary, and evidence capture refuses to run on a dirty worktree.

## Scope and caveats

- The conformance geometry is deliberately tiny (kB-scale caches) so property
  lines actually reach the LLC. It proves mechanism behavior, not performance.
- cache_sim and gem5 run with `ECG_STORED_REFRESH=1` to force decisive
  epoch-ranked evictions. That is a coverage device, not a claim that stored
  refresh is hardware-free; Sniper runs without it.
- gem5 here uses TimingSimpleCPU, so K2 metadata arrives through the serialized
  mailbox and accepts are correlated serially against the delivery trace. The
  race-free per-request O3 binding path is exercised separately.
- gem5 accept records are a *subset* of the traced loads because loads that hit
  in an inner cache never reach the LLC callback. Every traced request record is
  still checked against its expected payload, so a missing accept cannot hide a
  corrupted delivery.
- This gate says nothing about K2 speedup or miss-rate superiority. Those remain
  governed by `claim_gate.json`.
