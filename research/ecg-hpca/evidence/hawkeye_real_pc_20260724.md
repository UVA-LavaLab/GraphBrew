# Faithful gem5 Hawkeye Gate — 2026-07-24

- Commit: `ed372d2b34ad5e64cf39041ce6cd81b6689e5d60`
- Profile: `ecg_gem5_hawkeye_gate`
- Scope: synthetic `kron_s12_k4`, gem5 TimingSimpleCPU, 32 KiB/8-way LLC,
  no prefetch, PR/BFS/SSSP/BC/CC.
- Policies: LRU, SRRIP, GRASP, faithful real-PC Hawkeye, charged P-OPT, and
  prototype indexed K2.
- Result: **30/30 rows passed** and all five Hawkeye rows report
  `hawkeye_pc_source=request_instruction_pc` and
  `hawkeye_faithfulness=faithful_real_instruction_pc`.

Hawkeye versus LRU:

| Kernel | L3 miss ratio | Miss change | Diagnostic speedup |
|---|---:|---:|---:|
| PR | 1.098x | +9.80% | 0.985x |
| BFS | 1.216x | +21.58% | 0.971x |
| SSSP | 1.182x | +18.23% | 0.986x |
| BC | 1.162x | +16.19% | 0.960x |
| CC | 1.144x | +14.38% | 0.981x |
| Geomean | **1.160x** | **+15.97%** | **0.977x** |

This establishes that the faithful request-PC adapter executes and produces a
complete comparison. It does **not** establish general Hawkeye quality: this is
a small synthetic gate, and Hawkeye is worse than LRU in every evaluated cell.
The real-graph learned-policy claim remains pending.

Frozen artifacts:

| Artifact | SHA-256 |
|---|---|
| `combined_roi_matrix.csv` | `e44cd4369498059ce24f2e3a58a5a4f42ae25f709aa9fcd0b77f5bb18d4eab77` |
| `hawkeye_summary.json` | `3e2446f7a83899ff21f927b2bffda0ea61390614a4792bb347f07c907241c93a` |
| `hawkeye_certification.json` | `992276e91586a37acc35a03a9575fcb522d3c6b7a4075dfa64c6ffbd0b249aa6` |

Run directory:
`results/ecg_experiments/final_paper_runs/ecg_gem5_hawkeye_gate_20260724`.
