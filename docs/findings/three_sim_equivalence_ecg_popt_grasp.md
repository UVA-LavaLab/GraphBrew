# 3-simulator equivalence: ECG / P-OPT (charged) / GRASP

**Date:** 2026-07-01
**Cell:** kron_s16_k4 (65 536 vtx) · PR · L3=128kB/16w · L1d=16kB · L2=64kB · -o5 ·
ECG_VARIANT=rrip_first (the do-no-harm variant for this synthetic equivalence cell).
Run via `three_sim_showcase.py` (→ committed `roi_matrix.py`, `--no-build`).

## L3 miss rate (lower = better)
| policy | cache_sim | gem5 (ROI) | Sniper |
|---|---|---|---|
| LRU | 0.6606 | 0.6447 | 0.5744 |
| GRASP | 0.5319 | 0.5621 | 0.4824 |
| P-OPT (charged) | 0.4763 | 0.4686 | 0.4816 |
| ECG (ECG_GRASP_POPT) | 0.5709 | 0.5824 | 0.4638 |

## Direction vs LRU (Δ = mr − LRU; negative = HELPS) — the equivalence read
Absolute rates are NOT comparable across sims (cache_sim sees graph accesses only;
gem5/Sniper see the full ISA stream), so equivalence = same DIRECTION per sim.

| sim | GRASP | P-OPT | ECG |
|---|---|---|---|
| cache_sim | −0.129 ✓ | −0.184 ✓ | −0.090 ✓ |
| gem5 | −0.083 ✓ | −0.176 ✓ | −0.062 ✓ |
| Sniper | −0.092 ✓ | −0.093 ✓ | −0.111 ✓ |

**All three policies reduce the L3 miss rate vs LRU in all three simulators — 3-sim
equivalence achieved.** Relative ordering: cache_sim & gem5 rank P-OPT best (iterative PR
is P-OPT's sweet spot); Sniper ranks ECG best (0.4638) — the documented substrate
difference (Sniper rewards ECG's GRASP-insertion), not a bug.

## Two gotchas recorded (for future showcase runs)
1. **gem5 ECG mode-6 is slow** — it exceeds roi_matrix's default `--timeout-gem5=900s` on
   this cell (LRU/GRASP/POPT finish in ~5-6 min). Re-run the ECG cell with
   `--timeout-gem5 3600` (it completed in ~22 min).
2. **gem5 stats have 2 sections** — section 1 = the main kernel ROI (~1.19B insts),
   section 2 = a tiny ~5.5M-inst tail (noisy). `three_sim_showcase.py` takes the LAST
   row (section 2), which is anomalous for ECG (0.7248 vs the true ROI 0.5824). Read
   **section 1** for the kernel miss rate. (Candidate showcase fix: prefer section 1.)
