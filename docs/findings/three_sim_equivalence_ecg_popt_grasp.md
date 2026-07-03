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

## ⚠️ CAVEAT: this cell certifies DIRECTION only — it CANNOT rank ECG
A follow-up ("the ECG gap between gem5 and Sniper is too big to conclude") exposed a real
limit. **ECG's relative rank flips across simulators** on this cell:

| sim | GRASP | P-OPT | ECG | ECG rank |
|---|---|---|---|---|
| cache_sim | 0.5319 | 0.4763 | 0.5709 | worst (3rd) |
| gem5 | 0.5621 | 0.4686 | 0.5824 | worst (3rd) |
| Sniper | 0.4824 | 0.4816 | 0.4638 | **best (1st)** |

cache_sim and gem5 AGREE (ECG worst here); **Sniper is the outlier.** Root cause: on this
SYNTHETIC kron cell with `rrip_first`, **ECG's epoch is inert** — the eviction trace shows
`curEpoch=0` in all three sims, and the Kronecker structure does not reward the epoch by
design. So this cell tests only ECG's *degree-insertion* path, where ECG diverges slightly
and inconsistently from GRASP across simulators (a substrate difference in the degree path,
±2–4pp with a sign that flips). **No ECG *quality* conclusion is valid from this cell in
any simulator** — it only certifies that every policy beats LRU (direction).

## Where ECG conclusions ARE valid (and cross-sim consistent)
- **cache_sim is the functional authority** for ECG's advantage on real graphs (PR epoch on
  web-Google/soc-pokec within <1pp of P-OPT; BFS/SSSP traversal +3–7pp over charged P-OPT).
- The **BFS/SSSP traversal advantage is the robust cross-sim result**: it is degree-based
  (order-independent, no epoch needed), so it is exercisable on both small (gem5-feasible)
  and large (Sniper) graphs. Spot-check — BFS on kron_s16_k4 (`-r 212` hub, 128kB): LRU
  0.9515, GRASP 0.8872, **POPT 0.8913 (worse than GRASP)**, **ECG 0.8613 (best)** — the same
  ordering as the real-graph traversal cells, and gem5-feasible.
- **gem5's role is a small-graph mechanism/ISA spot-check**, not the ECG-advantage graph
  (its cycle-accurate model can't run the large real graphs where the PR-epoch advantage
  lives). Reading ECG's *rank* off gem5's synthetic equivalence cell over-reads it.

**Recommendation:** for the paper's cross-sim ECG story, use the **BFS/SSSP traversal
advantage** (robust, feasible on all 3 sims), and take the PR-epoch advantage from cache_sim
(authority) + Sniper (scale). Do NOT rank ECG from the kron PR equivalence cell.

## Two gotchas recorded (for future showcase runs)
1. **gem5 ECG mode-6 is slow** — it exceeds roi_matrix's default `--timeout-gem5=900s` on
   this cell (LRU/GRASP/POPT finish in ~5-6 min). Re-run the ECG cell with
   `--timeout-gem5 3600` (it completed in ~22 min).
2. **gem5 stats have 2 sections** — section 1 = the main kernel ROI (~1.19B insts),
   section 2 = a tiny ~5.5M-inst tail (noisy). `three_sim_showcase.py` takes the LAST
   row (section 2), which is anomalous for ECG (0.7248 vs the true ROI 0.5824). Read
   **section 1** for the kernel miss rate. (Candidate showcase fix: prefer section 1.)
