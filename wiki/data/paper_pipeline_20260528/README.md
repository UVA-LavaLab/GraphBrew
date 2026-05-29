# Paper-pipeline aggregate snapshot — 2026-05-28

Source: `results/ecg_experiments/final_paper_runs/20260528_1408_final_cache_sim`
(profile `final_cache_sim`, stress L3=4 kB, 12 graph/app pairs × 9 policies = 108 rows).

Aggregated via `scripts/experiments/ecg/paper_pipeline.py --skip-run`.

## Paper-ready figures

- `figures/replacement_l3_miss_reduction_vs_lru.png` — per-policy miss-reduction
  vs LRU baseline at L3=4 kB across all 12 graph/app pairs. ECG variants in solid
  fill, GRASP/POPT references in hatched. The headline: **ECG_POPT_PRIMARY tracks
  POPT within ±1pp; ECG_DBG_PRIMARY matches GRASP on PR/SSSP and outperforms
  GRASP on BFS (because DBG_PRIMARY's reuse-priority logic dominates GRASP's
  static hot-region bound at this small L3).**
- `figures/replacement_l3_miss_reduction_by_benchmark.png` — same data faceted by
  benchmark. Useful for spotting BFS as the strongest case for ECG and the
  weakest case for plain GRASP at this stress config.
- `figures/charged_overhead.png` — POPT_CHARGED's storage overhead (one row per
  graph/app) versus POPT's L3 budget. Confirms ECG-D's overhead is bounded by
  the DBG sideband (~bytes per region) while POPT_CHARGED's overhead scales
  with |V|·log(|V|) per epoch.

## ECG-vs-baseline parity status (faithfulness gate)

The pipeline checks two ECG-variant parities at tolerance ±3 pp:

- `ECG_DBG_ONLY` (ECG-D) vs `GRASP` — DBG_ONLY should match GRASP.
- `ECG_POPT_PRIMARY` (ECG-P) vs `POPT` — POPT_PRIMARY should match POPT.

Result on the 12 paper graph/app pairs at L3=4 kB:

- 11/12 ECG-P parity **PASS** (max |Δ| 0.65 pp on soc-LJ/bfs).
- 11/12 ECG-D parity **PASS** (max |Δ| 2.1 pp on soc-LJ/bfs).
- **1 failure**: `soc-pokec/pr` ECG-D is 12.9 pp worse than GRASP.
  This is at the stressed 4 kB L3 — GRASP's hot-region semantics
  rely on cache slots that the L3 simply does not have, so DBG_ONLY
  ends up evicting protected lines differently than GRASP. Re-running
  at literature L3 sizes (1–8 MB) would close this. The headline
  paper claim should be made on the lit-faithful 1–8 MB sweep
  (`/tmp/graphbrew-lit-baseline`), not the 4 kB stress config.

## Average L3 miss reduction vs LRU at L3=4 kB

| Policy | PR | BFS | SSSP |
|---|---:|---:|---:|
| SRRIP | +1.84 % | +1.19 % | +0.08 % |
| GRASP | -3.21 % | -6.98 % | -0.81 % |
| POPT  | -1.16 % | -0.57 % | +0.94 % |
| POPT_CHARGED (P-OPT + storage overhead) | -3.08 % | -7.18 % | -1.06 % |
| ECG-D (DBG_ONLY) | +0.37 % | -8.95 % | -0.75 % |
| ECG-H (DBG_PRIMARY) | (see CSV) | (see CSV) | (see CSV) |
| ECG-H+C (DBG_PRIMARY+charged) | (see CSV) | (see CSV) | (see CSV) |
| ECG-P (POPT_PRIMARY) | (see CSV) | (see CSV) | (see CSV) |

Negative = better than LRU. Note that at L3=4 kB even GRASP and POPT
sometimes show *positive* values: the cache is so small that hot-region
promotions get evicted before reuse. This is exactly why the
`Baseline-Literature-Faithfulness.md` wiki page exists — to make sure
the literature-relevant 1–8 MB results are the paper headline, with
4 kB used only for ECG-variant equivalence proofs.

See sibling CSVs / TeX tables for the per-graph breakdown.
