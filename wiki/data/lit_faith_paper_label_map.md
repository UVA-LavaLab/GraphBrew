# Paper label-map integrity (gate 242)

**Status:** active  •  labels: 9  •  sources scanned: 13 (ok 13, missing 0)  •  violations: 0

## Rules
- **G1** — every POLICY_LABELS key has matching description and color
- **G2** — every figure_label is unique across the map
- **G3** — every policy_label in tracked sources is in POLICY_LABELS (modulo allowlist)
- **G4** — latest committed policy_label_map.csv matches code byte-for-byte
- **G5** — every POLICY_LABELS key appears in at least one tracked source

## Canonical policy → figure-label map

| policy_label | figure_label | description |
|---|---|---|
| ECG_DBG_ONLY | ECG-D | ECG DBG-only mode, GRASP-equivalence check |
| ECG_DBG_PRIMARY | ECG-H | ECG DBG-primary hybrid mode |
| ECG_DBG_PRIMARY_CHARGED | ECG-H+C | ECG hybrid mode with P-OPT overhead charged |
| ECG_POPT_PRIMARY | ECG-P | ECG P-OPT-primary oracle-validation mode |
| GRASP | GRASP | GRASP degree-aware replacement |
| LRU | LRU | Least recently used baseline |
| POPT | P-OPT | P-OPT oracle-capacity replacement |
| POPT_CHARGED | P-OPT+C | P-OPT with matrix capacity and stream overhead charged |
| SRRIP | SRRIP | Static re-reference interval prediction baseline |

## Source-artifact scan

| source | status | label_count | labels |
|---|---|---:|---|
| ecg_gem5_parity_postfix.json | ok | 3 | ECG_POPT_PRIMARY, LRU, POPT |
| ecg_pfx_vs_droplet_postfix.json | ok | 1 | LRU |
| ecg_sniper_parity_postfix.json | ok | 0 | — |
| ecg_substrate_parity_postfix.json | ok | 0 | — |
| lit_faith_ecg_gem5_parity.json | ok | 0 | — |
| lit_faith_ecg_parity.json | ok | 0 | — |
| literature_faithfulness_postfix.json | ok | 6 | GRASP, LRU, POPT, POPT_GE_GRASP, POPT_NEAR_GRASP_IF_BIG_GAP, SRRIP |
| policy_winner_table.json | ok | 4 | GRASP, LRU, POPT, SRRIP |
| paper_pipeline_20260528/roi_matrix_all.csv | ok | 9 | ECG_DBG_ONLY, ECG_DBG_PRIMARY, ECG_DBG_PRIMARY_CHARGED, ECG_POPT_PRIMARY, GRASP, LRU, POPT, POPT_CHARGED, SRRIP |
| paper_pipeline_20260528/roi_policy_summary.csv | ok | 9 | ECG_DBG_ONLY, ECG_DBG_PRIMARY, ECG_DBG_PRIMARY_CHARGED, ECG_POPT_PRIMARY, GRASP, LRU, POPT, POPT_CHARGED, SRRIP |
| paper_pipeline_20260528/ecg_mode_overhead_summary.csv | ok | 4 | ECG_DBG_ONLY, ECG_DBG_PRIMARY, ECG_DBG_PRIMARY_CHARGED, ECG_POPT_PRIMARY |
| paper_pipeline_20260528/popt_storage_overhead_summary.csv | ok | 2 | ECG_DBG_PRIMARY_CHARGED, POPT_CHARGED |
| paper_pipeline_20260528/popt_charged_overhead.csv | ok | 0 | — |

**0 violations** — paper label map is consistent.
