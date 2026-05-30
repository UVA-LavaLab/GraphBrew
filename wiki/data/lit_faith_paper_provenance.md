# Paper-table CSV provenance (gate 250)

**Status:** active  •  pipeline dir: `wiki/data/paper_pipeline_20260528`  •  pairs: 5/5  •  tracked key columns: 14  •  tex rows: 78  •  csv rows: 85  •  violations: 0

## Rules
- **P1** — every registered .tex and .csv file exists
- **P2** — tex data-row count ≤ csv data-row count (subset)
- **P3** — every tex key-column value traces to csv (subset)
- **P4** — every declared key column exists in tex/csv header
- **P5** — no empty value in tracked CSV key columns
- **P6** — no unregistered CSV sibling of a registered .tex
- **P7** — every registered csv has a non-empty header row

## Registered pairs

| tex_file | csv_file | normalizer | key columns | tex rows | csv rows |
|---|---|---|---|---:|---:|
| `ecg_mode_overhead_summary.tex` | `ecg_mode_overhead_summary.csv` | `strip` | `policy`⇄`policy_short` | 4 | 4 |
| `faithfulness_summary.tex` | `faithfulness_summary.csv` | `latex` | `check`⇄`check`, `benchmark`⇄`benchmark`, `prefetcher`⇄`prefetcher`, `candidate`⇄`candidate_short` | 24 | 24 |
| `popt_charged_overhead.tex` | `popt_charged_overhead.csv` | `latex` | `charged`⇄`charged_policy`, `oracle`⇄`oracle_policy`, `benchmark`⇄`benchmark` | 20 | 24 |
| `popt_storage_overhead_summary.tex` | `popt_storage_overhead_summary.csv` | `latex` | `policy`⇄`policy_short`, `benchmark`⇄`benchmark`, `prefetcher`⇄`prefetcher` | 6 | 6 |
| `roi_policy_summary.tex` | `roi_policy_summary.csv` | `latex` | `policy`⇄`policy_short`, `benchmark`⇄`benchmark`, `prefetcher`⇄`prefetcher` | 24 | 27 |

**0 violations** — every shipped paper-table .tex row traces 1:1 to the sibling CSV, and every tracked key column matches as a multiset.
