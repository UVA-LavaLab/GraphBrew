# Paper LaTeX-table emit invariant (gate 247)

**Status:** active  •  pipeline dir: `wiki/data/paper_pipeline_20260528`  •  tables: 5/5  •  rows: 78  •  violations: 0

## Rules
- **T1** — every registered table file exists
- **T2** — registered caption matches in-file caption
- **T3** — registered tabular col-spec matches in-file spec
- **T4** — registered column-header tuple matches in-file
- **T5** — every data row has correct column count and no NaN/Inf cells
- **T6** — no unregistered .tex file in paper_pipeline dir
- **T7** — every table ends with \bottomrule\end{tabular}\end{table}

## Registered tables

| filename | col_spec | columns | rows |
|---|---|---|---:|
| `ecg_mode_overhead_summary.tex` | `llll` | `policy`, `P-OPT lookup`, `charged\_alias`, `reserved ways` | 4 |
| `faithfulness_summary.tex` | `llllllll` | `check`, `benchmark`, `prefetcher`, `reference`, `candidate`, `max tick delta (\%)`, `max LLC delta (\%)`, `pass` | 24 |
| `popt_charged_overhead.tex` | `llllll` | `charged`, `oracle`, `benchmark`, `section`, `tick delta (\%)`, `LLC delta (\%)` | 20 |
| `popt_storage_overhead_summary.tex` | `lllllll` | `policy`, `benchmark`, `prefetcher`, `reserved ways`, `reserved B`, `reserved LLC (\%)`, `matrix stream lines` | 6 |
| `roi_policy_summary.tex` | `lllll` | `policy`, `benchmark`, `prefetcher`, `avg speedup`, `avg LLC red. (\%)` | 24 |

**0 violations** — every registered paper table matches its expected caption, column-spec, header, and emits clean rows.
