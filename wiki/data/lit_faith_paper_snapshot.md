# Paper-figure data snapshot integrity (gate 244)

**Status:** active  •  snapshots found: 1  •  violations: 0

## Snapshot

- name: `paper_pipeline_20260528`
- name date: 2026-05-28
- age days: 4
- rows: 108
- policies: 9
- graphs: 4
- benchmarks: 3
- L3 sizes: 1

## Rules
- **F1** — exactly one paper_pipeline_YYYYMMDD/ dir exists
- **F2** — snapshot age ≤ 365 days
- **F3** — every row has pipeline_source_csv + pipeline_run_dir + pipeline_run_name
- **F4** — all rows share a single pipeline_run_dir
- **F5** — per (benchmark, graph, l3_size): policy_labels == POLICY_LABELS palette
- **F6** — l3_miss_rate ∈ [0.0, 1.0] (all rows); total_accesses ≥ 1 for high-activity benchmarks (pr)

**0 violations** — paper-figure snapshot is fresh, single-source, fully provenanced, with a rectangular coverage matrix.
