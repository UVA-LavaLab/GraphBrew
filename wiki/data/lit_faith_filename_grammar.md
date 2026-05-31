# Gate 264 — wiki/data artifact filename grammar registry

Status: **active**

## Totals

- n_files: 326
- n_stems: 131
- n_json: 130
- n_md: 127
- n_csv: 69
- n_subdirs: 1
- n_catalog: 126
- n_catalog_stems: 126

## Rules

- **F1** — every file in wiki/data/ matches `^[a-z][a-z0-9_]*\.(json|md|csv)$`
- **F2** — every .json has a matching .md sibling OR is in MD_OPTIONAL_STEMS
- **F3** — every .md has a matching .json sibling OR is in JSON_OPTIONAL_STEMS
- **F4** — every catalog.artifact path exists on disk; catalog id+path unique
- **F5** — every catalog.artifact extension is in {.json, .csv}
- **F6** — every wiki/data stem is in artifact_catalog OR in IMPLICIT_PAPER_PIPELINE_STEMS
- **F7** — every wiki/data subdirectory matches DOCUMENTED_SUBDIR_RE (paper_pipeline_<YYYYMMDD>...)

## Allow-lists

- MD_OPTIONAL_STEMS: `['ecg_gem5_parity_postfix', 'ecg_pfx_vs_droplet_postfix', 'ecg_sniper_parity_postfix', 'ecg_substrate_parity_postfix']`
- JSON_OPTIONAL_STEMS: `['literature_reproduction_summary']`
- META_ARTIFACT_STEMS: `['artifact_catalog']`

## Subdirectories

- `paper_pipeline_20260528`

## Violations

None.
