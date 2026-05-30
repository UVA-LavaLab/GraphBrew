# Gate 262 — ECG cross-tool aggregator schema registry

Status: **active**

## Totals

- n_aggregators: 6
- n_evidence_rows_total: 25
- canonical_tools: ['cache_sim', 'cache-sim', 'gem5', 'gem5-riscv', 'gem5-x86', 'sniper', 'sniper-sift']

## Rules

- **S1** — every CROSS_TOOL_AGGREGATORS entry's artifact exists on disk
- **S2** — every artifact is valid JSON with a top-level object
- **S3** — every declared top-level key is present in the on-disk JSON
- **S4** — every declared evidence path resolves to a non-empty list/dict
- **S5** — every declared cell-row required-key is present in every evidence row
- **S6** — every declared tools-path value is a subset of canonical ['cache_sim', 'cache-sim', 'gem5', 'gem5-riscv', 'gem5-x86', 'sniper', 'sniper-sift']
- **S7** — every declared verdict-path value has the declared type

## Aggregators

| name | purpose | top_keys | evidence_path | tools_path | verdict_path | verdict_type |
|---|---|---|---|---|---|---|
| `cross_tool_lru_regime` | Reports the LRU-vs-GRASP sub-WSS vs post-WSS regime inversion across all 3 tools (does LRU climb out of the deficit once the working-set spills L3?). | `['meta']` | `meta.tool_results` | `meta.tools` | `meta.verdict` | str |
| `cross_tool_saturation` | Reports the doubly-saturated cell census: which (graph, app, L3) tuples have BOTH cache_sim AND anchor (gem5 or sniper) collapsing all policies to within sat_floor_pp. | `['cells', 'schema_version', 'summary']` | `cells` | `-` | `summary.doubly_saturated_agree` | int |
| `cross_tool_slope_ordering` | Reports the strict per-tool slope ordering (POPT > GRASP > SRRIP > LRU at the corpus-median level) across all 3 tools. | `['meta', 'per_tool']` | `per_tool` | `meta.tools` | `meta.verdict` | str |
| `cross_tool_slope_universality` | Reports whether the steepness-band universality claim (every policy's POPT-vs-LRU steepness lies in the same band across cache_sim/gem5/sniper) holds today. | `['meta']` | `meta.tool_policies` | `meta.tools` | `meta.violations` | list |
| `cross_tool_winners` | Reports the per-cell winner-policy across cache_sim/gem5/sniper (does GRASP win in cache_sim AND in sniper, or is it split?). | `['cells', 'summary']` | `cells` | `-` | `summary.n_cells` | int |
| `anchor_cross_tool_agreement` | Reports the gem5-vs-sniper anchor-cell agreement: for the shared anchor cells, does sniper's POPT slope and gem5's POPT slope agree in sign and magnitude within thresholds? | `['checks', 'meta', 'schema', 'shared_cells', 'summary', 'verdict_ok']` | `shared_cells` | `-` | `verdict_ok` | bool |

## Aggregator status (today)

| name | exists | top_keys_ok | evidence_nonempty | rows | row_keys_ok | tools_ok | verdict_ok |
|---|---|---|---|---:|---|---|---|
| `cross_tool_lru_regime` | True | True | True | 3 | True | True | True |
| `cross_tool_saturation` | True | True | True | 7 | True | True | True |
| `cross_tool_slope_ordering` | True | True | True | 3 | True | True | True |
| `cross_tool_slope_universality` | True | True | True | 3 | True | True | True |
| `cross_tool_winners` | True | True | True | 6 | True | True | True |
| `anchor_cross_tool_agreement` | True | True | True | 3 | True | True | True |

## Violations

None.
