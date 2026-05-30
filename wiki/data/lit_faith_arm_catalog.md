# Gate 261 — ECG arm catalog registry

Status: **active**

## Totals
- n_paper_policies: 9
- n_paper_charged: 2
- n_registry_arms: 9
- n_ablations: 16
- n_adaptive_selectors: 2

## Rules
- **A1** — every paper_pipeline.POLICY_ORDER non-baseline entry has a CANONICAL_ECG_ARMS entry after namespace translation
- **A2** — every POLICY_ORDER entry has POLICY_LABELS + POLICY_DESCRIPTIONS + POLICY_COLORS
- **A3** — every paper policy with suffix _CHARGED has a POLICY_HATCHES entry
- **A4** — every proof_matrix.ABLATIONS.policy is a canonical baseline or a CANONICAL_ECG_ARMS key
- **A5** — every proof_matrix.ADAPTIVE_SELECTORS.candidates entry references a real ablation label
- **A6** — proof_matrix.ABLATIONS has no duplicate labels
- **A7** — every CANONICAL_ECG_ARMS entry with parent='ECG' has at least one ABLATIONS row, OR (for _CHARGED arms) its uncharged parent has one

## Paper-policy → registry-arm map

| paper_policy | registry_key |
|---|---|
| `ECG_DBG_ONLY` | `ECG:DBG_ONLY` |
| `ECG_DBG_PRIMARY` | `ECG:DBG_PRIMARY` |
| `ECG_DBG_PRIMARY_CHARGED` | `ECG:DBG_PRIMARY_CHARGED` |
| `ECG_POPT_PRIMARY` | `ECG:POPT_PRIMARY` |
| `POPT_CHARGED` | `POPT_CHARGED` |

## Ablations

| label | group | policy | pfx_mode | lookahead |
|---|---|---|---:|---:|
| `LRU_cache_only` | `cache_alone` | `LRU` | 0 | 0 |
| `SRRIP_cache_only` | `cache_alone` | `SRRIP` | 0 | 0 |
| `GRASP_DBG_only` | `cache_alone` | `GRASP` | 0 | 0 |
| `POPT_only` | `cache_alone` | `POPT` | 0 | 0 |
| `ECG_DBG_only` | `ecg_replacement` | `ECG:DBG_ONLY` | 0 | 0 |
| `ECG_POPT_primary` | `ecg_replacement` | `ECG:POPT_PRIMARY` | 0 | 0 |
| `ECG_DBG_POPT` | `ecg_replacement` | `ECG:DBG_PRIMARY` | 0 | 0 |
| `ECG_POPT_TIE` | `ecg_replacement` | `ECG:POPT_TIE` | 0 | 0 |
| `ECG_EMBEDDED` | `ecg_replacement` | `ECG:ECG_EMBEDDED` | 0 | 0 |
| `ECG_EPOCH_EMBEDDED` | `ecg_replacement` | `ECG:ECG_EPOCH_EMBEDDED` | 0 | 0 |
| `ECG_COMBINED` | `ecg_replacement` | `ECG:ECG_COMBINED` | 0 | 0 |
| `PFX_degree_only` | `pfx_only` | `LRU` | 1 | 4 |
| `PFX_POPT_only` | `pfx_only` | `LRU` | 2 | 4 |
| `DBG_PFX` | `combined` | `ECG:DBG_ONLY` | 2 | 4 |
| `POPT_PFX` | `combined` | `ECG:POPT_PRIMARY` | 2 | 4 |
| `DBG_POPT_PFX` | `combined` | `ECG:DBG_PRIMARY` | 2 | 4 |

## Adaptive selectors

- **ECG_ADAPTIVE_ORACLE** ← `ECG_DBG_only`, `ECG_POPT_primary`, `ECG_DBG_POPT`, `ECG_POPT_TIE`, `ECG_EMBEDDED`, `ECG_EPOCH_EMBEDDED`, `ECG_COMBINED`
- **ECG_ADAPTIVE_NO_FULL_POPT** ← `ECG_DBG_only`, `ECG_DBG_POPT`, `ECG_POPT_TIE`, `ECG_EMBEDDED`, `ECG_EPOCH_EMBEDDED`, `ECG_COMBINED`

## Registry arms

| arm | parent | ablation_count |
|---|---|---:|
| `ECG:DBG_ONLY` | `ECG` | 2 |
| `ECG:DBG_PRIMARY` | `ECG` | 2 |
| `ECG:DBG_PRIMARY_CHARGED` | `ECG` | 0 |
| `ECG:ECG_COMBINED` | `ECG` | 1 |
| `ECG:ECG_EMBEDDED` | `ECG` | 1 |
| `ECG:ECG_EPOCH_EMBEDDED` | `ECG` | 1 |
| `ECG:POPT_PRIMARY` | `ECG` | 2 |
| `ECG:POPT_TIE` | `ECG` | 1 |
| `POPT_CHARGED` | `POPT` | 0 |

## Violations

None.
