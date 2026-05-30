# L3 regime-classifier consistency (gate 245)

**Status:** active  •  classifiers: 5  •  families: 4  •  violations: 0

## Rules
- **R1** — every registered classifier resolves to a callable
- **R2** — byte-input classifiers return only declared-vocabulary labels
- **R3** — byte-input classifiers within a family agree on the canonical L3 grid
- **R4** — every regime-classifier function in ECG dir is registered
- **R5** — non-byte-label classifiers have signature + note

## Families
### `tiny_small_large_v1`
- signature: `byte_label`
- vocabulary: ['unknown', 'tiny', 'small', 'large']
- members:
  - `_l3_regime` in `scripts/experiments/ecg/policy_winner_table.py`
  - `_l3_regime` in `scripts/experiments/ecg/popt_vs_grasp_report.py`

### `tiny_small_large_v2_oracle_gap`
- signature: `byte_label`
- vocabulary: ['unknown', 'tiny', 'small', 'large']
- members:
  - `_regime` in `scripts/experiments/ecg/oracle_gap_report.py`

### `wss_range`
- signature: `kb_range`
- vocabulary: ['sub-WSS', 'post-WSS', 'mixed']
- members:
  - `_classify_regime` in `scripts/experiments/ecg/cross_tool_lru_regime.py`

### `wss_ratio`
- signature: `ratio`
- vocabulary: ['under_wss', 'near_wss', 'over_wss']
- members:
  - `_wss_regime` in `scripts/experiments/ecg/wss_relative_l3.py`

## Canonical-grid classification

| classifier | 1kB | 4kB | 16kB | 32kB | 64kB | 128kB | 256kB | 512kB | 1MB | 2MB | 4MB | 8MB | 16MB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `tiny_small_large_v1::_l3_regime@scripts/experiments/ecg/policy_winner_table.py` | `unknown` | `tiny` | `tiny` | `unknown` | `small` | `unknown` | `small` | `unknown` | `large` | `large` | `large` | `large` | `unknown` |
| `tiny_small_large_v1::_l3_regime@scripts/experiments/ecg/popt_vs_grasp_report.py` | `unknown` | `tiny` | `tiny` | `unknown` | `small` | `unknown` | `small` | `unknown` | `large` | `large` | `large` | `large` | `unknown` |
| `tiny_small_large_v2_oracle_gap::_regime@scripts/experiments/ecg/oracle_gap_report.py` | `unknown` | `tiny` | `tiny` | `tiny` | `tiny` | `unknown` | `small` | `unknown` | `large` | `large` | `large` | `large` | `unknown` |

**0 violations** — every regime classifier is registered, vocab-clean, in-family agreement holds, and non-default signatures are documented.
