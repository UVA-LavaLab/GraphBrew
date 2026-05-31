# Receiver dataclass schema (gate 280)

Locks the orchestration-carrier dataclasses on the RECEIVING side of the publish-reproduction subprocess contract — analog of gate 279's sender-side `Job`.

**Classes:** 3
**Fields:** 13
**Violations:** 0

## Registry

### `scripts/experiments/ecg/roi_matrix.py` :: `PolicySpec` — `@dataclass(frozen=True)`

| field | annotation | default |
|---|---|---|
| `label` | `str` | `none` |
| `policy` | `str` | `none` |
| `ecg_mode` | `str | None` | `literal:None` |
| `charge_popt_overhead` | `bool` | `literal:False` |

### `scripts/experiments/ecg/proof_matrix.py` :: `Ablation` — `@dataclass(frozen=True)`

| field | annotation | default |
|---|---|---|
| `label` | `str` | `none` |
| `group` | `str` | `none` |
| `policy` | `str` | `none` |
| `pfx_mode` | `int` | `literal:0` |
| `pfx_lookahead` | `int` | `literal:0` |
| `note` | `str` | `literal:''` |

### `scripts/experiments/ecg/proof_matrix.py` :: `AdaptiveSelector` — `@dataclass(frozen=True)`

| field | annotation | default |
|---|---|---|
| `label` | `str` | `none` |
| `candidates` | `tuple[str, ...]` | `none` |
| `note` | `str` | `none` |

## ✅ No violations
