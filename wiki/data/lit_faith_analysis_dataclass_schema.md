# Analysis dataclass schema (gate 281)

Locks the per-row analysis dataclasses across the four ECG analysis modules that aggregate sweep observations into paper-ready tables and invariant verdicts — extends gate 280's receiver-side registry beyond the orchestration carriers into the analysis layer.

**Classes:** 6
**Fields:** 41
**Violations:** 0

## Registry

### `scripts/experiments/ecg/paper_baseline_table.py` :: `Row` — `@dataclass(frozen=True)`

| field | annotation | default |
|---|---|---|
| `graph` | `str` | `none` |
| `app` | `str` | `none` |
| `l3_size` | `str` | `none` |
| `miss` | `dict[str, float]` | `none` |
| `delta` | `dict[str, float]` | `none` |
| `verdict` | `dict[str, str]` | `none` |
| `accesses` | `int` | `none` |

### `scripts/experiments/ecg/literature_baselines.py` :: `LiteratureClaim` — `@dataclass(frozen=True)`

| field | annotation | default |
|---|---|---|
| `graph` | `str` | `none` |
| `app` | `str` | `none` |
| `l3_size` | `str` | `none` |
| `policy` | `str` | `none` |
| `expected_sign` | `str` | `none` |
| `min_abs_delta_pct` | `float | None` | `none` |
| `max_abs_delta_pct` | `float | None` | `none` |
| `tolerance_pct` | `float` | `none` |
| `rationale` | `str` | `none` |
| `citation` | `str` | `none` |

### `scripts/experiments/ecg/literature_baselines.py` :: `CacheOrg` — `@dataclass(frozen=True)`

| field | annotation | default |
|---|---|---|
| `name` | `str` | `none` |
| `l1d_size` | `str` | `none` |
| `l1d_ways` | `str` | `none` |
| `l2_size` | `str` | `none` |
| `l2_ways` | `str` | `none` |
| `l3_size` | `str` | `none` |
| `l3_ways` | `str` | `none` |
| `line_size` | `str` | `none` |
| `rationale` | `str` | `none` |

### `scripts/experiments/ecg/corpus_diversity.py` :: `GraphProfile` — `@dataclass()`

| field | annotation | default |
|---|---|---|
| `graph` | `str` | `none` |
| `log_path` | `str` | `none` |
| `nodes` | `int` | `literal:0` |
| `edges` | `int` | `literal:0` |
| `edges_directed` | `bool` | `literal:False` |
| `features` | `dict` | `factory:field(default_factory=dict)` |

### `scripts/experiments/ecg/gem5_anchor_summary.py` :: `CellSummary` — `@dataclass()`

| field | annotation | default |
|---|---|---|
| `graph` | `str` | `none` |
| `app` | `str` | `none` |
| `l3_size` | `str` | `none` |
| `miss_rate_by_policy` | `dict[str, float]` | `factory:field(default_factory=dict)` |
| `ok_rows` | `int` | `literal:0` |
| `error_rows` | `int` | `literal:0` |

### `scripts/experiments/ecg/gem5_anchor_summary.py` :: `AnchorInvariant` — `@dataclass()`

| field | annotation | default |
|---|---|---|
| `name` | `str` | `none` |
| `status` | `str` | `none` |
| `detail` | `str` | `none` |

## ✅ No violations
