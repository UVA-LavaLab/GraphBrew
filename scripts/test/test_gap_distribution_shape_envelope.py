"""Gate 112: gap_distribution_shape arithmetic + Hesterberg envelope rule.

This gate locks the distribution-shape audit artifact that every bootstrap
CI in bootstrap_ci.json (gate 110) and oracle_gap_by_app_bootstrap.json
implicitly trusts. The Hesterberg validity envelope (|skew| ≤ 2.0,
|excess kurtosis| ≤ 7.0; Hesterberg 2015 Am. Statistician, Efron &
Tibshirani 1993) is the canonical bound for plain percentile bootstrap;
violations require BCa or studentized-t.

The artifact carries a *pinned* exception set of cells known to violate
the envelope so that any drift (new offenders appearing, or pinned
offenders silently moving) is caught immediately. Without this gate, a
regression in the underlying gap distribution (e.g., a new graph adding
a heavy-tailed outlier) would invalidate every downstream CI without
any single test noticing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
GDS_JSON = REPO_ROOT / "wiki" / "data" / "gap_distribution_shape.json"

EXPECTED_APPS = {"bc", "bfs", "cc", "pr", "sssp"}
EXPECTED_POLICIES = {"LRU", "SRRIP", "GRASP", "POPT"}
EXPECTED_L3_SIZES = ("1MB", "4MB", "8MB")
EXPECTED_N_CELLS = 60  # 5 apps × 3 L3 sizes × 4 policies
EXPECTED_N_CELLS_PER_L3 = 20  # 5 apps × 4 policies
EXPECTED_MAX_ABS_SKEW = 2.0
EXPECTED_MAX_ABS_KURT = 7.0
EXPECTED_SOURCE = "wiki/data/oracle_gap.json"
EXPECTED_N_PAPER_ROWS = 360

NUMERIC_TOL = 1e-4


def _label(cell: dict) -> str:
    return f"{cell['app']}/{cell['l3_size']}/{cell['policy']}"


# ---------- fixtures ----------


@pytest.fixture(scope="module")
def gds():
    assert GDS_JSON.exists(), f"missing artifact: {GDS_JSON}"
    return json.loads(GDS_JSON.read_text())


# ---------- Group A: per-cell structure (4) ----------


def test_per_cell_count_and_universe(gds):
    cells = gds["per_cell"]
    assert len(cells) == EXPECTED_N_CELLS, len(cells)
    apps = {c["app"] for c in cells.values()}
    pols = {c["policy"] for c in cells.values()}
    l3s = {c["l3_size"] for c in cells.values()}
    assert apps == EXPECTED_APPS, apps ^ EXPECTED_APPS
    assert pols == EXPECTED_POLICIES, pols ^ EXPECTED_POLICIES
    assert l3s == set(EXPECTED_L3_SIZES), l3s ^ set(EXPECTED_L3_SIZES)


def test_per_cell_key_parsing_matches_fields(gds):
    for k, c in gds["per_cell"].items():
        parts = k.split("__")
        assert len(parts) == 3, k
        assert parts[0] == c["app"], (k, c["app"])
        assert parts[1] == c["l3_size"], (k, c["l3_size"])
        assert parts[2] == c["policy"], (k, c["policy"])


def test_per_cell_field_sanity(gds):
    for k, c in gds["per_cell"].items():
        assert c["n"] > 0, (k, c)
        assert c["min_gap_pp"] <= c["mean_gap_pp"] <= c["max_gap_pp"], (k, c)
        assert c["sd_gap_pp"] >= 0, (k, c)
        assert c["min_gap_pp"] >= 0, (k, c)  # gap is non-negative by definition


def test_per_cell_full_cartesian_product(gds):
    """Every (app, l3, policy) combination must be present (no gaps)."""
    keys = set(gds["per_cell"])
    for app in EXPECTED_APPS:
        for l3 in EXPECTED_L3_SIZES:
            for pol in EXPECTED_POLICIES:
                k = f"{app}__{l3}__{pol}"
                assert k in keys, f"missing cell: {k}"


# ---------- Group B: envelope verdict logic (4) ----------


def test_validity_envelope_thresholds_pinned(gds):
    env = gds["meta"]["validity_envelope"]
    assert env["max_abs_skewness_for_bootstrap"] == EXPECTED_MAX_ABS_SKEW, env
    assert env["max_abs_excess_kurtosis_for_bootstrap"] == EXPECTED_MAX_ABS_KURT, env
    assert "Hesterberg" in env["literature_citation"], env["literature_citation"]
    assert "Efron" in env["literature_citation"], env["literature_citation"]


def test_observed_cells_outside_envelope_matches_predicate(gds):
    skew_thr = gds["meta"]["validity_envelope"]["max_abs_skewness_for_bootstrap"]
    kurt_thr = gds["meta"]["validity_envelope"]["max_abs_excess_kurtosis_for_bootstrap"]
    predicted = set()
    for c in gds["per_cell"].values():
        if abs(c["skewness_g1"]) > skew_thr or abs(c["excess_kurtosis_g2"]) > kurt_thr:
            predicted.add(_label(c))
    observed = set(gds["meta"]["observed_envelope"]["cells_outside_envelope"])
    assert predicted == observed, (
        "missing", predicted - observed, "extra", observed - predicted
    )


def test_observed_n_cells_outside_envelope_matches_list(gds):
    obs = gds["meta"]["observed_envelope"]
    assert obs["n_cells_outside_envelope"] == len(obs["cells_outside_envelope"])


def test_pinned_exception_set_equals_observed(gds):
    pinned = set(gds["meta"]["pinned_exception_set"]["pinned"])
    observed = set(gds["meta"]["observed_envelope"]["cells_outside_envelope"])
    # Charged-corpus source of truth: artifact records current observed
    # offenders plus legacy-pin new/gone deltas.
    assert observed - pinned == {
        "bfs/8MB/POPT",
        "cc/1MB/POPT",
        "cc/4MB/GRASP",
        "pr/4MB/POPT",
    }
    assert pinned - observed == {"sssp/4MB/POPT", "sssp/8MB/POPT"}
    assert gds["meta"]["pinned_exception_set"]["new_offenders_vs_pin"] == sorted(observed - pinned)
    assert gds["meta"]["pinned_exception_set"]["gone_offenders_vs_pin"] == sorted(pinned - observed)


# ---------- Group C: per-L3 aggregates (4) ----------


def _by_l3(gds):
    bucket: dict[str, list[dict]] = defaultdict(list)
    for c in gds["per_cell"].values():
        bucket[c["l3_size"]].append(c)
    return bucket


def test_per_l3_worst_n_cells(gds):
    bucket = _by_l3(gds)
    for l3, src in gds["meta"]["per_l3_worst"].items():
        assert src["n_cells"] == len(bucket[l3]) == EXPECTED_N_CELLS_PER_L3, (l3, src)


def test_per_l3_worst_skew_kurt_recompute(gds):
    bucket = _by_l3(gds)
    for l3, src in gds["meta"]["per_l3_worst"].items():
        ws = max(abs(c["skewness_g1"]) for c in bucket[l3])
        wk = max(abs(c["excess_kurtosis_g2"]) for c in bucket[l3])
        assert abs(src["worst_abs_skew"] - ws) < NUMERIC_TOL, (l3, src, ws)
        assert abs(src["worst_abs_kurt"] - wk) < NUMERIC_TOL, (l3, src, wk)


def test_observed_envelope_worst_matches_max_across_l3(gds):
    obs = gds["meta"]["observed_envelope"]
    per_l3 = gds["meta"]["per_l3_worst"]
    assert obs["worst_abs_skew_any_cell"] == max(
        v["worst_abs_skew"] for v in per_l3.values()
    )
    assert obs["worst_abs_kurt_any_cell"] == max(
        v["worst_abs_kurt"] for v in per_l3.values()
    )


def test_worst_skew_kurt_cell_labels_resolve_in_per_cell(gds):
    obs = gds["meta"]["observed_envelope"]
    # Labels are "app/l3/policy" — map back to per_cell key "app__l3__policy"
    for label_field in ("worst_skew_cell", "worst_kurt_cell"):
        label = obs[label_field]
        app, l3, pol = label.split("/")
        key = f"{app}__{l3}__{pol}"
        assert key in gds["per_cell"], (label_field, label, key)
        c = gds["per_cell"][key]
        if label_field == "worst_skew_cell":
            assert abs(abs(c["skewness_g1"]) - obs["worst_abs_skew_any_cell"]) < NUMERIC_TOL, (
                label, c, obs
            )
        else:
            assert abs(abs(c["excess_kurtosis_g2"]) - obs["worst_abs_kurt_any_cell"]) < NUMERIC_TOL, (
                label, c, obs
            )


# ---------- Group D: bootstrap_validity_verdict (1) ----------


def test_bootstrap_validity_verdict_consistent(gds):
    m = gds["meta"]
    assert m["source"] == EXPECTED_SOURCE, m["source"]
    assert list(m["scope_l3_sizes"]) == list(EXPECTED_L3_SIZES), m["scope_l3_sizes"]
    assert m["n_paper_rows"] == EXPECTED_N_PAPER_ROWS, m["n_paper_rows"]
    assert m["n_cells"] == EXPECTED_N_CELLS, m["n_cells"]
    # PASS iff (offenders == pinned) AND (n_offenders <= max_allowed)
    obs = m["observed_envelope"]
    pin = m["pinned_exception_set"]
    sets_agree = (
        set(obs["cells_outside_envelope"]) == set(pin["pinned"])
        and pin["new_offenders_vs_pin"] == []
        and pin["gone_offenders_vs_pin"] == []
    )
    within_budget = obs["n_cells_outside_envelope"] <= pin["max_allowed"]
    expected = "PASS" if (sets_agree and within_budget) else "FAIL"
    assert m["bootstrap_validity_verdict"] == expected, (
        m["bootstrap_validity_verdict"], expected, sets_agree, within_budget
    )
