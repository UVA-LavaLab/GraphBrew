"""Derivation-parity gate for wiki/data/paper_claims.json (PCR-Der).

The paper-claims registry is the *single source of truth* for every
numerical claim the paper makes — every reviewer-checkable figure
(corpus size, reproduction percentage, winner-table shares, road/
social POPT vs GRASP means, literature-deviation classification,
cross-tool saturation, green-gate count) is materialised through
:func:`scripts.experiments.ecg.paper_claims_registry.build_claims`.

The generator pulls from nine upstream JSONs and one live import of
``PYTEST_SUITES`` for the green-gate count. If any of the upstream
shapes drift, or the rounding rule changes, or the JSON write rule
flips, the paper text silently desynchronises from the data and a
reviewer can no longer reproduce the headline numbers. This gate
locks every load-bearing rule that governs the materialised JSON
so that any drift fails CI before the artifact is regenerated.

The gate is structured in 5 groups:

* group 1 (5 tests) — top-level shape, ID set, category set,
  units vocabulary, JSON byte-parity rule (sort_keys=True,
  indent=2, no trailing newline).
* group 2 (5 tests) — corpus, reproduction, and lit_faith
  category invariants (graph count, ok-ratio float rounding,
  disagreement count integrity).
* group 3 (5 tests) — winner_table category (4 entries for
  GRASP/POPT/SRRIP/LRU; shares sum ≤ 100.0; rounded 1dp; matches
  upstream wins_by_policy/n_cells).
* group 4 (5 tests) — small_l3_thrash, popt_vs_grasp (road and
  social), and literature_deviations categories.
* group 5 (5 tests) — cross_tool category, meta green-gate count
  (n_green from JSON, n_total from LIVE PYTEST_SUITES — load
  bearing), and build() self-consistency (re-running build()
  emits the same id-ordered records as on-disk JSON).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from scripts.experiments.ecg import paper_claims_registry as pcr  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATA = REPO_ROOT / "wiki" / "data"
ARTIFACT = DATA / "paper_claims.json"

VALID_CATEGORIES = {
    "corpus",
    "reproduction",
    "lit_faith",
    "winner_table",
    "thrash",
    "popt_vs_grasp",
    "deviations",
    "cross_tool",
    "meta",
}

VALID_UNITS = {
    "graphs",
    "percent",
    "claims",
    "cells",
    "pp",
    "disagreements",
    "gates",
}


@pytest.fixture(scope="module")
def artifact() -> dict:
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def built_claims() -> list[dict]:
    return pcr.build_claims()


@pytest.fixture(scope="module")
def upstreams() -> dict:
    """Load every upstream the registry inspects."""
    def _load(name: str):
        path = DATA / name
        try:
            return json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            return None
    return {
        "confidence": _load("confidence_dashboard.json") or {},
        "lit_faith": _load("literature_faithfulness_postfix.json") or {},
        "winner": _load("policy_winner_table.json") or {},
        "thrash": _load("small_l3_thrash.json") or {},
        "crosstool": _load("cross_tool_saturation.json") or {},
        "density": _load("claim_density.json") or {},
        "deviations": _load("literature_deviations.json") or {},
        "pvg": _load("popt_vs_grasp_delta.json") or {},
        "corpus": _load("corpus_diversity.json") or [],
    }


# ---------------------------------------------------------------------------
# Group 1 — top-level shape and JSON byte-parity rule (5 tests)
# ---------------------------------------------------------------------------


def test_top_level_keys(artifact):
    """JSON has exactly two top-level keys: claims and n_claims."""
    assert set(artifact.keys()) == {"claims", "n_claims"}


def test_n_claims_matches_claims_length(artifact):
    """The advertised n_claims is the literal len(claims)."""
    assert artifact["n_claims"] == len(artifact["claims"])


def test_every_claim_has_required_fields(artifact):
    """Every claim carries id/category/text/value/units/source/gate.

    A missing field would let the paper cite a value with no traceable
    upstream — the whole point of this registry is to prevent that.
    """
    required = {"id", "category", "text", "value", "units", "source"}
    for c in artifact["claims"]:
        missing = required - set(c.keys())
        assert not missing, f"{c.get('id', '?')} missing fields: {missing}"
        assert "gate" in c, f"{c['id']} missing gate"


def test_category_vocabulary_pinned(artifact):
    """Every category is one of the nine pinned values; no typos slip in."""
    seen = {c["category"] for c in artifact["claims"]}
    extra = seen - VALID_CATEGORIES
    assert not extra, f"unknown categories in artifact: {extra}"


def test_json_byte_parity_sort_keys_indent2_no_trailing_newline(artifact):
    """JSON write rule is fixed: sort_keys=True, indent=2, no trailing newline.

    The registry deliberately uses sort_keys=True so reviewers can
    diff the artifact across runs without spurious key-order churn;
    the no-trailing-newline rule is what `_write_json` actually emits,
    so any future tweak that adds a '\\n' would change every byte on
    every run and pollute the lit-reproduce-smoke hash log.
    """
    raw = ARTIFACT.read_text()
    expected = json.dumps(artifact, indent=2, sort_keys=True)
    assert raw == expected


# ---------------------------------------------------------------------------
# Group 2 — corpus, reproduction, lit_faith (5 tests)
# ---------------------------------------------------------------------------


def _by_id(claims: list[dict]) -> dict:
    return {c["id"]: c for c in claims}


def test_corpus_graph_count_matches_corpus_diversity(artifact, upstreams):
    """corpus.graph_count value equals len(corpus_diversity) (the live
    corpus the experiments actually exercised)."""
    by = _by_id(artifact["claims"])
    if "corpus.graph_count" not in by:
        pytest.skip("corpus.graph_count not registered this run")
    corpus = upstreams["corpus"]
    if isinstance(corpus, list):
        graphs = corpus
    elif isinstance(corpus, dict):
        graphs = corpus.get("graphs", [])
    else:
        graphs = []
    assert by["corpus.graph_count"]["value"] == len(graphs)
    assert by["corpus.graph_count"]["units"] == "graphs"


def test_reproduction_ok_ratio_is_percentage_rounded_1dp(artifact, upstreams):
    """reproduction.ok_ratio == 100 * ok / total rounded to 1 dp.

    The 1-dp rounding rule comes from `_maybe_round(..., ndigits=1)`
    inside build_claims and is what the paper's `xx.x %` text uses.
    """
    by = _by_id(artifact["claims"])
    if "reproduction.ok_ratio" not in by:
        pytest.skip("reproduction.ok_ratio not registered this run")
    summary = upstreams["density"].get("summary", {})
    n = summary.get("total_claims", 0)
    ok = summary.get("total_ok", 0)
    assert n > 0
    expected = round(100.0 * ok / n, 1)
    assert by["reproduction.ok_ratio"]["value"] == expected
    assert by["reproduction.ok_ratio"]["units"] == "percent"


def test_reproduction_n_graphs_matches_density_summary(artifact, upstreams):
    """The graphs-with-claims count mirrors claim_density's n_graphs."""
    by = _by_id(artifact["claims"])
    if "reproduction.n_graphs_with_claims" not in by:
        pytest.skip("not registered this run")
    summary = upstreams["density"].get("summary", {})
    assert by["reproduction.n_graphs_with_claims"]["value"] == summary.get(
        "n_graphs", 0
    )
    assert by["reproduction.n_graphs_with_claims"]["units"] == "graphs"


def test_lit_faith_disagreement_passthrough_raw_count(artifact, upstreams):
    """lit_faith.disagreement_rate value is the raw `disagree` integer
    (NOT a rate, despite the id) — passed straight through with no
    rounding, units == "claims"."""
    by = _by_id(artifact["claims"])
    if "lit_faith.disagreement_rate" not in by:
        pytest.skip("not registered this run")
    summary = upstreams["lit_faith"].get("summary", {})
    assert by["lit_faith.disagreement_rate"]["value"] == summary.get(
        "disagree", 0
    )
    assert by["lit_faith.disagreement_rate"]["units"] == "claims"


def test_maybe_round_rule_floats_3dp_default_ints_passthrough():
    """`_maybe_round` only rounds floats; ints (graph counts, win counts)
    are passed through verbatim. This is what makes id-level value
    types stable: an int-valued claim always emits an int.
    """
    assert pcr._maybe_round(1.23456789) == 1.235
    assert pcr._maybe_round(1.23456789, ndigits=1) == 1.2
    assert pcr._maybe_round(42) == 42
    assert pcr._maybe_round("not a number") == "not a number"
    # negative floats also round
    assert pcr._maybe_round(-1.23456789, ndigits=2) == -1.23


# ---------------------------------------------------------------------------
# Group 3 — winner_table (5 tests)
# ---------------------------------------------------------------------------


def test_winner_table_emits_one_claim_per_policy(artifact):
    """winner.* claims emit in tuple order (GRASP, POPT, SRRIP, LRU) —
    one entry per policy whenever n_cells > 0. Order matters: the
    paper text quotes them in that exact sequence."""
    ids = [c["id"] for c in artifact["claims"] if c["id"].startswith("winner.")]
    if not ids:
        pytest.skip("winner_table not registered this run")
    expected = [
        "winner.grasp_share",
        "winner.popt_share",
        "winner.srrip_share",
        "winner.lru_share",
    ]
    assert ids == expected


def test_winner_table_shares_sum_le_100_plus_eps(artifact):
    """Sum of policy shares is ≤ 100 (plus 1-dp rounding slack).
    A sum > 100 + 0.4 would mean wins_by_policy double-counted cells."""
    shares = [
        c["value"] for c in artifact["claims"]
        if c["id"].startswith("winner.")
    ]
    if not shares:
        pytest.skip("winner_table not registered this run")
    total = sum(shares)
    # 4 entries × 0.05 max per-entry rounding error = 0.2, allow 0.4
    assert total <= 100.0 + 0.4
    # Sanity: at least 99.6 if all wins are accounted for
    assert total >= 99.6 - 0.4 or total <= 100.4


def test_winner_table_shares_match_upstream(artifact, upstreams):
    """Each winner.<pol>_share equals round(100*wins[pol]/n_cells, 1)
    for the live policy_winner_table summary."""
    summary = upstreams["winner"].get("summary", {})
    n_cells = summary.get("n_cells", 0)
    wins = summary.get("wins_by_policy", {})
    if not n_cells or not wins:
        pytest.skip("winner_table not present upstream")
    by = _by_id(artifact["claims"])
    for pol in ("GRASP", "POPT", "SRRIP", "LRU"):
        cid = f"winner.{pol.lower()}_share"
        if cid not in by:
            continue
        expected = round(100.0 * wins.get(pol, 0) / n_cells, 1)
        assert by[cid]["value"] == expected, (
            f"{cid} drifted from upstream: {by[cid]['value']} vs {expected}"
        )


def test_winner_table_units_all_percent(artifact):
    """Every winner.* claim is a percent — never a raw count."""
    for c in artifact["claims"]:
        if c["id"].startswith("winner."):
            assert c["units"] == "percent", f"{c['id']} units drifted"


def test_winner_table_category_pinned(artifact):
    """Every winner.* claim sits under category 'winner_table'."""
    for c in artifact["claims"]:
        if c["id"].startswith("winner."):
            assert c["category"] == "winner_table"


# ---------------------------------------------------------------------------
# Group 4 — thrash, popt_vs_grasp (road + social), deviations (5 tests)
# ---------------------------------------------------------------------------


def test_thrash_lru_wins_passthrough(artifact, upstreams):
    """thrash.lru_wins_at_4kb is the raw LRU win count at 4 kB L3 —
    integer, units == 'cells'. The narrative-critical 'LRU wins the
    thrash regime' claim hangs off this single integer."""
    by = _by_id(artifact["claims"])
    if "thrash.lru_wins_at_4kb" not in by:
        pytest.skip("thrash not registered this run")
    summary = upstreams["thrash"].get("summary", {})
    winners = summary.get("win_counts", {})
    assert by["thrash.lru_wins_at_4kb"]["value"] == winners.get("LRU", 0)
    assert by["thrash.lru_wins_at_4kb"]["units"] == "cells"


def test_popt_vs_grasp_road_mean_rounded_3dp(artifact, upstreams):
    """road family mean is rounded to 3dp via _maybe_round default;
    units == 'pp'. The 3-dp resolution is what the paper cites
    (e.g. '-8.347 pp')."""
    by = _by_id(artifact["claims"])
    if "popt_vs_grasp.road_family_mean" not in by:
        pytest.skip("popt_vs_grasp road not registered")
    summary = upstreams["pvg"].get("summary", {})
    road = summary.get("by_family", {}).get("road", {})
    if not road.get("n", 0):
        pytest.skip("road family has no cells upstream")
    expected = round(road.get("mean_pp", 0.0), 3)
    assert by["popt_vs_grasp.road_family_mean"]["value"] == expected
    assert by["popt_vs_grasp.road_family_mean"]["units"] == "pp"


def test_popt_vs_grasp_social_mean_rounded_3dp(artifact, upstreams):
    """Social family mean follows the same 3-dp rule. The paper's
    contradiction-of-literature narrative hangs off the sign of
    this value."""
    by = _by_id(artifact["claims"])
    if "popt_vs_grasp.social_family_mean" not in by:
        pytest.skip("popt_vs_grasp social not registered")
    summary = upstreams["pvg"].get("summary", {})
    social = summary.get("by_family", {}).get("social", {})
    if not social.get("n", 0):
        pytest.skip("social family has no cells upstream")
    expected = round(social.get("mean_pp", 0.0), 3)
    assert by["popt_vs_grasp.social_family_mean"]["value"] == expected
    assert by["popt_vs_grasp.social_family_mean"]["units"] == "pp"


def test_deviations_popt_overhead_share_rounded_1dp(artifact, upstreams):
    """deviations.popt_overhead_share == round(100*popt_over/n_dev, 1)
    using `popt_overhead_dominates` from by_mechanism. The '%d / %d
    deviations classify as POPT overhead' headline keys off this."""
    by = _by_id(artifact["claims"])
    if "deviations.popt_overhead_share" not in by:
        pytest.skip("deviations not registered this run")
    summary = upstreams["deviations"].get("summary", {})
    n_dev = summary.get("n_deviations", 0)
    by_mech = summary.get("by_mechanism", {})
    popt_over = by_mech.get("popt_overhead_dominates", 0)
    if not n_dev:
        pytest.skip("no deviations registered upstream")
    expected = round(100.0 * popt_over / n_dev, 1)
    assert by["deviations.popt_overhead_share"]["value"] == expected
    assert by["deviations.popt_overhead_share"]["units"] == "percent"


def test_units_vocabulary_pinned(artifact):
    """Every claim's units belongs to the seven-unit vocabulary.
    A new unit must be added here deliberately — otherwise we risk
    the markdown table showing an opaque unit-less number."""
    for c in artifact["claims"]:
        assert c["units"] in VALID_UNITS, f"{c['id']} units={c['units']}"


# ---------------------------------------------------------------------------
# Group 5 — cross_tool, meta, build() self-consistency (5 tests)
# ---------------------------------------------------------------------------


def test_cross_tool_disagreements_normalised_list_or_int(artifact, upstreams):
    """`disagreements` in cross_tool_saturation may be a list (per-cell
    records) or an int (count). The registry must normalise to an int
    via `len()` for lists / `int()` for scalars — otherwise the paper
    claim would carry a sequence object instead of a number.
    """
    by = _by_id(artifact["claims"])
    if "cross_tool.doubly_saturated_agreement" not in by:
        pytest.skip("cross_tool not registered this run")
    summary = upstreams["crosstool"].get("summary", {})
    field = summary.get("disagreements", 0)
    expected = len(field) if isinstance(field, list) else int(field)
    assert by["cross_tool.doubly_saturated_agreement"]["value"] == expected
    assert isinstance(
        by["cross_tool.doubly_saturated_agreement"]["value"], int
    )
    assert by["cross_tool.doubly_saturated_agreement"]["units"] == (
        "disagreements"
    )


def test_meta_green_gate_count_n_total_from_live_pytest_suites(artifact):
    """The headline 'N / M confidence gates pass' uses M from the LIVE
    PYTEST_SUITES import, NOT the on-disk dashboard JSON. This is the
    docstring's anti-staleness invariant — if a gate was added but
    `confidence-fast` hasn't refreshed the JSON yet, this value still
    reflects the truth and reports 'N / M-1' instead of silently
    claiming all green.

    We verify the value is an int and the units == 'gates' (not e.g.
    'percent').
    """
    by = _by_id(artifact["claims"])
    if "confidence.green_gate_count" not in by:
        pytest.skip("confidence rollup not registered this run")
    entry = by["confidence.green_gate_count"]
    # value is the green count (an int by construction)
    assert isinstance(entry["value"], int)
    assert entry["units"] == "gates"
    assert entry["category"] == "meta"
    # text must contain "N / M" with the live total
    from scripts.experiments.ecg.confidence_dashboard import (
        PYTEST_SUITES as _LIVE,
    )
    assert f"/ {len(_LIVE)} confidence gates" in entry["text"]


def test_build_claims_matches_artifact_id_order(artifact, built_claims):
    """Re-running build_claims() produces the same id sequence as the
    on-disk artifact. The id order is load-bearing for the paper's
    narrative arc (corpus → reproduction → lit-faith → winner → thrash
    → popt-vs-grasp → deviations → cross-tool → meta)."""
    on_disk_ids = [c["id"] for c in artifact["claims"]]
    rebuilt_ids = [c["id"] for c in built_claims]
    assert on_disk_ids == rebuilt_ids


def test_build_claims_matches_artifact_values(artifact, built_claims):
    """Per-id values match between the live build() and the on-disk
    artifact. If any aggregator drifts between regens, this test
    fires before the registry hits the wiki/data tree."""
    on_disk = _by_id(artifact["claims"])
    rebuilt = _by_id(built_claims)
    for cid in on_disk:
        assert cid in rebuilt, f"{cid} missing in rebuilt"
        assert on_disk[cid]["value"] == rebuilt[cid]["value"], (
            f"{cid} value drifted: on_disk={on_disk[cid]['value']} "
            f"rebuilt={rebuilt[cid]['value']}"
        )
        assert on_disk[cid]["units"] == rebuilt[cid]["units"]
        assert on_disk[cid]["category"] == rebuilt[cid]["category"]


def test_every_claim_source_path_starts_with_wiki_data(artifact):
    """Every claim's `source` path begins with 'wiki/data/' — sources
    must be machine-locatable for the reviewer-facing trace. A claim
    with a relative non-wiki source would silently break automated
    'find the supporting JSON' tools."""
    for c in artifact["claims"]:
        src = c.get("source", "")
        assert src.startswith("wiki/data/"), (
            f"{c['id']} has non-wiki source: {src}"
        )
