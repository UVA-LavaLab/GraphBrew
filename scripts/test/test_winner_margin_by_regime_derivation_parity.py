"""Derivation parity gate for ``wiki/data/winner_margin_by_regime.json``.

Locks the gate-62 winner-margin × WSS-regime artifact against its two
upstream sources so any silent drift in the cell aggregator, the
regime classifier, the percentile reducer, or the verdict predicates
trips a test before the dashboard re-publishes:

    oracle_gap.json#rows                  → per-(app,graph,l3,policy) miss_rate
    wss_relative_l3.json#meta.wss_proxies → graph WSS proxies (bytes)
                  │
       winner_margin_by_regime.py:build()
                  │
                  ▼
    wiki/data/winner_margin_by_regime.json   ← gate target

The gated claim: oracle-aware policies show LARGER winner-margins
(in pp of miss-rate over second-best) in tight-capacity regimes
(under_wss) than in loose ones (over_wss) — replacement value
shrinks as capacity loosens. PASS iff every regime records at least
one win AND at least one of {GRASP, POPT} has under_wss median
margin strictly greater than over_wss.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ARTIFACT_PATH = WIKI_DATA / "winner_margin_by_regime.json"
ORACLE_PATH = WIKI_DATA / "oracle_gap.json"
WSS_PATH = WIKI_DATA / "wss_relative_l3.json"

# Pinned mirror of generator constants.
REGIMES = ("under_wss", "near_wss", "over_wss")
POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
ORACLE_AWARE = ("GRASP", "POPT")
L3_BYTES = {
    "4kB":   4 * 1024,
    "16kB":  16 * 1024,
    "64kB":  64 * 1024,
    "256kB": 256 * 1024,
    "1MB":   1024 * 1024,
    "4MB":   4 * 1024 * 1024,
    "8MB":   8 * 1024 * 1024,
}


def _classify(l3_bytes: float, wss_bytes: float) -> str:
    """Exact mirror of generator's regime classifier."""
    ratio = l3_bytes / wss_bytes
    if ratio < 0.25:
        return "under_wss"
    if ratio > 4.0:
        return "over_wss"
    return "near_wss"


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


# ----------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def artifact() -> dict:
    if not ARTIFACT_PATH.exists():
        pytest.skip(f"missing {ARTIFACT_PATH}")
    return json.loads(ARTIFACT_PATH.read_text())


@pytest.fixture(scope="module")
def oracle_doc() -> dict:
    if not ORACLE_PATH.exists():
        pytest.skip(f"missing {ORACLE_PATH}")
    return json.loads(ORACLE_PATH.read_text())


@pytest.fixture(scope="module")
def wss_doc() -> dict:
    if not WSS_PATH.exists():
        pytest.skip(f"missing {WSS_PATH}")
    return json.loads(WSS_PATH.read_text())


@pytest.fixture(scope="module")
def reconstructed(oracle_doc, wss_doc) -> dict:
    """Re-run the generator's reducer against the same upstream blobs."""
    wss_proxies = wss_doc["meta"]["wss_proxies"]
    rows = oracle_doc["rows"]
    cells: dict[tuple, list] = defaultdict(list)
    for r in rows:
        cells[(r["app"], r["graph"], r["l3_size"])].append(r)
    margins: dict[tuple, list[float]] = defaultdict(list)
    classified, skipped = 0, 0
    for (app, graph, l3), rs in cells.items():
        if len(rs) != len(POLICIES):
            skipped += 1
            continue
        if graph not in wss_proxies or l3 not in L3_BYTES:
            skipped += 1
            continue
        regime = _classify(L3_BYTES[l3], wss_proxies[graph])
        miss = sorted([(float(r["miss_rate"]), r["policy"]) for r in rs])
        best_miss, best_pol = miss[0]
        second_miss, _ = miss[1]
        margin_pp = (second_miss - best_miss) * 100.0
        margins[(best_pol, regime)].append(margin_pp)
        classified += 1
    return {
        "classified": classified,
        "skipped": skipped,
        "margins": margins,
    }


# ----------------------------------------------------------------------
# Group A: shape & schema
# ----------------------------------------------------------------------

def test_top_level_keys(artifact):
    assert set(artifact.keys()) == {"meta", "per_policy_regime"}


def test_meta_carries_canonical_fields(artifact):
    expected = {
        "policies", "regimes", "cells_classified", "cells_skipped",
        "regime_has_wins", "shrink_evidence", "verdict",
        "verdict_invariant",
    }
    missing = expected - set(artifact["meta"].keys())
    assert not missing, f"meta missing fields: {missing}"


def test_policies_pinned(artifact):
    assert tuple(artifact["meta"]["policies"]) == POLICIES


def test_regimes_pinned(artifact):
    assert tuple(artifact["meta"]["regimes"]) == REGIMES


def test_per_policy_regime_keys_complete(artifact):
    expected = {f"{p}/{r}" for p in POLICIES for r in REGIMES}
    assert set(artifact["per_policy_regime"].keys()) == expected, (
        "per_policy_regime missing or extra (policy, regime) entries — "
        "drift in either pinned tuple would break dashboard rendering."
    )


def test_per_policy_regime_entry_shape(artifact):
    expected_keys = {
        "policy", "wss_regime", "cells_won", "median_margin_pp",
        "mean_margin_pp", "p90_margin_pp", "max_margin_pp",
    }
    for k, e in artifact["per_policy_regime"].items():
        missing = expected_keys - set(e.keys())
        assert not missing, f"{k}: per_policy_regime entry missing {missing}"


def test_verdict_invariant_string_pinned(artifact):
    expected = (
        "PASS iff every regime has at least one win AND at least one "
        "oracle-aware policy has median winner-margin strictly larger "
        "at under_wss than at over_wss"
    )
    assert artifact["meta"]["verdict_invariant"] == expected, (
        "verdict_invariant text drift — dashboard quotes this string "
        "verbatim, so changes here are reviewer-visible."
    )


# ----------------------------------------------------------------------
# Group B: cell aggregation cross-source parity
# ----------------------------------------------------------------------

def test_cells_classified_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["cells_classified"] == reconstructed["classified"]


def test_cells_skipped_matches_recomputation(artifact, reconstructed):
    assert artifact["meta"]["cells_skipped"] == reconstructed["skipped"]


def test_total_cells_won_equals_cells_classified(artifact):
    """Every classified cell produces exactly one (policy, regime) win."""
    total_wins = sum(
        e["cells_won"] for e in artifact["per_policy_regime"].values()
    )
    assert total_wins == artifact["meta"]["cells_classified"], (
        f"total wins {total_wins} ≠ cells_classified "
        f"{artifact['meta']['cells_classified']}; aggregation lost a cell"
    )


def test_per_pair_cells_won_matches_recomputation(artifact, reconstructed):
    for pol in POLICIES:
        for regime in REGIMES:
            expected = len(reconstructed["margins"].get((pol, regime), []))
            got = artifact["per_policy_regime"][f"{pol}/{regime}"]["cells_won"]
            assert got == expected, (
                f"{pol}/{regime}: cells_won drift — "
                f"recomputed {expected}, got {got}"
            )


def test_policy_and_regime_match_key(artifact):
    for k, e in artifact["per_policy_regime"].items():
        pol, regime = k.split("/")
        assert e["policy"] == pol
        assert e["wss_regime"] == regime


# ----------------------------------------------------------------------
# Group C: reducer cross-source parity
# ----------------------------------------------------------------------

def test_median_margin_matches_recomputation(artifact, reconstructed):
    for pol in POLICIES:
        for regime in REGIMES:
            xs = reconstructed["margins"].get((pol, regime), [])
            expected = round(_median(xs), 4)
            got = artifact["per_policy_regime"][f"{pol}/{regime}"]["median_margin_pp"]
            assert got == expected, (
                f"{pol}/{regime}: median_margin_pp drift — "
                f"recomputed {expected!r}, got {got!r}"
            )


def test_mean_margin_matches_recomputation(artifact, reconstructed):
    for pol in POLICIES:
        for regime in REGIMES:
            xs = reconstructed["margins"].get((pol, regime), [])
            expected = round(sum(xs) / len(xs), 4) if xs else 0.0
            got = artifact["per_policy_regime"][f"{pol}/{regime}"]["mean_margin_pp"]
            assert got == expected, (
                f"{pol}/{regime}: mean_margin_pp drift"
            )


def test_p90_margin_matches_recomputation(artifact, reconstructed):
    for pol in POLICIES:
        for regime in REGIMES:
            xs = reconstructed["margins"].get((pol, regime), [])
            expected = round(_pct(xs, 0.9), 4)
            got = artifact["per_policy_regime"][f"{pol}/{regime}"]["p90_margin_pp"]
            assert got == expected, (
                f"{pol}/{regime}: p90_margin_pp drift"
            )


def test_max_margin_matches_recomputation(artifact, reconstructed):
    for pol in POLICIES:
        for regime in REGIMES:
            xs = reconstructed["margins"].get((pol, regime), [])
            expected = round(max(xs), 4) if xs else 0.0
            got = artifact["per_policy_regime"][f"{pol}/{regime}"]["max_margin_pp"]
            assert got == expected, (
                f"{pol}/{regime}: max_margin_pp drift"
            )


def test_max_ge_p90_ge_median(artifact):
    """Reducer sanity — for any cell with wins, max ≥ p90 ≥ median."""
    for k, e in artifact["per_policy_regime"].items():
        if e["cells_won"] == 0:
            continue
        assert e["max_margin_pp"] >= e["p90_margin_pp"] >= e["median_margin_pp"], (
            f"{k}: monotonicity broken (max={e['max_margin_pp']}, "
            f"p90={e['p90_margin_pp']}, median={e['median_margin_pp']})"
        )


# ----------------------------------------------------------------------
# Group D: verdict + headline
# ----------------------------------------------------------------------

def test_regime_has_wins_matches_recomputation(artifact, reconstructed):
    expected = {
        regime: any(
            reconstructed["margins"].get((pol, regime)) for pol in POLICIES
        )
        for regime in REGIMES
    }
    assert artifact["meta"]["regime_has_wins"] == expected


def test_shrink_evidence_only_oracle_aware(artifact):
    for ev in artifact["meta"]["shrink_evidence"]:
        assert ev["policy"] in ORACLE_AWARE, (
            f"shrink_evidence includes non-oracle-aware policy "
            f"{ev['policy']!r}; only {ORACLE_AWARE} are eligible"
        )


def test_shrink_evidence_matches_recomputation(artifact, reconstructed):
    expected: list[dict] = []
    for pol in ORACLE_AWARE:
        u = _median(reconstructed["margins"].get((pol, "under_wss"), []))
        o = _median(reconstructed["margins"].get((pol, "over_wss"), []))
        if u > 0 and u > o:
            expected.append({
                "policy": pol,
                "under_median": u,
                "over_median": o,
            })
    got = artifact["meta"]["shrink_evidence"]
    assert len(got) == len(expected), (
        f"shrink_evidence length drift — expected {len(expected)}, got {len(got)}"
    )
    for e_exp, e_got in zip(expected, got):
        assert e_got["policy"] == e_exp["policy"]
        assert e_got["under_median"] == pytest.approx(e_exp["under_median"])
        assert e_got["over_median"] == pytest.approx(e_exp["over_median"])


def test_verdict_matches_and_of_predicates(artifact):
    has_all = all(artifact["meta"]["regime_has_wins"].values())
    has_shrink = len(artifact["meta"]["shrink_evidence"]) >= 1
    expected = "PASS" if (has_all and has_shrink) else "FAIL"
    assert artifact["meta"]["verdict"] == expected


def test_current_verdict_is_pass(artifact):
    assert artifact["meta"]["verdict"] == "PASS", (
        "winner_margin_by_regime regressed to FAIL — the paper's "
        "core narrative ('oracle-aware pays off most under tight "
        "capacity') is gated here, so a regression is a story-changing "
        "event, not a stylistic one."
    )


def test_at_least_one_oracle_shows_shrinking_margins(artifact):
    assert len(artifact["meta"]["shrink_evidence"]) >= 1, (
        "shrink_evidence is empty — no oracle-aware policy shows a "
        "median winner margin that strictly shrinks from under_wss "
        "to over_wss; this contradicts the paper's headline."
    )
