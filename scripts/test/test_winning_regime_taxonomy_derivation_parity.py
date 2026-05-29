"""Derivation parity gate for wiki/data/winning_regime_taxonomy.json.

Reproduces the paper's headline-figure aggregator from the upstream
policy_winner_table.json + corpus_diversity.json. The generator at
scripts/experiments/ecg/winning_regime_taxonomy.py bins each winner cell
into a (family, regime) bucket, counts winners per known policy, and
emits a textual "rule" whenever one policy dominates >=80% of cells in a
bin. Load-bearing semantics pinned here:

* GRAPH_FAMILY map (8 entries) is the single source of truth; cells with
  family=='unknown' are dropped from the matrix entirely;
* REGIME_ORDER = (tiny, small, medium, large); cells outside this set
  are dropped (medium currently has zero rows, intentionally reserved);
* KNOWN_POLICIES = (LRU, SRRIP, GRASP, POPT); anything else becomes
  'OTHER';
* RULE_THRESHOLD = 0.80 (>=80% wins extracts a rule);
* per-bin entry has total + per-policy wins + per-policy share rounded
  to 6dp + OTHER_wins;
* extracted rule short-circuits after the FIRST dominant policy
  (break statement in _extract_rules);
* rules sorted by (family, REGIME_ORDER.index(regime));
* per-cell flat sort key = (family, graph, app, l3_size).
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ART = REPO_ROOT / "wiki" / "data" / "winning_regime_taxonomy.json"
WINNERS = REPO_ROOT / "wiki" / "data" / "policy_winner_table.json"

GRAPH_FAMILY = {
    "email-Eu-core": "social", "soc-pokec": "social",
    "soc-LiveJournal1": "social", "com-orkut": "social",
    "cit-Patents": "citation", "web-Google": "web",
    "roadNet-CA": "road", "delaunay_n19": "mesh",
}
REGIME_ORDER = ("tiny", "small", "medium", "large")
KNOWN_POLICIES = ("LRU", "SRRIP", "GRASP", "POPT")
RULE_THRESHOLD = 0.80


def _art():
    return json.loads(ART.read_text())


def _winners():
    return json.loads(WINNERS.read_text())["cells"]


def _build_matrix():
    """Re-execute _bucket_winners() over the upstream winner table."""
    matrix = defaultdict(Counter)
    totals = defaultdict(int)
    for c in _winners():
        fam = GRAPH_FAMILY.get(c.get("graph", ""), "unknown")
        if fam == "unknown":
            continue
        regime = c.get("l3_regime", "")
        if regime not in REGIME_ORDER:
            continue
        pol = c.get("winner_policy", "")
        if not pol:
            continue
        bucket = pol if pol in KNOWN_POLICIES else "OTHER"
        matrix[(fam, regime)][bucket] += 1
        totals[(fam, regime)] += 1
    return matrix, totals


# Group A — meta constants ---------------------------------------------

def test_rule_threshold_pinned_at_80pct():
    assert _art()["summary"]["rule_threshold"] == RULE_THRESHOLD


def test_n_cells_matches_flat_table_length():
    art = _art()
    assert art["summary"]["n_cells"] == len(art["cells"])


def test_n_family_regime_bins_matches_by_family_regime_length():
    art = _art()
    assert art["summary"]["n_family_regime_bins"] == len(art["summary"]["by_family_regime"])


# Group B — bucketing derivation parity --------------------------------

def test_overall_winner_counts_match_flat_table():
    art = _art()
    expected = Counter(c["winner"] for c in art["cells"] if c["winner"])
    assert dict(art["summary"]["overall_winner_counts"]) == dict(expected.most_common())


def test_by_family_regime_bins_match_matrix_keys():
    art = _art()
    matrix, _ = _build_matrix()
    actual_keys = {(b["family"], b["regime"]) for b in art["summary"]["by_family_regime"]}
    assert actual_keys == set(matrix.keys())


def test_per_bin_total_matches_matrix_total():
    art = _art()
    _, totals = _build_matrix()
    for b in art["summary"]["by_family_regime"]:
        assert b["total"] == totals[(b["family"], b["regime"])]


def test_per_bin_per_policy_wins_match_matrix():
    art = _art()
    matrix, _ = _build_matrix()
    for b in art["summary"]["by_family_regime"]:
        counts = matrix[(b["family"], b["regime"])]
        for pol in KNOWN_POLICIES:
            assert b[f"{pol}_wins"] == counts.get(pol, 0)
        assert b["OTHER_wins"] == counts.get("OTHER", 0)


def test_per_bin_per_policy_share_rounded_to_6dp():
    art = _art()
    for b in art["summary"]["by_family_regime"]:
        for pol in KNOWN_POLICIES:
            wins = b[f"{pol}_wins"]
            expected = round(wins / b["total"], 6) if b["total"] else 0.0
            assert b[f"{pol}_share"] == expected, (b["family"], b["regime"], pol)


def test_per_bin_known_plus_other_wins_equal_total():
    art = _art()
    for b in art["summary"]["by_family_regime"]:
        s = sum(b[f"{p}_wins"] for p in KNOWN_POLICIES) + b["OTHER_wins"]
        assert s == b["total"]


# Group C — rules extraction parity ------------------------------------

def test_rules_only_above_threshold():
    for r in _art()["summary"]["rules"]:
        assert r["fraction"] >= RULE_THRESHOLD


def test_rules_text_format_pinned():
    """Rule text must match the exact f-string the paper quotes."""
    for r in _art()["summary"]["rules"]:
        expected = (
            f"on {r['family']}-family graphs at L3 regime "
            f'"{r["regime"]}", {r["winner"]} wins {r["wins"]}/{r["sample_size"]} cells '
            f"({r['fraction'] * 100.0:.1f}%)"
        )
        assert r["rule_text"] == expected, r


def test_rule_wins_over_sample_matches_fraction_to_6dp():
    for r in _art()["summary"]["rules"]:
        expected = round(r["wins"] / r["sample_size"], 6)
        assert r["fraction"] == expected


def test_rules_sorted_by_family_then_regime_order():
    rules = _art()["summary"]["rules"]
    keys = [(r["family"], REGIME_ORDER.index(r["regime"])) for r in rules]
    assert keys == sorted(keys)


def test_at_most_one_rule_per_bin():
    """The _extract_rules break statement enforces this."""
    rules = _art()["summary"]["rules"]
    keys = [(r["family"], r["regime"]) for r in rules]
    assert len(keys) == len(set(keys))


def test_rules_match_derived_extraction():
    """Re-derive rules from matrix; compare against artifact."""
    matrix, totals = _build_matrix()
    expected = []
    for key, counts in matrix.items():
        fam, regime = key
        total = totals[key]
        if total == 0:
            continue
        for pol, n in counts.most_common():
            frac = n / total
            if frac >= RULE_THRESHOLD:
                expected.append((fam, regime, pol, n, total, round(frac, 6)))
                break
    expected.sort(key=lambda r: (r[0], REGIME_ORDER.index(r[1])))
    actual = [
        (r["family"], r["regime"], r["winner"], r["wins"], r["sample_size"], r["fraction"])
        for r in _art()["summary"]["rules"]
    ]
    assert actual == expected


# Group D — per-cell flat table parity ---------------------------------

def test_flat_cells_sorted_by_family_graph_app_l3():
    art = _art()
    keys = [(c["family"], c["graph"], c["app"], c["l3_size"]) for c in art["cells"]]
    assert keys == sorted(keys)


def test_flat_cells_all_known_family():
    art = _art()
    for c in art["cells"]:
        assert c["family"] in {"social", "citation", "web", "road", "mesh", "unknown"}


def test_flat_cells_no_unknown_family_after_filter():
    """unknown graphs are dropped from the matrix but appear in flat table.
    Still: every flat-table family matches the GRAPH_FAMILY map."""
    art = _art()
    for c in art["cells"]:
        expected = GRAPH_FAMILY.get(c["graph"], "unknown")
        assert c["family"] == expected, c


def test_flat_cells_winners_subset_of_known_or_empty():
    art = _art()
    for c in art["cells"]:
        assert c["winner"] in set(KNOWN_POLICIES) | {""}


# Group E — corpus invariants ------------------------------------------

def test_known_policies_set_pinned_to_four():
    """Ensures upstream winner doesn't introduce a new policy silently."""
    art = _art()
    counts = art["summary"]["overall_winner_counts"]
    extras = set(counts.keys()) - set(KNOWN_POLICIES)
    assert not extras, f"unexpected winners: {extras}"


def test_at_least_one_rule_extracted():
    """Headline figure must have at least one quotable rule."""
    assert len(_art()["summary"]["rules"]) >= 1
