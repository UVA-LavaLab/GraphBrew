"""Derivation parity gate for family_margin_replay.json (gate 190).

Regenerates per-family winner-margin distributions from the joint
oracle_gap.json#rows + wss_relative_l3.json#meta.wss_proxies upstream
and asserts byte-equality with the committed artifact. Validates the
per-family replay of the global winner-margin-shrink signal (gate 62).

Load-bearing rules:

- REGIMES = ("under_wss", "near_wss", "over_wss")
- POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
- ORACLE_AWARE = ("GRASP", "POPT") — TUPLE (NOT set), preserves order
- L3_BYTES = canonical power-of-2 byte counts
- Regime classification on STRICT bounds:
    ratio < 0.25 → under_wss; ratio > 4.0 → over_wss; else near_wss
- Cell key (app, graph, l3) — skipped iff:
    a) len(rs) != len(POLICIES)  (need all 4 policies present)
    b) graph not in wss_proxies OR l3 not in L3_BYTES
- Winner per cell = min miss_rate via sorted((miss_rate, policy))
  (tie-break by policy NAME, default tuple sort)
- Margin = (second_miss - best_miss) * 100.0  (percentage points)
- per_policy_regime is the FULL 4×3 cartesian product (12 entries),
  zero-filled when no wins; key format 'P/R'
- _median: pair-average even n; mid odd n; empty → 0.0
- _pct (bespoke percentile): s[max(0, min(n-1, int(round(p·(n-1)))))]
  (NOT numpy interpolation)
- All emitted floats rounded to 4dp
- Family qualifying iff some oracle-aware policy has wins in BOTH
  under_wss AND over_wss
- shrink_evidence: ORACLE_AWARE-ordered list of policies where
  under_median > over_median (STRICT >)
- replays = qualifying AND len(shrink_evidence) >= 1
- Iteration: sorted(by_family.keys()) — alphabetical
- new_deviating = deviating − PINNED (empty tuple today)
- verdict = PASS iff len(replays) >= 1 AND len(new_deviating) == 0
- JSON written sort_keys=True
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "wiki" / "data" / "family_margin_replay.json"
ORACLE = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
WSS = REPO_ROOT / "wiki" / "data" / "wss_relative_l3.json"

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
PINNED_DEVIATING_FAMILIES: tuple[str, ...] = ()


def _classify(l3_bytes, wss_bytes):
    ratio = l3_bytes / wss_bytes
    if ratio < 0.25:
        return "under_wss"
    if ratio > 4.0:
        return "over_wss"
    return "near_wss"


def _median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def _pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def _family_stats(family_rows, wss_proxies):
    cells = defaultdict(list)
    for r in family_rows:
        cells[(r["app"], r["graph"], r["l3_size"])].append(r)

    margins = defaultdict(list)
    classified = 0
    skipped = 0
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
        margins[(best_pol, regime)].append((second_miss - best_miss) * 100.0)
        classified += 1

    per_policy_regime = {}
    for pol in POLICIES:
        for regime in REGIMES:
            xs = margins.get((pol, regime), [])
            per_policy_regime[f"{pol}/{regime}"] = {
                "policy":            pol,
                "wss_regime":        regime,
                "cells_won":         len(xs),
                "median_margin_pp":  round(_median(xs), 4),
                "mean_margin_pp":    round(sum(xs) / len(xs), 4) if xs else 0.0,
                "p90_margin_pp":     round(_pct(xs, 0.9), 4),
                "max_margin_pp":     round(max(xs), 4) if xs else 0.0,
            }

    qualifies = False
    shrink = []
    for pol in ORACLE_AWARE:
        u_xs = margins.get((pol, "under_wss"), [])
        o_xs = margins.get((pol, "over_wss"), [])
        if u_xs and o_xs:
            qualifies = True
            u_med = _median(u_xs)
            o_med = _median(o_xs)
            if u_med > o_med:
                shrink.append({
                    "policy":       pol,
                    "under_median": round(u_med, 4),
                    "over_median":  round(o_med, 4),
                })
    return {
        "cells_classified":  classified,
        "cells_skipped":     skipped,
        "qualifying":        qualifies,
        "replays":           qualifies and len(shrink) >= 1,
        "shrink_evidence":   shrink,
        "per_policy_regime": per_policy_regime,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def artifact():
    assert ARTIFACT.exists(), f"missing {ARTIFACT}"
    return json.loads(ARTIFACT.read_text())


@pytest.fixture(scope="module")
def oracle():
    assert ORACLE.exists(), f"missing {ORACLE}"
    return json.loads(ORACLE.read_text())


@pytest.fixture(scope="module")
def wss_proxies():
    assert WSS.exists(), f"missing {WSS}"
    return json.loads(WSS.read_text())["meta"]["wss_proxies"]


@pytest.fixture(scope="module")
def derived(oracle, wss_proxies):
    by_family = defaultdict(list)
    for r in oracle["rows"]:
        by_family[r["family"]].append(r)
    per_family = {}
    qualifying = []
    replays = []
    deviating = []
    for fam in sorted(by_family.keys()):
        stats = _family_stats(by_family[fam], wss_proxies)
        per_family[fam] = stats
        if stats["qualifying"]:
            qualifying.append(fam)
            if stats["replays"]:
                replays.append(fam)
            else:
                deviating.append(fam)
    new_dev = [f for f in deviating if f not in PINNED_DEVIATING_FAMILIES]
    verdict = "PASS" if (len(replays) >= 1 and not new_dev) else "FAIL"
    return {
        "meta": {
            "policies": list(POLICIES),
            "regimes": list(REGIMES),
            "oracle_aware": list(ORACLE_AWARE),
            "qualifying_families": qualifying,
            "replay_count": len(replays),
            "replaying_families": replays,
            "deviating_families": deviating,
            "pinned_deviating_families": list(PINNED_DEVIATING_FAMILIES),
            "new_deviating_families": new_dev,
            "verdict": verdict,
        },
        "per_family": per_family,
    }


# ---------------------------------------------------------------------------
# Group A — meta constants & verdict
# ---------------------------------------------------------------------------


def test_meta_policies(artifact):
    assert artifact["meta"]["policies"] == list(POLICIES)


def test_meta_regimes(artifact):
    assert artifact["meta"]["regimes"] == list(REGIMES)


def test_meta_oracle_aware_tuple_order(artifact):
    """ORACLE_AWARE is a TUPLE — order is load-bearing for shrink_evidence emission."""
    assert artifact["meta"]["oracle_aware"] == list(ORACLE_AWARE)


def test_meta_qualifying_families_alpha(artifact):
    qf = artifact["meta"]["qualifying_families"]
    assert qf == sorted(qf)


def test_meta_qualifying_families_match(artifact, derived):
    assert artifact["meta"]["qualifying_families"] == derived["meta"]["qualifying_families"]


def test_meta_replaying_families_match(artifact, derived):
    assert artifact["meta"]["replaying_families"] == derived["meta"]["replaying_families"]


def test_meta_deviating_families_match(artifact, derived):
    assert artifact["meta"]["deviating_families"] == derived["meta"]["deviating_families"]


def test_meta_replay_count_matches_replaying_len(artifact):
    assert artifact["meta"]["replay_count"] == len(artifact["meta"]["replaying_families"])


def test_meta_qualifying_partitioned(artifact):
    """qualifying == replaying + deviating (set equality, partition)."""
    qf = set(artifact["meta"]["qualifying_families"])
    rep = set(artifact["meta"]["replaying_families"])
    dev = set(artifact["meta"]["deviating_families"])
    assert rep | dev == qf
    assert rep & dev == set()


def test_meta_pinned_empty(artifact):
    assert artifact["meta"]["pinned_deviating_families"] == list(PINNED_DEVIATING_FAMILIES)


def test_meta_new_deviating_formula(artifact):
    pinned = set(artifact["meta"]["pinned_deviating_families"])
    expected = [f for f in artifact["meta"]["deviating_families"] if f not in pinned]
    assert artifact["meta"]["new_deviating_families"] == expected


def test_meta_verdict_closed_form(artifact):
    m = artifact["meta"]
    expected = "PASS" if (m["replay_count"] >= 1 and not m["new_deviating_families"]) else "FAIL"
    assert m["verdict"] == expected


# ---------------------------------------------------------------------------
# Group B — per_family shape
# ---------------------------------------------------------------------------


def test_per_family_record_shape(artifact):
    for fam, rec in artifact["per_family"].items():
        assert set(rec.keys()) == {
            "cells_classified",
            "cells_skipped",
            "qualifying",
            "replays",
            "shrink_evidence",
            "per_policy_regime",
        }


def test_per_family_per_policy_regime_full_grid(artifact):
    """All 4 policies × 3 regimes = 12 entries, regardless of empty wins."""
    expected = {f"{p}/{r}" for p in POLICIES for r in REGIMES}
    for fam, rec in artifact["per_family"].items():
        assert set(rec["per_policy_regime"].keys()) == expected


def test_per_family_pp_regime_record_shape(artifact):
    for fam, rec in artifact["per_family"].items():
        for key, entry in rec["per_policy_regime"].items():
            assert set(entry.keys()) == {
                "policy", "wss_regime", "cells_won",
                "median_margin_pp", "mean_margin_pp",
                "p90_margin_pp", "max_margin_pp",
            }
            pol, reg = key.split("/", 1)
            assert entry["policy"] == pol
            assert entry["wss_regime"] == reg


def test_per_family_cells_won_nonneg(artifact):
    for fam, rec in artifact["per_family"].items():
        for entry in rec["per_policy_regime"].values():
            assert entry["cells_won"] >= 0


def test_per_family_zero_fill_when_no_wins(artifact):
    for fam, rec in artifact["per_family"].items():
        for entry in rec["per_policy_regime"].values():
            if entry["cells_won"] == 0:
                assert entry["median_margin_pp"] == 0.0
                assert entry["mean_margin_pp"] == 0.0
                assert entry["p90_margin_pp"] == 0.0
                assert entry["max_margin_pp"] == 0.0


def test_per_family_qualifying_implies_oracle_under_and_over(artifact):
    for fam, rec in artifact["per_family"].items():
        if rec["qualifying"]:
            has_pair = any(
                rec["per_policy_regime"][f"{p}/under_wss"]["cells_won"] >= 1
                and rec["per_policy_regime"][f"{p}/over_wss"]["cells_won"] >= 1
                for p in ORACLE_AWARE
            )
            assert has_pair, f"{fam} marked qualifying but no oracle-aware policy has both regimes"


def test_per_family_replays_implies_qualifying(artifact):
    for fam, rec in artifact["per_family"].items():
        if rec["replays"]:
            assert rec["qualifying"]


def test_per_family_replays_iff_shrink_nonempty(artifact):
    for fam, rec in artifact["per_family"].items():
        assert rec["replays"] == (rec["qualifying"] and len(rec["shrink_evidence"]) >= 1)


# ---------------------------------------------------------------------------
# Group C — shrink_evidence semantics
# ---------------------------------------------------------------------------


def test_shrink_evidence_only_oracle_aware(artifact):
    for fam, rec in artifact["per_family"].items():
        for e in rec["shrink_evidence"]:
            assert e["policy"] in ORACLE_AWARE


def test_shrink_evidence_strict_inequality(artifact):
    for fam, rec in artifact["per_family"].items():
        for e in rec["shrink_evidence"]:
            assert e["under_median"] > e["over_median"]


def test_shrink_evidence_record_shape(artifact):
    for fam, rec in artifact["per_family"].items():
        for e in rec["shrink_evidence"]:
            assert set(e.keys()) == {"policy", "under_median", "over_median"}


def test_shrink_evidence_oracle_aware_order(artifact):
    """Entries appear in ORACLE_AWARE iteration order (GRASP then POPT)."""
    oa_index = {p: i for i, p in enumerate(ORACLE_AWARE)}
    for fam, rec in artifact["per_family"].items():
        indices = [oa_index[e["policy"]] for e in rec["shrink_evidence"]]
        assert indices == sorted(indices)


# ---------------------------------------------------------------------------
# Group D — classifier + margin math
# ---------------------------------------------------------------------------


def test_classify_under_wss_strict_lt():
    """ratio < 0.25 (strict)."""
    assert _classify(1, 5) == "under_wss"  # 0.2 < 0.25
    assert _classify(1, 4) == "near_wss"   # 0.25 == 0.25 → near (NOT under)


def test_classify_over_wss_strict_gt():
    """ratio > 4.0 (strict)."""
    assert _classify(5, 1) == "over_wss"  # 5 > 4
    assert _classify(4, 1) == "near_wss"  # 4 == 4 → near (NOT over)


def test_classify_near_wss_inclusive_band():
    assert _classify(1, 1) == "near_wss"
    assert _classify(1, 2) == "near_wss"


def test_l3_bytes_powers_of_two():
    assert L3_BYTES["4kB"] == 4 * 1024
    assert L3_BYTES["1MB"] == 1024 * 1024
    assert L3_BYTES["8MB"] == 8 * 1024 * 1024


def test_median_pair_average_for_even_n():
    assert _median([1.0, 2.0, 3.0, 4.0]) == 2.5
    assert _median([1.0]) == 1.0
    assert _median([]) == 0.0


def test_pct_bespoke_formula_at_p90():
    """_pct(xs, 0.9) returns s[max(0, min(n-1, int(round(0.9·(n-1)))))]."""
    xs = list(range(11))  # 0..10
    # round(0.9 * 10) = 9 → s[9] = 9
    assert _pct(xs, 0.9) == 9
    assert _pct([], 0.9) == 0.0


# ---------------------------------------------------------------------------
# Group E — full byte parity
# ---------------------------------------------------------------------------


def test_per_family_byte_parity(artifact, derived):
    assert artifact["per_family"] == derived["per_family"]


def test_full_artifact_byte_parity(artifact, derived):
    a = dict(artifact)
    a_meta = dict(a["meta"])
    a_meta.pop("verdict_invariant", None)
    a["meta"] = a_meta
    d = dict(derived)
    d_meta = dict(d["meta"])
    d_meta.pop("verdict_invariant", None)
    d["meta"] = d_meta
    assert a == d
