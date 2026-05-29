"""Confidence gate 101 — literature deviations ↔ winning regime taxonomy
cross-artifact agreement.

Two artifacts capture overlapping but distinct slices of the corpus:

  - ``wiki/data/literature_deviations.json`` — the 30 cells where
    POPT >= GRASP or POPT ~ GRASP-with-big-gap, each tagged with the
    paper citation it deviates from, mechanism (currently 100 %
    popt_overhead_dominates), and per-{app, family, graph, policy,
    mechanism} summary cross-tabs.
  - ``wiki/data/winning_regime_taxonomy.json`` — the full 114-cell
    universe partitioned by (family, regime), with per-bin winner
    shares and a rules block listing fraction-1.0 family/regime
    winners (mesh-tiny→GRASP, mesh-small→POPT, mesh-large→POPT).

The paper cites both — deviations to acknowledge known POPT-favorable
cells in spite of the overall GRASP majority, taxonomy to ground the
"regime determines policy" framing. If they disagree, the paper text
either undercounts deviations relative to the actual corpus or quotes
a taxonomy regime that has no real cells behind it. This gate locks
13 invariants so they stay in sync as the corpus, deviation tagging,
or regime binning evolves.

Invariants (4 / 4 / 4 / 1):

  literature_deviations internal (4):
    1. summary.n_deviations == len(deviations)
    2. each by_X summary (by_app, by_family, by_graph, by_mechanism,
       by_policy) sums to n_deviations
    3. summary.mechanism_family_cross_tab equals the recomputed
       Counter of f"{mechanism}|{graph_family}" across deviations
    4. every deviation entry has exactly the 11 required keys (schema
       lock)

  winning_regime_taxonomy internal (4):
    5. summary.n_cells == len(cells)
    6. sum(by_family_regime[].total) == summary.n_cells
    7. summary.overall_winner_counts sums to n_cells AND for each
       policy in {GRASP, LRU, POPT, SRRIP}: sum of f"{policy}_wins"
       across by_family_regime == overall_winner_counts[policy]
    8. summary.n_family_regime_bins == len(by_family_regime)

  Cross-artifact (4):
    9. every deviation (graph, app, l3_size) tuple resolves to an
       existing cell in winning_regime_taxonomy.cells
   10. deviation graph_family set is a subset of taxonomy family set
       (currently dev={citation, social, web} ⊆ wrt={citation, mesh,
       road, social, web})
   11. per-family deviation count <= sum(by_family_regime[].total for
       that family) — the deviation universe cannot exceed the
       corpus universe per family
   12. for every rule in summary.rules, sample_size == count of
       cells matching (family, regime) AND wins == count of cells
       additionally matching winner

  Math (1):
   13. for every by_family_regime entry, the four explicit shares
       (GRASP/LRU/POPT/SRRIP) plus 1.0 - share_sum (OTHER share)
       round-trip back to total wins; share_sum is within 1e-5 of
       (1.0 - OTHER_wins/total)
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LD_PATH = PROJECT_ROOT / "wiki" / "data" / "literature_deviations.json"
WRT_PATH = PROJECT_ROOT / "wiki" / "data" / "winning_regime_taxonomy.json"

DEV_REQUIRED_KEYS = frozenset({
    "app", "citation", "delta_pct", "expected_sign", "graph",
    "graph_family", "l3_size", "mechanism", "policy",
    "popt_vs_grasp_pp", "tolerance_pct",
})
WRT_POLICIES = ("GRASP", "LRU", "POPT", "SRRIP")
SHARE_TOL = 1e-5


@pytest.fixture(scope="module")
def ld() -> dict:
    assert LD_PATH.exists(), f"missing literature_deviations.json at {LD_PATH}"
    return json.loads(LD_PATH.read_text())


@pytest.fixture(scope="module")
def wrt() -> dict:
    assert WRT_PATH.exists(), f"missing winning_regime_taxonomy.json at {WRT_PATH}"
    return json.loads(WRT_PATH.read_text())


# ---------------------------------------------------------------------------
# literature_deviations internal (4)
# ---------------------------------------------------------------------------


def test_ld_n_deviations_matches_list(ld: dict) -> None:
    declared = ld["summary"]["n_deviations"]
    actual = len(ld["deviations"])
    assert declared == actual, f"summary.n_deviations={declared} but len(deviations)={actual}"


def test_ld_by_x_sums_to_n_deviations(ld: dict) -> None:
    n = ld["summary"]["n_deviations"]
    bad: list[tuple[str, int]] = []
    for key, bucket in ld["summary"].items():
        if not key.startswith("by_") or not isinstance(bucket, dict):
            continue
        s = sum(bucket.values())
        if s != n:
            bad.append((key, s))
    assert not bad, f"summary.{key} sums != n_deviations={n}: {bad}"


def test_ld_mechanism_family_cross_tab_recomputable(ld: dict) -> None:
    actual = Counter(f"{d['mechanism']}|{d['graph_family']}" for d in ld["deviations"])
    declared = ld["summary"]["mechanism_family_cross_tab"]
    assert dict(actual) == declared, (
        f"mechanism_family_cross_tab mismatch: declared={declared}, recomputed={dict(actual)}"
    )


def test_ld_deviation_schema(ld: dict) -> None:
    bad: list[tuple[int, list[str], list[str]]] = []
    for i, d in enumerate(ld["deviations"]):
        keys = set(d.keys())
        if keys != DEV_REQUIRED_KEYS:
            bad.append((i, sorted(keys - DEV_REQUIRED_KEYS), sorted(DEV_REQUIRED_KEYS - keys)))
    assert not bad, f"deviation schema violations (idx, extra, missing): {bad}"


# ---------------------------------------------------------------------------
# winning_regime_taxonomy internal (4)
# ---------------------------------------------------------------------------


def test_wrt_n_cells_matches_list(wrt: dict) -> None:
    declared = wrt["summary"]["n_cells"]
    actual = len(wrt["cells"])
    assert declared == actual, f"summary.n_cells={declared} but len(cells)={actual}"


def test_wrt_family_regime_total_sums_to_n_cells(wrt: dict) -> None:
    s = sum(fr["total"] for fr in wrt["summary"]["by_family_regime"])
    n = wrt["summary"]["n_cells"]
    assert s == n, f"sum(by_family_regime[].total)={s} != n_cells={n}"


def test_wrt_overall_winner_counts_partition(wrt: dict) -> None:
    owc = wrt["summary"]["overall_winner_counts"]
    n = wrt["summary"]["n_cells"]
    s = sum(owc.values())
    assert s == n, f"sum(overall_winner_counts)={s} != n_cells={n} (counts={owc})"
    bad: list[tuple[str, int, int]] = []
    for policy in WRT_POLICIES:
        if policy not in owc:
            continue
        per_policy = sum(fr[f"{policy}_wins"] for fr in wrt["summary"]["by_family_regime"])
        if per_policy != owc[policy]:
            bad.append((policy, per_policy, owc[policy]))
    assert not bad, f"per-policy wins mismatch across by_family_regime: {bad}"


def test_wrt_n_family_regime_bins_matches(wrt: dict) -> None:
    declared = wrt["summary"]["n_family_regime_bins"]
    actual = len(wrt["summary"]["by_family_regime"])
    assert declared == actual, f"n_family_regime_bins={declared} != len(by_family_regime)={actual}"


# ---------------------------------------------------------------------------
# Cross-artifact (4)
# ---------------------------------------------------------------------------


def test_every_deviation_in_wrt_cells(ld: dict, wrt: dict) -> None:
    wrt_keys = {(c["graph"], c["app"], c["l3_size"]) for c in wrt["cells"]}
    missing = [
        (d["graph"], d["app"], d["l3_size"])
        for d in ld["deviations"]
        if (d["graph"], d["app"], d["l3_size"]) not in wrt_keys
    ]
    assert not missing, f"deviations missing from WRT cell universe: {missing}"


def test_deviation_families_subset_of_wrt(ld: dict, wrt: dict) -> None:
    dev = {d["graph_family"] for d in ld["deviations"]}
    wrt_fams = {c["family"] for c in wrt["cells"]}
    assert dev <= wrt_fams, (
        f"deviation families={dev} not subset of WRT families={wrt_fams}"
    )


def test_deviation_count_per_family_bounded_by_wrt_total(ld: dict, wrt: dict) -> None:
    dev_per_family = Counter(d["graph_family"] for d in ld["deviations"])
    wrt_per_family: Counter = Counter()
    for fr in wrt["summary"]["by_family_regime"]:
        wrt_per_family[fr["family"]] += fr["total"]
    bad: list[tuple[str, int, int]] = []
    for fam, c in dev_per_family.items():
        if c > wrt_per_family[fam]:
            bad.append((fam, c, wrt_per_family[fam]))
    assert not bad, f"per-family deviation count exceeds WRT total: {bad}"


def test_wrt_rules_recomputable_from_cells(wrt: dict) -> None:
    bad: list[tuple[str, str, str, int, int, int, int]] = []
    for rule in wrt["summary"]["rules"]:
        actual_sample = sum(
            1 for c in wrt["cells"]
            if c["family"] == rule["family"] and c["regime"] == rule["regime"]
        )
        actual_wins = sum(
            1 for c in wrt["cells"]
            if c["family"] == rule["family"]
            and c["regime"] == rule["regime"]
            and c["winner"] == rule["winner"]
        )
        if actual_sample != rule["sample_size"] or actual_wins != rule["wins"]:
            bad.append(
                (rule["family"], rule["regime"], rule["winner"],
                 actual_sample, rule["sample_size"], actual_wins, rule["wins"])
            )
    assert not bad, f"WRT rule sample_size/wins do not match cells (family, regime, winner, ...): {bad}"


# ---------------------------------------------------------------------------
# Math (1)
# ---------------------------------------------------------------------------


def test_wrt_family_regime_shares_sum_to_one(wrt: dict) -> None:
    bad: list[tuple[str, str, float, int]] = []
    for fr in wrt["summary"]["by_family_regime"]:
        share_sum = sum(fr[f"{p}_share"] for p in WRT_POLICIES)
        # OTHER share == 1 - share_sum; should equal OTHER_wins / total
        if fr["total"] > 0:
            expected_other = fr["OTHER_wins"] / fr["total"]
            implied_other = 1.0 - share_sum
            if not math.isclose(implied_other, expected_other, abs_tol=SHARE_TOL):
                bad.append((fr["family"], fr["regime"], share_sum, fr["OTHER_wins"]))
            # Each per-policy share == wins / total
            for p in WRT_POLICIES:
                expected = fr[f"{p}_wins"] / fr["total"]
                if not math.isclose(fr[f"{p}_share"], expected, abs_tol=SHARE_TOL):
                    bad.append((fr["family"], fr["regime"],
                                fr[f"{p}_share"], fr[f"{p}_wins"]))
    assert not bad, f"by_family_regime share/wins mismatches: {bad}"
