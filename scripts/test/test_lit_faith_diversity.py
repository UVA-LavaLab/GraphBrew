"""LIT-Cov — literature-faithfulness diversity coverage gate.

Locks the corpus's coverage-breadth of the literature it claims to
reproduce. Without this gate, a future regen could silently drop a
paper or power-law family (e.g. Jaleel ISCA10) and the headline
"all lit-faith cells agree" would still be GREEN despite collapsing
the comparator's representativeness.

Floors are pinned to TODAY'S OBSERVED MINIMUMS to ensure regressions
are caught the moment they happen. They can be relaxed only when a
deliberate corpus shrink is reviewed.

Invariants (organised in 7 groups):

1. **Aggregate floors** — claims_total ≥ 270; non-zero per axis.
2. **Per-family floor** — every represented graph family has ≥ 40
   claims (power-law scoped; today's minimum is 50).
3. **Per-paper floor** — every paper cited has ≥ 50 claims (today's
   minimum: Jaleel ISCA10 at 75).
4. **Per-app floor** — every GAPBS app (bc, bfs, cc, pr, sssp) has
   ≥ 40 claims (today's minimum: 45).
5. **Per-L3 floor** — every L3 size sampled has ≥ 80 claims (today's
   minimum: 84 each for the 3 power-law L3 sizes).
6. **Triangulation** — ≥ 100 cells receive claims from ≥ 2 distinct
   papers (today: 174); zero sign-inconsistent triangulated cells
   (the literature itself must not self-disagree).
7. **Schema / consistency** — every per-claim status ∈ closed
   vocabulary; every expected_sign ∈ {-, +, ~}; counts in
   distribution dicts sum to claims_total.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "wiki" / "data" / "lit_faith_diversity.json"

KNOWN_STATUSES = {"ok", "within_tolerance", "disagree", "known_deviation",
                  "insufficient_data", "missing"}
KNOWN_SIGNS = {"-", "+", "~"}
GAPBS_APPS = {"bc", "bfs", "cc", "pr", "sssp"}

# Floors pinned to today's observed minimums.
CLAIMS_TOTAL_FLOOR = 270
FAMILY_FLOOR = 40
PAPER_FLOOR = 50
APP_FLOOR = 40
L3_FLOOR = 80
TRIANGULATION_FLOOR = 100


@pytest.fixture(scope="module")
def audit() -> dict:
    assert ARTIFACT.is_file(), (
        f"missing {ARTIFACT}; run "
        "`python3 -m scripts.experiments.ecg.lit_faith_diversity` first"
    )
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Group 1 — Aggregate floors
# ---------------------------------------------------------------------------


def test_artifact_exists():
    assert ARTIFACT.is_file()


def test_summary_present(audit):
    assert "summary" in audit
    s = audit["summary"]
    assert isinstance(s, dict)


def test_claims_total_floor(audit):
    n = audit["summary"]["claims_total"]
    assert n >= CLAIMS_TOTAL_FLOOR, (
        f"claims_total={n} below floor {CLAIMS_TOTAL_FLOOR} — corpus shrank"
    )


def test_nonzero_axes(audit):
    s = audit["summary"]
    for axis in ("n_families", "n_apps", "n_l3_sizes", "n_policies",
                 "n_graphs", "n_papers"):
        assert s[axis] > 0, f"{axis} is zero"


# ---------------------------------------------------------------------------
# Group 2 — Per-family floor
# ---------------------------------------------------------------------------


def test_family_floor(audit):
    low = [(f, n) for f, n in audit["by_family"].items() if n < FAMILY_FLOOR]
    assert not low, f"families below floor {FAMILY_FLOOR}: {low}"


def test_summary_min_family_consistent(audit):
    s_min = audit["summary"]["min_family_count"]
    real_min = min(audit["by_family"].values())
    assert s_min == real_min, (s_min, real_min)


# ---------------------------------------------------------------------------
# Group 3 — Per-paper floor
# ---------------------------------------------------------------------------


def test_paper_floor(audit):
    low = [(p, n) for p, n in audit["by_paper"].items() if n < PAPER_FLOOR]
    assert not low, f"papers below floor {PAPER_FLOOR}: {low}"


def test_summary_min_paper_consistent(audit):
    s_min = audit["summary"]["min_paper_count"]
    real_min = min(audit["by_paper"].values())
    assert s_min == real_min, (s_min, real_min)


def test_at_least_three_papers(audit):
    assert audit["summary"]["n_papers"] >= 3, audit["by_paper"]


# ---------------------------------------------------------------------------
# Group 4 — Per-app floor
# ---------------------------------------------------------------------------


def test_app_floor(audit):
    low = [(a, n) for a, n in audit["by_app"].items() if n < APP_FLOOR]
    assert not low, f"apps below floor {APP_FLOOR}: {low}"


def test_all_gapbs_apps_present(audit):
    present = set(audit["by_app"].keys())
    missing = GAPBS_APPS - present
    assert not missing, f"GAPBS apps missing from lit-faith corpus: {missing}"


def test_summary_min_app_consistent(audit):
    s_min = audit["summary"]["min_app_count"]
    real_min = min(audit["by_app"].values())
    assert s_min == real_min, (s_min, real_min)


# ---------------------------------------------------------------------------
# Group 5 — Per-L3 floor
# ---------------------------------------------------------------------------


def test_l3_floor(audit):
    low = [(l, n) for l, n in audit["by_l3_size"].items() if n < L3_FLOOR]
    assert not low, f"L3 sizes below floor {L3_FLOOR}: {low}"


def test_at_least_three_l3_sizes(audit):
    assert audit["summary"]["n_l3_sizes"] >= 3, audit["by_l3_size"]


def test_summary_min_l3_consistent(audit):
    s_min = audit["summary"]["min_l3_count"]
    real_min = min(audit["by_l3_size"].values())
    assert s_min == real_min, (s_min, real_min)


# ---------------------------------------------------------------------------
# Group 6 — Triangulation
# ---------------------------------------------------------------------------


def test_triangulation_floor(audit):
    n = audit["summary"]["n_triangulated_cells"]
    assert n >= TRIANGULATION_FLOOR, (
        f"only {n} cells receive ≥ 2 paper claims (floor {TRIANGULATION_FLOOR}); "
        "the corpus has lost paper-overlap coverage"
    )


def test_no_sign_inconsistent_triangulated(audit):
    bad = audit["sign_inconsistent_cells"]
    assert not bad, (
        f"{len(bad)} triangulated cell(s) carry inconsistent expected_sign "
        "(the cited papers themselves disagree on direction):\n"
        + json.dumps(bad[:5], indent=2)
    )


def test_triangulation_count_consistent(audit):
    s = audit["summary"]
    assert s["n_triangulated_cells"] == len(audit["triangulated_cells"])
    assert s["n_sign_inconsistent_cells"] == len(audit["sign_inconsistent_cells"])


# ---------------------------------------------------------------------------
# Group 7 — Schema / vocabulary
# ---------------------------------------------------------------------------


def test_status_vocabulary(audit):
    unknown = set(audit["by_status"].keys()) - KNOWN_STATUSES
    assert not unknown, f"unknown status values: {unknown}"


def test_expected_sign_vocabulary(audit):
    unknown = set(audit["by_expected_sign"].keys()) - KNOWN_SIGNS
    assert not unknown, f"unknown expected_sign values: {unknown}"


def test_axis_sums_match_claims_total(audit):
    total = audit["summary"]["claims_total"]
    for axis in ("by_family", "by_app", "by_l3_size", "by_policy",
                 "by_graph", "by_status", "by_expected_sign"):
        s = sum(audit[axis].values())
        assert s == total, f"{axis} sums to {s} ≠ claims_total {total}"


def test_triangulated_cells_all_n_papers_ge_two(audit):
    for t in audit["triangulated_cells"]:
        assert t["n_papers"] >= 2, t
        assert len(t["papers"]) == t["n_papers"], t


def test_floors_are_lower_bounds(audit):
    """The pinned floors must be ≤ today's observed minimums."""
    s = audit["summary"]
    assert FAMILY_FLOOR <= s["min_family_count"]
    assert PAPER_FLOOR <= s["min_paper_count"]
    assert APP_FLOOR <= s["min_app_count"]
    assert L3_FLOOR <= s["min_l3_count"]
    assert CLAIMS_TOTAL_FLOOR <= s["claims_total"]
    assert TRIANGULATION_FLOOR <= s["n_triangulated_cells"]
