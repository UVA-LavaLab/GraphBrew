"""Gate 225 — LIT-Dev: known-deviation completeness audit.

Locks the invariants that ``KNOWN_DEVIATIONS`` in
``literature_baselines.py`` (the whitelist that downgrades live
lit-faith ``disagree`` rows to ``known_deviation``) stays grounded:

* every entry has a non-empty reason ≥ ``MIN_REASON_LENGTH`` chars
* every reason contains a quantitative magnitude phrase
* every reason contains at least one anchor (paper venue tag,
  ``§``/Fig/Sec, design term, or algorithmic root-cause vocabulary)
* the whitelist is a strict superset of the live faith corpus —
  zero ``status="known_deviation"`` rows without a matching key
* no orphan entries — every key is exercised by lit-faith
* coverage spans ≥ 2 policies, ≥ 4 graphs, ≥ 3 apps, ≥ 3 L3 sizes
* sentinel vocabulary (e.g. ``GRASP``, ``frontier``) shows up
  in expected minimum counts — drift-alarm

The generator is ``scripts/experiments/ecg/lit_faith_deviations.py``;
all artifacts live in ``wiki/data/lit_faith_deviations.{json,md,csv}``.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = REPO_ROOT / "wiki" / "data" / "lit_faith_deviations.json"

sys.path.insert(0, str(REPO_ROOT))


@pytest.fixture(scope="module")
def payload() -> dict:
    assert DATA_PATH.exists(), (
        "lit_faith_deviations.json missing — run "
        "`make lit-deviations` (or "
        "`python3 scripts/experiments/ecg/lit_faith_deviations.py`) first."
    )
    return json.loads(DATA_PATH.read_text())


# ---------- shape / coverage ----------

def test_schema_version(payload):
    assert payload["schema_version"] == 1


def test_summary_keys(payload):
    needed = {
        "known_deviations_total",
        "faith_known_deviation_rows",
        "orphan_known_deviations",
        "inactive_known_deviations",
        "faith_kd_without_entry_count",
        "well_formed_reasons",
        "short_reasons",
        "missing_quantitative_phrase",
        "missing_anchor",
        "policies_covered",
        "graphs_covered",
        "apps_covered",
        "l3_sizes_covered",
        "min_reason_length",
    }
    assert needed.issubset(payload["summary"])


def test_min_reason_length_floor(payload):
    """Generator's minimum-length threshold is fixed (no silent drift)."""
    assert payload["summary"]["min_reason_length"] >= 80


def test_known_deviations_count_floor(payload):
    """At least 20 KD entries to remain meaningful as a corpus."""
    assert payload["summary"]["known_deviations_total"] >= 20


def test_live_faith_known_deviation_rows_floor(payload):
    """Live faith corpus has at least 15 known_deviation rows."""
    assert payload["summary"]["faith_known_deviation_rows"] >= 15


# ---------- core invariants ----------

def test_no_orphan_known_deviations(payload):
    """Whitelist entry must correspond to a real (graph,app,l3,policy)
    that lit-faith actually exercises. Orphans accumulate silently when
    the corpus shrinks; this prevents that."""
    orphans = [r for r in payload["reason_audits"] if r["is_orphan"]]
    assert orphans == [], (
        f"orphan KD entries (no live lit-faith cell): "
        f"{[(r['graph'], r['app'], r['l3_size'], r['policy']) for r in orphans]}"
    )


def test_no_live_kd_without_whitelist(payload):
    """Every live status=known_deviation row must have a KNOWN_DEVIATIONS
    entry. If this fires, the comparator is silently downgrading
    disagrees that nobody documented."""
    assert payload["faith_kd_without_entry"] == []
    assert payload["summary"]["faith_kd_without_entry_count"] == 0


def test_all_reasons_meet_length_floor(payload):
    """Every reason must be ≥ MIN_REASON_LENGTH chars. Short reasons
    are almost always placeholders."""
    short = [
        r for r in payload["reason_audits"]
        if r["length"] < payload["summary"]["min_reason_length"]
    ]
    assert short == [], (
        f"reasons shorter than {payload['summary']['min_reason_length']} "
        f"chars: {[(r['graph'], r['app'], r['l3_size'], r['length']) for r in short]}"
    )
    assert payload["summary"]["short_reasons"] == 0


def test_all_reasons_have_quantitative_phrase(payload):
    """Every reason must quote at least one magnitude (pp, MB, %, MPKI,
    times, …) so reviewers can see the size of the documented gap."""
    missing = [
        r for r in payload["reason_audits"]
        if not r["has_quantitative_phrase"]
    ]
    assert missing == [], (
        f"reasons missing a quantitative phrase: "
        f"{[(r['graph'], r['app'], r['l3_size'], r['policy']) for r in missing]}"
    )
    assert payload["summary"]["missing_quantitative_phrase"] == 0


def test_all_reasons_have_anchor(payload):
    """Every reason must mention an anchor — paper section, design
    term, or algorithmic root-cause vocabulary (PR-rank, frontier,
    union-find, mismatch, …)."""
    missing = [r for r in payload["reason_audits"] if not r["has_anchor"]]
    assert missing == [], (
        f"reasons missing an anchor: "
        f"{[(r['graph'], r['app'], r['l3_size'], r['policy']) for r in missing]}"
    )
    assert payload["summary"]["missing_anchor"] == 0


def test_all_reasons_well_formed(payload):
    """Composite gate: every KD entry passes length + quantitative +
    anchor + not-orphan. Treat the count as a hard 100 % floor."""
    s = payload["summary"]
    assert s["well_formed_reasons"] == s["known_deviations_total"], (
        f"well_formed_reasons={s['well_formed_reasons']} vs "
        f"total={s['known_deviations_total']}"
    )


# ---------- coverage diversity ----------

def test_policy_coverage_floor(payload):
    """KD coverage spans ≥ 2 policies (today: POPT_GE_GRASP, NEAR)."""
    assert payload["summary"]["policies_covered"] >= 2


def test_expected_policies_present(payload):
    """The two known POPT-extended policies must be represented."""
    pols = set(payload["policy_coverage"])
    assert {"POPT_GE_GRASP", "POPT_NEAR_GRASP_IF_BIG_GAP"}.issubset(pols)


def test_graph_coverage_floor(payload):
    """KDs span ≥ 4 graphs so failures don't concentrate on one dataset."""
    assert payload["summary"]["graphs_covered"] >= 4


def test_app_coverage_floor(payload):
    """KDs span ≥ 3 apps (cc, bc, sssp/bfs at minimum)."""
    assert payload["summary"]["apps_covered"] >= 3


def test_l3_coverage_floor(payload):
    """KDs span ≥ 3 L3 sizes (1MB, 4MB, 8MB)."""
    assert payload["summary"]["l3_sizes_covered"] >= 3


def test_cc_app_dominates_kd(payload):
    """CC + BC together account for ≥ 60 % of KDs; these are the two
    apps whose algorithmic mismatch with POPT's PR-ranking is most
    documented."""
    total = payload["summary"]["known_deviations_total"]
    cc = payload["app_coverage"].get("cc", 0)
    bc = payload["app_coverage"].get("bc", 0)
    assert (cc + bc) / total >= 0.60, (
        f"CC+BC = {cc + bc}/{total} = {(cc+bc)/total:.2%}"
    )


# ---------- vocabulary fingerprint ----------

def test_vocab_grasp_count_floor(payload):
    """`GRASP` appears in at least 10 KD reasons."""
    assert payload["vocab_counts"].get("GRASP", 0) >= 10


def test_vocab_frontier_count_floor(payload):
    """`frontier` appears in ≥ 5 KD reasons (BC/CC algorithmic anchor)."""
    assert payload["vocab_counts"].get("frontier", 0) >= 5


def test_vocab_phase1_present(payload):
    """`Phase 1` (P-OPT's well-known design choice) appears at least once."""
    assert payload["vocab_counts"].get("Phase 1", 0) >= 1


# ---------- structural / sanity ----------

def test_per_entry_lengths_match_reason(payload):
    """`length` field equals len(reason) — guards against unicode bugs."""
    for r in payload["reason_audits"]:
        assert r["length"] == len(r["reason"])


def test_kd_keys_are_tuples_of_four(payload):
    """Each KD entry is (graph, app, l3_size, policy) — 4 strings."""
    for r in payload["reason_audits"]:
        assert isinstance(r["graph"], str) and r["graph"]
        assert isinstance(r["app"], str) and r["app"]
        assert isinstance(r["l3_size"], str) and re.match(
            r"^\d+MB$", r["l3_size"]
        ), r["l3_size"]
        assert isinstance(r["policy"], str) and r["policy"]


def test_inactive_count_within_bounds(payload):
    """Inactive KDs (cell present but currently agrees) are tolerated
    but should not exceed half the corpus — otherwise the whitelist
    is over-prescribed."""
    total = payload["summary"]["known_deviations_total"]
    inactive = payload["summary"]["inactive_known_deviations"]
    assert inactive <= total // 2, (
        f"inactive={inactive} vs total={total}"
    )


def test_per_entry_dedup(payload):
    """KD keys are unique across reason_audits (no double-entries)."""
    keys = [
        (r["graph"], r["app"], r["l3_size"], r["policy"])
        for r in payload["reason_audits"]
    ]
    assert len(keys) == len(set(keys)), "duplicate KD keys"
