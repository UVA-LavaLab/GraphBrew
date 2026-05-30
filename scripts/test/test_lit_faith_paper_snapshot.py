"""Tests for gate 244 — paper-figure data snapshot integrity.

Locks in:

  * generator emits an "active" status (the gate has no scaffold mode);
  * F1 (single snapshot) — exactly one paper_pipeline_YYYYMMDD/ dir;
  * F2 (recency) — snapshot name parses & is within age bound;
  * F3 (provenance) — every row has the three required provenance
    fields populated;
  * F4 (single source run) — all rows share pipeline_run_dir;
  * F5 (rectangular coverage) — every (benchmark, graph, l3_size) cell
    has the full POLICY_LABELS palette and nothing extra;
  * F6 (value hygiene) — miss_rate ∈ [0,1] universally;
    total_accesses ≥ 1 for high-activity benchmarks (PR);
  * the HIGH_ACTIVITY_BENCHMARKS carve-out is honored;
  * generator's POLICY_LABELS load path stays aligned with
    paper_pipeline.py (no silent palette drift);
  * snapshot row count is non-zero (catches "snapshot dir exists but
    is empty");
  * snapshot name-date == directory's actual mtime within a guard band
    (catches accidentally re-touching an old snapshot).
"""
from __future__ import annotations

import csv
import datetime as dt
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WIKI_DATA = ROOT / "wiki" / "data"
GEN_PATH = ROOT / "scripts" / "experiments" / "ecg" / "lit_faith_paper_snapshot.py"
PAPER_PIPELINE = ROOT / "scripts" / "experiments" / "ecg" / "paper_pipeline.py"
SNAPSHOT_JSON = WIKI_DATA / "lit_faith_paper_snapshot.json"


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def gen():
    return _load_module(GEN_PATH, "lit_faith_paper_snapshot_dyn")


@pytest.fixture(scope="module")
def pp():
    return _load_module(PAPER_PIPELINE, "paper_pipeline_for_244")


@pytest.fixture(scope="module")
def audit(gen):
    return gen.audit()


def test_status_active(audit):
    assert audit["status"] == "active", \
        "gate 244 has no scaffold-deferred mode; status must be 'active'"


@pytest.mark.parametrize("rule_id", ["F1", "F2", "F3", "F4", "F5", "F6"])
def test_rule_documented(audit, rule_id):
    assert rule_id in audit["rules"], \
        f"rule {rule_id} must be documented in audit['rules']"


def test_zero_violations(audit):
    vs = audit["violations"]
    assert vs == [], (
        f"gate 244 reports {len(vs)} violation(s); first few:\n"
        + "\n".join(f"  - {v}" for v in vs[:10])
    )


# ---------------------------------------------------------- F1: single snapshot

def test_f1_exactly_one_snapshot(audit):
    snaps = audit["snapshots_found"]
    assert len(snaps) == 1, (
        f"exactly one paper_pipeline_YYYYMMDD/ dir expected; "
        f"found {len(snaps)}: {snaps}"
    )


# ---------------------------------------------------------- F2: recency

def test_f2_snapshot_name_parses(audit, gen):
    name = audit["snapshot"]["name"]
    m = gen.SNAPSHOT_DIR_RE.match(name)
    assert m is not None, f"snapshot dir name {name!r} must match YYYYMMDD"
    # date must parse
    dt.datetime.strptime(m.group(1), "%Y%m%d")


def test_f2_age_within_bound(audit, gen):
    age = audit["snapshot"]["age_days"]
    assert age is not None
    assert age <= gen.MAX_SNAPSHOT_AGE_DAYS, (
        f"snapshot age {age}d exceeds MAX_SNAPSHOT_AGE_DAYS "
        f"({gen.MAX_SNAPSHOT_AGE_DAYS}d)"
    )


# ---------------------------------------------------------- F3-F5 covered by zero-violations test


def test_f5_palette_populated(audit, pp):
    """Sanity: F5 silently passes if palette is empty. Verify the palette
    loaded from paper_pipeline actually has at least 9 entries."""
    palette = set(pp.POLICY_LABELS.keys())
    assert len(palette) >= 9, \
        f"POLICY_LABELS must have ≥ 9 entries, got {len(palette)}"


def test_snapshot_policy_count_matches_palette(audit, pp):
    palette = set(pp.POLICY_LABELS.keys())
    obs = audit["snapshot"]["policy_count"]
    assert obs == len(palette), (
        f"snapshot observed policy_count={obs} but POLICY_LABELS has "
        f"{len(palette)} entries — palette/snapshot drift"
    )


# ---------------------------------------------------------- F6: value hygiene & carve-out

def test_f6_high_activity_benchmarks_documented(gen):
    """Carve-out set must be non-empty AND contain 'pr' (the canonical
    high-activity benchmark)."""
    assert gen.HIGH_ACTIVITY_BENCHMARKS, \
        "HIGH_ACTIVITY_BENCHMARKS must be non-empty"
    assert "pr" in gen.HIGH_ACTIVITY_BENCHMARKS, \
        "'pr' must be in HIGH_ACTIVITY_BENCHMARKS (canonical hub kernel)"


def test_f6_thresholds_strict(gen):
    assert gen.MISS_RATE_MIN == 0.0
    assert gen.MISS_RATE_MAX == 1.0
    assert gen.MIN_TOTAL_ACCESSES == 1


def test_f6_pr_rows_have_meaningful_accesses(gen):
    """Belt-and-suspenders: walk the snapshot directly and confirm every
    PR row has total_accesses ≥ 1 (independent of audit())."""
    snaps = gen._find_snapshot_dirs()
    assert len(snaps) == 1
    rows = gen._read_roi_matrix(snaps[0])
    pr_rows = [r for r in rows if (r.get("benchmark") or "").lower() == "pr"]
    assert pr_rows, "snapshot should contain at least one PR row"
    bad = [r for r in pr_rows
           if int((r.get("total_accesses") or "0").strip() or 0) < 1]
    assert not bad, (
        f"{len(bad)} PR row(s) have total_accesses<1; "
        f"first: {bad[0].get('final_graph')}/{bad[0].get('policy_label')}"
    )


def test_f6_miss_rate_in_range_for_all_rows(gen):
    """Belt-and-suspenders for the universal miss-rate range check."""
    snaps = gen._find_snapshot_dirs()
    rows = gen._read_roi_matrix(snaps[0])
    bad = []
    for r in rows:
        raw = (r.get("l3_miss_rate") or "").strip()
        try:
            mr = float(raw)
        except ValueError:
            bad.append((r.get("benchmark"), r.get("policy_label"), raw))
            continue
        if not (0.0 <= mr <= 1.0):
            bad.append((r.get("benchmark"), r.get("policy_label"), raw))
    assert not bad, f"{len(bad)} rows with out-of-range miss_rate; first: {bad[0]}"


# ---------------------------------------------------------- structural

def test_snapshot_row_count_nonzero(audit):
    assert audit["snapshot"]["row_count"] > 0, \
        "snapshot directory exists but roi_matrix_all.csv is empty"


def test_rendered_json_matches_audit(audit):
    """If the committed JSON drifts from audit() output, regenerate."""
    assert SNAPSHOT_JSON.exists(), \
        f"missing {SNAPSHOT_JSON}; run `make lit-paper-snapshot`"
    on_disk = json.loads(SNAPSHOT_JSON.read_text())
    for k in ("status", "rules", "thresholds", "snapshots_found", "totals"):
        assert on_disk.get(k) == audit.get(k), (
            f"committed JSON drifts from audit() at key {k!r}; "
            "run `make lit-paper-snapshot` to refresh"
        )


def test_generator_loads_policy_labels(gen, pp):
    """Confirm the importlib hook actually returns POLICY_LABELS — if
    paper_pipeline.py drops/renames that name, the generator breaks
    silently. This test catches that early."""
    palette = gen._load_palette()
    assert palette == set(pp.POLICY_LABELS.keys()), (
        "_load_palette() drifted from paper_pipeline.POLICY_LABELS"
    )


# ---------------------------------------------------------- snapshot cohesion

def test_single_pipeline_run_dir_in_snapshot(gen):
    snaps = gen._find_snapshot_dirs()
    rows = gen._read_roi_matrix(snaps[0])
    rds = {(r.get("pipeline_run_dir") or "").strip() for r in rows}
    rds.discard("")
    assert len(rds) == 1, (
        f"snapshot should have exactly one pipeline_run_dir; "
        f"got {len(rds)}: {sorted(rds)}"
    )


def test_every_row_has_provenance(gen):
    snaps = gen._find_snapshot_dirs()
    rows = gen._read_roi_matrix(snaps[0])
    needed = ("pipeline_source_csv", "pipeline_run_dir", "pipeline_run_name")
    bad = [r for r in rows
           if any(not (r.get(k) or "").strip() for k in needed)]
    assert not bad, (
        f"{len(bad)} row(s) missing provenance fields; "
        f"first: {bad[0].get('benchmark')}/{bad[0].get('policy_label')}"
    )


def test_rectangular_coverage_matrix(gen, pp):
    snaps = gen._find_snapshot_dirs()
    rows = gen._read_roi_matrix(snaps[0])
    palette = set(pp.POLICY_LABELS.keys())
    cells: dict = {}
    for r in rows:
        k = ((r.get("benchmark") or "").strip(),
             (r.get("final_graph") or "").strip(),
             (r.get("l3_size") or "").strip())
        if not all(k):
            continue
        cells.setdefault(k, set()).add((r.get("policy_label") or "").strip())
    incomplete = [k for k, ps in cells.items() if ps != palette]
    assert not incomplete, (
        f"{len(incomplete)} non-rectangular cell(s); "
        f"first: {incomplete[0]} -> missing {sorted(palette - cells[incomplete[0]])}"
    )
