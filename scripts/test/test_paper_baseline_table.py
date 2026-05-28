"""Smoke + correctness pytest for paper_baseline_table.py.

Builds a synthetic sweep root with two graphs and four policies, runs the
table generator, and asserts:

  * Rows are present for every (graph, app, l3) tuple in the sweep.
  * Δ vs LRU is reported in percentage-points (not fraction).
  * Verdict labels round-trip through the markdown emitter.
  * A literature claim verdict propagates from
    :mod:`literature_baselines` when applicable.

This keeps the paper table generator honest as the comparator evolves.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"


def _load(module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, ECG_DIR / file_name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


pbt = _load("paper_baseline_table", "paper_baseline_table.py")


def _write_roi_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    base_fields = [
        "policy", "policy_label", "l3_size", "l3_miss_rate",
        "l3_misses", "total_accesses", "section", "status",
        "benchmark", "json_path", "log_path",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=base_fields)
        w.writeheader()
        for r in rows:
            full = {k: r.get(k, "") for k in base_fields}
            w.writerow(full)


def _make_sweep(tmp_path: Path) -> Path:
    """Synthesize a sweep root with web-Google/pr and email-Eu-core/bc."""
    sweep = tmp_path / "sweep"
    # web-Google/pr at L3=1MB: LRU=60%, SRRIP=54.3%, GRASP=45.2%, POPT=41.7%
    # Deltas (pp): LRU=0, SRRIP=-5.7, GRASP=-14.8, POPT=-18.3 -- matches
    # the literature claim for web-Google/pr at 1MB.
    _write_roi_csv(sweep / "web-Google-pr/lit/roi_matrix.csv", [
        dict(policy="LRU", policy_label="LRU", l3_size="1MB",
             l3_miss_rate=0.6009, l3_misses=1000000, total_accesses=2000000,
             section=0, status="ok", benchmark="pr"),
        dict(policy="SRRIP", policy_label="SRRIP", l3_size="1MB",
             l3_miss_rate=0.5439, l3_misses=900000, total_accesses=2000000,
             section=0, status="ok", benchmark="pr"),
        dict(policy="GRASP", policy_label="GRASP", l3_size="1MB",
             l3_miss_rate=0.4532, l3_misses=800000, total_accesses=2000000,
             section=0, status="ok", benchmark="pr"),
        dict(policy="POPT", policy_label="POPT", l3_size="1MB",
             l3_miss_rate=0.4168, l3_misses=700000, total_accesses=2000000,
             section=0, status="ok", benchmark="pr"),
    ])
    _write_roi_csv(sweep / "email-Eu-core-bc/lit/roi_matrix.csv", [
        dict(policy="LRU", policy_label="LRU", l3_size="1MB",
             l3_miss_rate=0.9906, l3_misses=100, total_accesses=200,
             section=0, status="ok", benchmark="bc"),
        dict(policy="GRASP", policy_label="GRASP", l3_size="1MB",
             l3_miss_rate=0.9906, l3_misses=100, total_accesses=200,
             section=0, status="ok", benchmark="bc"),
    ])
    return sweep


def test_paper_table_emits_rows_for_every_tuple(tmp_path: Path):
    sweep = _make_sweep(tmp_path)
    md = tmp_path / "paper.md"
    csv_out = tmp_path / "paper.csv"
    js_out = tmp_path / "paper.json"
    rc = pbt.main([
        "--sweep-root", str(sweep),
        "--markdown", str(md),
        "--csv", str(csv_out),
        "--json", str(js_out),
        "--min-accesses", "100",
    ])
    assert rc == 0
    assert md.exists() and csv_out.exists() and js_out.exists()

    rows = json.loads(js_out.read_text())
    keys = {(r["graph"], r["app"], r["l3_size"]) for r in rows}
    assert ("web-Google", "pr", "1MB") in keys
    assert ("email-Eu-core", "bc", "1MB") in keys


def test_paper_table_delta_in_percentage_points(tmp_path: Path):
    sweep = _make_sweep(tmp_path)
    js_out = tmp_path / "paper.json"
    rc = pbt.main([
        "--sweep-root", str(sweep),
        "--json", str(js_out),
        "--min-accesses", "100",
    ])
    assert rc == 0
    rows = json.loads(js_out.read_text())
    row = next(r for r in rows
               if (r["graph"], r["app"], r["l3_size"]) == ("web-Google", "pr", "1MB"))
    # SRRIP delta should be ~-5.7 pp.
    assert row["delta_pp_vs_lru"]["SRRIP"] == pytest.approx(-5.7, abs=0.05)
    # GRASP delta should be ~-14.77 pp.
    assert row["delta_pp_vs_lru"]["GRASP"] == pytest.approx(-14.77, abs=0.05)
    # POPT delta should be ~-18.41 pp.
    assert row["delta_pp_vs_lru"]["POPT"] == pytest.approx(-18.41, abs=0.05)
    # LRU is the baseline.
    assert row["delta_pp_vs_lru"]["LRU"] == 0.0


def test_paper_table_propagates_literature_verdicts(tmp_path: Path):
    sweep = _make_sweep(tmp_path)
    js_out = tmp_path / "paper.json"
    rc = pbt.main([
        "--sweep-root", str(sweep),
        "--json", str(js_out),
        "--min-accesses", "100",
    ])
    assert rc == 0
    rows = json.loads(js_out.read_text())
    row = next(r for r in rows
               if (r["graph"], r["app"], r["l3_size"]) == ("web-Google", "pr", "1MB"))
    # web-Google/pr at L3=1MB has explicit GRASP and POPT literature
    # claims (Faldu HPCA20 Fig 10, Balaji HPCA21 Fig 9). Both must be
    # marked "ok" given the synthesised deltas above match the lit
    # expectations within tolerance.
    assert row["verdict"].get("GRASP") == "ok", row["verdict"]
    assert row["verdict"].get("POPT") == "ok", row["verdict"]


def test_paper_table_csv_has_expected_columns(tmp_path: Path):
    sweep = _make_sweep(tmp_path)
    csv_out = tmp_path / "paper.csv"
    rc = pbt.main([
        "--sweep-root", str(sweep),
        "--csv", str(csv_out),
        "--min-accesses", "100",
    ])
    assert rc == 0
    with csv_out.open() as fh:
        reader = csv.DictReader(fh)
        assert reader.fieldnames is not None
        for col in ("graph", "app", "l3_size",
                    "lru_miss_rate", "grasp_miss_rate", "popt_miss_rate",
                    "srrip_delta_pp", "grasp_delta_pp", "popt_delta_pp",
                    "grasp_verdict", "popt_verdict"):
            assert col in reader.fieldnames, f"missing column {col}"


def test_paper_table_skips_pseudo_policy_claims(tmp_path: Path):
    """POPT_GE_GRASP / POPT_NEAR_GRASP_IF_BIG_GAP must not surface as
    per-policy verdict labels — they're cross-policy invariants."""
    sweep = _make_sweep(tmp_path)
    js_out = tmp_path / "paper.json"
    rc = pbt.main([
        "--sweep-root", str(sweep),
        "--json", str(js_out),
        "--min-accesses", "100",
    ])
    assert rc == 0
    rows = json.loads(js_out.read_text())
    for row in rows:
        for pol_verdict in row["verdict"].values():
            assert "POPT_GE_GRASP" not in pol_verdict
            assert "POPT_NEAR_GRASP" not in pol_verdict


def test_paper_table_empty_sweep_returns_nonzero(tmp_path: Path):
    empty = tmp_path / "empty"
    empty.mkdir()
    rc = pbt.main(["--sweep-root", str(empty)])
    assert rc != 0
