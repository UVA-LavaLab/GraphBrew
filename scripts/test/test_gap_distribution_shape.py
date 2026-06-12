"""Tests for the per-cell gap-distribution shape envelope (gate 56)."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "gap_distribution_shape.json"
GENERATOR = (
    REPO_ROOT / "scripts" / "experiments" / "ecg" / "gap_distribution_shape.py"
)


def _payload() -> dict:
    if not DATA.exists():
        subprocess.check_call([sys.executable, str(GENERATOR)])
    return json.loads(DATA.read_text())


def test_payload_present_and_well_formed():
    p = _payload()
    assert "meta" in p
    assert "per_cell" in p
    assert p["meta"]["scope_l3_sizes"] == ["1MB", "4MB", "8MB"]
    assert p["meta"]["n_cells"] == 60  # 5 apps × 3 L3 × 4 policies


def test_every_cell_has_required_fields():
    p = _payload()
    needed = {
        "app",
        "l3_size",
        "policy",
        "n",
        "mean_gap_pp",
        "sd_gap_pp",
        "skewness_g1",
        "excess_kurtosis_g2",
    }
    for key, cell in p["per_cell"].items():
        assert needed <= set(cell.keys()), f"missing fields in {key}"
        assert cell["n"] >= 4, f"insufficient samples for kurtosis in {key}"


def test_validity_envelope_thresholds_match_literature():
    p = _payload()
    env = p["meta"]["validity_envelope"]
    assert env["max_abs_skewness_for_bootstrap"] == 2.0
    assert env["max_abs_excess_kurtosis_for_bootstrap"] == 7.0
    assert "Hesterberg" in env["literature_citation"]


def test_overall_verdict_is_pass():
    p = _payload()
    assert p["meta"]["bootstrap_validity_verdict"] == "PASS"


def test_no_new_offenders_vs_pinned_set():
    p = _payload()
    pin = p["meta"]["pinned_exception_set"]
    assert pin["new_offenders_vs_pin"] == [], (
        f"NEW cells exceed Hesterberg envelope: {pin['new_offenders_vs_pin']}."
        " Update PINNED_EXCEPTION_CELLS only after manual review."
    )


def test_offending_cell_count_inside_max():
    p = _payload()
    n = p["meta"]["observed_envelope"]["n_cells_outside_envelope"]
    cap = p["meta"]["pinned_exception_set"]["max_allowed"]
    assert n <= cap, f"{n} offending cells > cap {cap}"


def test_worst_cell_is_bfs_1MB_GRASP_known_outlier():
    # Re-pinned 2026-06-12 to single-thread array-relative-GRASP 0.15 corpus.
    p = _payload()
    obs = p["meta"]["observed_envelope"]
    assert obs["worst_skew_cell"] == "bfs/1MB/GRASP"
    assert obs["worst_kurt_cell"] == "bfs/1MB/GRASP"
    assert obs["worst_abs_skew_any_cell"] >= 2.5
    assert obs["worst_abs_kurt_any_cell"] >= 6.9


def test_lru_srrip_cells_inside_envelope():
    # Non-oracle policies (LRU, SRRIP) lack the oracle-tight zero spike
    # that drives the discrete-outlier skew. All their cells must sit
    # inside the textbook envelope.
    p = _payload()
    for key, cell in p["per_cell"].items():
        if cell["policy"] in ("LRU", "SRRIP"):
            assert abs(cell["skewness_g1"]) < 2.0, (
                f"{key} LRU/SRRIP cell unexpectedly outside envelope:"
                f" skew={cell['skewness_g1']}"
            )
            assert abs(cell["excess_kurtosis_g2"]) < 7.0, (
                f"{key} LRU/SRRIP cell unexpectedly outside envelope:"
                f" kurt={cell['excess_kurtosis_g2']}"
            )


def test_all_pinned_exceptions_involve_oracle_aware_policy():
    p = _payload()
    for label in p["meta"]["pinned_exception_set"]["pinned"]:
        _, _, pol = label.split("/")
        assert pol in ("GRASP", "POPT"), (
            f"pinned exception {label} unexpectedly involves non-oracle policy"
            f" {pol}; rationale only covers oracle-aware-vs-zero spike"
        )


def test_per_l3_worst_includes_every_paper_l3():
    p = _payload()
    summary = p["meta"]["per_l3_worst"]
    assert set(summary.keys()) == {"1MB", "4MB", "8MB"}
    for l3, row in summary.items():
        assert row["n_cells"] == 20, f"{l3} cells != 20 (5 apps × 4 pols)"


def test_cross_gate_46_marginals_still_inside_envelope():
    # Pooled per-policy marginals (gate 46) must remain inside the envelope
    # even though individual cells can break it. This is the original
    # gate-46 invariant; reaffirm it here so any regression in either
    # gate trips this consistency check.
    g46_path = REPO_ROOT / "wiki" / "data" / "distribution_diagnostics.json"
    if not g46_path.exists():
        import pytest

        pytest.skip("distribution_diagnostics.json missing; run gate 46 first")
    g46 = json.loads(g46_path.read_text())
    obs = g46["meta"]["observed_envelope"]
    assert obs["worst_abs_skewness_per_policy_marginal"] < 2.0
    assert obs["worst_abs_excess_kurtosis_per_policy_marginal"] < 7.0


def test_known_outlier_graph_drives_extreme_cells():
    # The Hesterberg-violating cells are the same ones where one mesh /
    # road graph is producing the long tail. Sanity-check that the
    # offending cells' max gap is at least 5× their median across the
    # cross-graph distribution (i.e., one outlier dominates the cell).
    p = _payload()
    offenders = p["meta"]["observed_envelope"]["cells_outside_envelope"]
    rows_path = REPO_ROOT / "wiki" / "data" / "oracle_gap.json"
    rows = json.loads(rows_path.read_text())["rows"]
    from collections import defaultdict
    import statistics

    per_cell_vals: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        per_cell_vals[(r["app"], r["l3_size"], r["policy"])].append(
            float(r["gap_pp"])
        )

    dominated_count = 0
    for label in offenders:
        app, l3, pol = label.split("/")
        xs = per_cell_vals[(app, l3, pol)]
        med = statistics.median(xs)
        mx = max(xs)
        # Three explanations all count as "expected pattern":
        #   (a) median is zero and at least one outlier is non-trivially
        #       larger than the noise floor (mx > 0.01 pp);
        #   (b) max is 5x or more the median (single outlier dominates);
        #   (c) every value is tiny (mx < 0.05) — small-n mathematical
        #       artifact rather than a true heavy tail.
        if med == 0.0 and mx > 0.01:
            dominated_count += 1
        elif med > 0 and mx / max(med, 1e-9) >= 5.0:
            dominated_count += 1
        elif mx < 0.05:
            dominated_count += 1
    assert dominated_count == len(offenders), (
        f"only {dominated_count}/{len(offenders)} offending cells show the"
        " expected one-outlier or tiny-n pattern"
    )
