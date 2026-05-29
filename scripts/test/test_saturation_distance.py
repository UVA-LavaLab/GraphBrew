"""Tests for gate 65 — per-app saturation distance at 4MB->8MB."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA = REPO_ROOT / "wiki" / "data" / "saturation_distance.json"
GENERATOR = (
    REPO_ROOT / "scripts" / "experiments" / "ecg" / "saturation_distance.py"
)


def _payload() -> dict:
    if not DATA.exists():
        subprocess.check_call([sys.executable, str(GENERATOR)])
    return json.loads(DATA.read_text())


def test_payload_well_formed():
    p = _payload()
    assert "meta" in p and "per_app" in p and "per_cell" in p
    assert p["meta"]["pico_graph"] == "email-Eu-core"


def test_all_five_apps_measured():
    p = _payload()
    assert p["meta"]["app_count"] == 5
    assert set(p["per_app"].keys()) == {"bc", "bfs", "cc", "pr", "sssp"}


def test_full_corpus_cell_count():
    # 5 apps * 6 graphs but two (app, graph) pairs are missing from
    # the corpus (cc/delaunay_n19 and sssp/delaunay_n19 etc); today
    # exactly 28 cells satisfy the 4MB+8MB requirement.
    p = _payload()
    assert p["meta"]["cell_count"] == 28


def test_no_non_negative_violations():
    p = _payload()
    assert p["meta"]["non_negative_violations"] == [], (
        "8MB cannot be worse than 4MB on best-policy miss rate for "
        "any graph with WSS > 4 MB"
    )


def test_pico_sentinel_saturated_everywhere():
    # email-Eu-core must be saturated for every app it's measured on.
    p = _payload()
    assert p["meta"]["pico_violations"] == [], (
        f"email-Eu-core must be saturated; got violations: "
        f"{p['meta']['pico_violations']}"
    )


def test_app_diversity_above_threshold():
    p = _payload()
    assert (
        p["meta"]["app_diversity_range_pp"]
        >= p["meta"]["app_diversity_threshold"]
    )


def test_bc_is_least_saturated_app():
    # BC streams edges and never reuses them; it has the largest
    # per-app median saturation distance in this corpus today.
    p = _payload()
    medians = {app: s["median_pp"] for app, s in p["per_app"].items()}
    assert max(medians, key=medians.get) == "bc", (
        f"bc must be the least-saturated app by median; got {medians}"
    )


def test_bfs_has_lowest_app_median():
    # BFS' frontier reuses pages within tightly knit graphs, so its
    # 4MB->8MB improvement is the smallest among apps today.
    p = _payload()
    medians = {app: s["median_pp"] for app, s in p["per_app"].items()}
    assert min(medians, key=medians.get) == "bfs", (
        f"bfs must be the most-saturated app by median; got {medians}"
    )


def test_at_least_three_apps_above_5pp_median():
    # Three or more apps must still have median improvement >= 5pp at
    # 8MB; this guards against the corpus collapsing into saturation
    # (which would make app-level signal disappear).
    p = _payload()
    above = sum(
        1 for s in p["per_app"].values() if s["median_pp"] >= 5.0
    )
    assert above >= 3, (
        f"at least 3 apps must still have median distance >= 5pp; "
        f"only {above} did"
    )


def test_all_cells_have_required_fields():
    p = _payload()
    for c in p["per_cell"]:
        for k in (
            "app",
            "graph",
            "wss_bytes",
            "best4_miss_pp",
            "best8_miss_pp",
            "distance_pp",
            "is_pico_sentinel",
        ):
            assert k in c, f"cell missing key {k}: {c}"


def test_pico_sentinel_flagged_correctly():
    p = _payload()
    sent = [c for c in p["per_cell"] if c["is_pico_sentinel"]]
    assert len(sent) >= 1, "no pico-sentinel cells flagged"
    assert all(c["graph"] == "email-Eu-core" for c in sent)


def test_verdict_is_pass():
    p = _payload()
    assert p["meta"]["verdict"] == "PASS"
