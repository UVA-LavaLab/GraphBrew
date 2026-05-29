"""Sanity gate for the paper claims registry.

The registry is the **single source of truth** for every numerical
claim the paper makes. This gate pins:

* Every claim has the required scalar fields (id, category, text,
  source, gate).
* Every claim's ``source`` file exists on disk.
* Every claim's ``gate`` pytest module exists on disk (the gate that
  is supposed to enforce the invariant the claim depends on).
* Headline claims that we will literally quote in the paper text are
  present (road-family popt-vs-grasp, social-family popt-vs-grasp,
  literature deviations share, GRASP win share, total green gates).
* Claim ids are unique.

Run via ``pytest scripts/test/test_paper_claims_registry.py``.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "wiki" / "data"
JSON_PATH = DATA_DIR / "paper_claims.json"
MD_PATH = DATA_DIR / "paper_claims.md"

REQUIRED_KEYS = {"id", "category", "text", "source"}
REQUIRED_HEADLINE_IDS = (
    "corpus.graph_count",
    "reproduction.ok_ratio",
    "lit_faith.disagreement_rate",
    "winner.grasp_share",
    "popt_vs_grasp.road_family_mean",
    "popt_vs_grasp.social_family_mean",
    "deviations.popt_overhead_share",
    "confidence.green_gate_count",
)


def _ensure_registry() -> None:
    if JSON_PATH.exists() and MD_PATH.exists():
        return
    cmd = [
        sys.executable,
        "-m",
        "scripts.experiments.ecg.paper_claims_registry",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


@pytest.fixture(scope="module")
def registry() -> dict:
    _ensure_registry()
    return json.loads(JSON_PATH.read_text())


def test_registry_artifacts_exist() -> None:
    _ensure_registry()
    assert JSON_PATH.exists(), JSON_PATH
    assert MD_PATH.exists(), MD_PATH


def test_registry_has_claims(registry: dict) -> None:
    assert registry["n_claims"] >= 8, registry["n_claims"]
    assert isinstance(registry["claims"], list)
    assert len(registry["claims"]) == registry["n_claims"]


def test_required_keys_present(registry: dict) -> None:
    for c in registry["claims"]:
        missing = REQUIRED_KEYS - set(c.keys())
        assert not missing, (c.get("id"), missing)


def test_claim_ids_are_unique(registry: dict) -> None:
    ids = [c["id"] for c in registry["claims"]]
    dupes = [i for i in ids if ids.count(i) > 1]
    assert not dupes, sorted(set(dupes))


def test_source_files_exist(registry: dict) -> None:
    for c in registry["claims"]:
        src = REPO_ROOT / c["source"]
        assert src.exists(), c


def test_gate_files_exist(registry: dict) -> None:
    """If a claim names a gate, the gate file must exist."""
    for c in registry["claims"]:
        gate = c.get("gate")
        if not gate or gate == "—":
            continue
        path = REPO_ROOT / gate
        assert path.exists(), c


def test_headline_claims_present(registry: dict) -> None:
    ids = {c["id"] for c in registry["claims"]}
    missing = [hid for hid in REQUIRED_HEADLINE_IDS if hid not in ids]
    assert not missing, (
        "headline claims missing from registry: " + ", ".join(missing)
    )


def test_road_popt_claim_value_negative(registry: dict) -> None:
    """The road-family POPT-vs-GRASP mean must remain negative
    (POPT improves on GRASP). If this flips, the paper's central
    "POPT wins on road graphs" narrative is dead.
    """
    row = next(
        c for c in registry["claims"]
        if c["id"] == "popt_vs_grasp.road_family_mean"
    )
    assert isinstance(row["value"], (int, float)), row
    assert row["value"] < 0, row


def test_confidence_claim_reports_all_green(registry: dict) -> None:
    """The meta claim must report all gates pass."""
    row = next(
        c for c in registry["claims"] if c["id"] == "confidence.green_gate_count"
    )
    assert row["value"] >= 22, row
