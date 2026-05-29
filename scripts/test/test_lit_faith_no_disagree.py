"""Hard pytest gate: literature_faithfulness comparator must show zero disagreements.

The lit-faithfulness comparator categorizes every cell in the literature
sweep as one of: ``ok`` / ``within_tolerance`` / ``insufficient_data`` /
``known_deviation`` / ``missing`` / ``disagree``.

``disagree`` is the only verdict that signals a real regression vs the
published paper claims (every other verdict is either passing or
explicitly registered as a documented deviation). This test makes that
invariant a hard CI failure rather than a buried JSON field.

We do **not** regenerate the JSON here — it is treated as the persisted
artifact contract. The test is skipped if the artifact is missing so
fresh clones (without a sweep) don't fail; in CI the artifact is always
present and the test will run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
LIT_FAITH_JSON = REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json"


def _load_summary() -> dict:
    if not LIT_FAITH_JSON.exists():
        pytest.skip(f"{LIT_FAITH_JSON.relative_to(REPO_ROOT)} not on disk")
    payload = json.loads(LIT_FAITH_JSON.read_text())
    summary = payload.get("summary")
    if not isinstance(summary, dict):
        pytest.fail(
            f"{LIT_FAITH_JSON.relative_to(REPO_ROOT)} is missing the "
            f"top-level 'summary' object — comparator output schema changed?"
        )
    return summary


def test_lit_faith_summary_has_required_fields() -> None:
    summary = _load_summary()
    required = {
        "claims_total",
        "disagree",
        "insufficient_data",
        "known_deviation",
        "missing",
        "ok",
        "within_tolerance",
    }
    missing = required - set(summary)
    assert not missing, (
        f"literature_faithfulness summary is missing keys: "
        f"{sorted(missing)} (have: {sorted(summary)})"
    )


def test_lit_faith_has_zero_disagreements() -> None:
    """Hard gate: ``disagree == 0``.

    A disagreement means an unregistered, unexplained mismatch with the
    literature. The fix is either:

      * register the cell in ``KNOWN_DEVIATIONS`` (with a paper-anchored
        rationale per the citation-locator test), or
      * fix the underlying baseline so it matches the paper, or
      * widen the tolerance in the matching ``LiteratureClaim`` if the
        paper itself reports a band rather than a point estimate.

    Suppressing this test is the wrong answer.
    """

    summary = _load_summary()
    disagree = int(summary.get("disagree", 0))
    payload = json.loads(LIT_FAITH_JSON.read_text())
    cells = [
        d for d in payload.get("disagreements", [])
        if isinstance(d, dict)
    ]
    detail = ""
    if cells:
        lines = [
            f"  {d.get('graph','?')}/{d.get('app','?')}/L3={d.get('l3_size','?')}/"
            f"policy={d.get('policy','?')}: {d.get('reason','no reason recorded')}"
            for d in cells
        ]
        detail = "\n" + "\n".join(lines)
    assert disagree == 0, (
        f"literature_faithfulness comparator reports {disagree} "
        f"unexplained disagreement(s) — register them in KNOWN_DEVIATIONS "
        f"or fix the baseline:{detail}"
    )


def test_lit_faith_known_deviation_count_matches_payload() -> None:
    """Catch silent schema drift where the count and the rows disagree."""

    summary = _load_summary()
    payload = json.loads(LIT_FAITH_JSON.read_text())
    rows = payload.get("known_deviations", [])
    assert isinstance(rows, list), "known_deviations should be a list"
    assert int(summary.get("known_deviation", 0)) == len(rows), (
        f"summary.known_deviation={summary.get('known_deviation')} but "
        f"len(known_deviations)={len(rows)} — comparator emitter is "
        f"inconsistent."
    )
