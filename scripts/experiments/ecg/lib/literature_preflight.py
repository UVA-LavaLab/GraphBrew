"""Shared snapshot-based literature-faithfulness pre-flight check.

Reads ``wiki/data/literature_faithfulness_postfix.json`` and decides whether
paper-relevant jobs may proceed. Both ``paper_pipeline.py`` and
``final_paper_run.py`` invoke :func:`snapshot_preflight` so they share the
same opt-out semantics and exit codes.

The check is deliberately snapshot-based: re-running the comparator can take
minutes and requires the literature sweep root on disk. Refresh the snapshot
with ``make lit-faith`` or ``make confidence`` before launching paper work.

Exit codes (callers should ``return`` these to the OS as-is):

* ``0`` — gate PASS.
* ``1`` — gate FAIL because there are unexplained disagreements.
* ``2`` — gate FAIL because the snapshot file is missing or unreadable.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import TextIO

DEFAULT_SNAPSHOT = (
    Path(__file__).resolve().parents[4]
    / "wiki"
    / "data"
    / "literature_faithfulness_postfix.json"
)


def snapshot_preflight(
    snapshot_path: Path | None = None,
    *,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> int:
    """Verify the lit-faith snapshot is GREEN.

    Returns the exit code to surface to the OS. Writes the human-readable
    PASS/FAIL message to ``out`` (success path) or ``err`` (failure path);
    both default to ``sys.stdout``/``sys.stderr`` so callers can capture or
    silence output for tests.
    """
    snapshot_path = snapshot_path or DEFAULT_SNAPSHOT
    out = out if out is not None else sys.stdout
    err = err if err is not None else sys.stderr

    if not snapshot_path.exists():
        print(
            f"[lit-gate] FAIL: {snapshot_path} not found. Run `make lit-faith` "
            f"(or `make confidence` for the full gate suite) first, then "
            f"retry. Use --skip-literature-gate to bypass at your own risk.",
            file=err,
        )
        return 2

    try:
        data = json.loads(snapshot_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(
            f"[lit-gate] FAIL: could not parse {snapshot_path}: {exc}",
            file=err,
        )
        return 2

    summary = data.get("summary", {})
    disagree = int(summary.get("disagree", 0))
    if disagree > 0:
        per = data.get("per_claim", [])
        examples = [
            f"{e.get('graph')}/{e.get('app')}/{e.get('l3_size')}/{e.get('policy')}"
            for e in per
            if e.get("status") == "disagree"
        ][:5]
        print(
            f"[lit-gate] FAIL: {disagree} unexplained literature "
            f"disagreement(s). Examples: {examples}. Either register "
            f"them in KNOWN_DEVIATIONS in literature_baselines.py with "
            f"a documented root cause, or fix the underlying behavior, "
            f"then re-run `make confidence`.",
            file=err,
        )
        return 1

    ok = int(summary.get("ok", 0))
    wt = int(summary.get("within_tolerance", 0))
    kd = int(summary.get("known_deviation", 0))
    total = int(summary.get("claims_total", ok + wt + kd))
    print(
        f"[lit-gate] PASS: lit-faith green "
        f"({ok} ok + {wt} within_tol + {kd} known_dev of {total} claims).",
        file=out,
    )
    return 0


__all__ = ["snapshot_preflight", "DEFAULT_SNAPSHOT"]
