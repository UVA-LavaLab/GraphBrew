#!/usr/bin/env python3
"""Gate 253 — HANDOFF gate-reference registry.

Locks the contract between ``wiki/HANDOFF-grasp-popt-validation.md``
and ``scripts/experiments/ecg/confidence_dashboard.py`` so the
narrative HANDOFF cannot silently lag behind the live gate suite.

Sources of truth:

* ``confidence_dashboard.PYTEST_SUITES`` — the live universe of
  pytest-tracked gates;
* ``wiki/HANDOFF-grasp-popt-validation.md`` — the user-visible
  narrative that pins the headline count, the per-gate paragraphs,
  and the refresh cadence.

7 rules:

  * H1 every ``gate N`` / ``gates N-M`` token in HANDOFF parses to one
    or more positive integers;
  * H2 every PYTEST_SUITES label that carries ``(gate N)`` must be
    mentioned somewhere in HANDOFF (no orphan dashboard labels);
  * H3 the HANDOFF headline ``**N gates, all GREEN, exit 0**`` must
    equal ``len(PYTEST_SUITES)``;
  * H4 the HANDOFF "Refresh complete at gate N" line must equal
    ``len(PYTEST_SUITES)``;
  * H5 the HANDOFF "Next refresh due at gate M" line must equal
    refresh-at + 5 (declared cadence);
  * H6 no duplicate ``(gate N)`` token in PYTEST_SUITES labels
    (each gate number labels at most one suite);
  * H7 ``max(labeled_dashboard_gates)`` must equal
    ``len(PYTEST_SUITES)`` (the newest labeled gate equals the
    live count, so a new gate cannot land in the dashboard
    without a label).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
HANDOFF_PATH = ROOT / "wiki" / "HANDOFF-grasp-popt-validation.md"
sys.path.insert(0, str(ROOT / "scripts" / "experiments" / "ecg"))

GATE_REF_RE = re.compile(r"\bgate[s]?\s+(\d+(?:-\d+)?)\b")
DASHBOARD_LABEL_RE = re.compile(r"\(gate\s+(\d+)\)")
HEADLINE_RE = re.compile(r"\*\*(\d+)\s+gates,\s+all\s+GREEN,\s+exit\s+0\*\*")
REFRESH_AT_RE = re.compile(r"Refresh\s+complete\s+at\s+gate\s+(\d+)")
REFRESH_DUE_RE = re.compile(r"Next\s+refresh\s+due\s+(?:at\s+gate\s+)?(\d+)")
REFRESH_CADENCE = 5


def _load_pytest_suites() -> dict[str, tuple[str, str]]:
    from confidence_dashboard import PYTEST_SUITES  # noqa: WPS433
    return PYTEST_SUITES


def _harvest_handoff_gate_refs(text: str) -> set[int]:
    refs: set[int] = set()
    for m in GATE_REF_RE.finditer(text):
        token = m.group(1)
        if "-" in token:
            lo, hi = (int(x) for x in token.split("-"))
            refs.update(range(lo, hi + 1))
        else:
            refs.add(int(token))
    return refs


def _harvest_dashboard_labeled_gates(suites: dict[str, tuple[str, str]]) -> dict[int, list[str]]:
    """Return {gate_number: [labels that mention it]}."""
    out: dict[int, list[str]] = {}
    for label in suites:
        for m in DASHBOARD_LABEL_RE.finditer(label):
            n = int(m.group(1))
            out.setdefault(n, []).append(label)
    return out


# --- rules -----------------------------------------------------------

def _rule_h1(text: str) -> list[dict[str, Any]]:
    # GATE_REF_RE only emits parseable tokens; truly malformed shapes
    # (e.g. "gate v3") will be missed but also won't match.  We
    # additionally flag empty captures defensively.
    out: list[dict[str, Any]] = []
    for m in GATE_REF_RE.finditer(text):
        token = m.group(1)
        try:
            if "-" in token:
                lo, hi = (int(x) for x in token.split("-"))
                if lo <= 0 or hi <= 0 or lo > hi:
                    out.append({
                        "rule": "H1", "token": m.group(0),
                        "msg": f"bad range {token!r}",
                    })
            else:
                if int(token) <= 0:
                    out.append({
                        "rule": "H1", "token": m.group(0),
                        "msg": f"non-positive gate ref {token!r}",
                    })
        except ValueError:
            out.append({
                "rule": "H1", "token": m.group(0),
                "msg": f"unparseable gate ref {token!r}",
            })
    return out


def _rule_h2(handoff_refs: set[int],
             labeled: dict[int, list[str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n in sorted(labeled):
        if n not in handoff_refs:
            out.append({
                "rule": "H2", "gate": n,
                "msg": f"dashboard labels (gate {n}) but HANDOFF never references it",
                "labels": labeled[n],
            })
    return out


def _rule_h3(text: str, n_suites: int) -> list[dict[str, Any]]:
    m = HEADLINE_RE.search(text)
    if not m:
        return [{
            "rule": "H3",
            "msg": "HANDOFF headline `**N gates, all GREEN, exit 0**` not found",
        }]
    n = int(m.group(1))
    if n != n_suites:
        return [{
            "rule": "H3",
            "headline_n": n,
            "actual_n": n_suites,
            "msg": f"headline says {n} gates but PYTEST_SUITES has {n_suites}",
        }]
    return []


def _rule_h4(text: str, n_suites: int) -> list[dict[str, Any]]:
    m = REFRESH_AT_RE.search(text)
    if not m:
        return [{
            "rule": "H4",
            "msg": "HANDOFF `Refresh complete at gate N` line not found",
        }]
    n = int(m.group(1))
    if n != n_suites:
        return [{
            "rule": "H4",
            "refresh_at": n,
            "actual_n": n_suites,
            "msg": f"refresh-at says {n} but PYTEST_SUITES has {n_suites}",
        }]
    return []


def _rule_h5(text: str) -> list[dict[str, Any]]:
    m_at = REFRESH_AT_RE.search(text)
    m_due = REFRESH_DUE_RE.search(text)
    if not m_at or not m_due:
        return [{
            "rule": "H5",
            "msg": "HANDOFF refresh-at / refresh-due lines not both present",
        }]
    at, due = int(m_at.group(1)), int(m_due.group(1))
    if due != at + REFRESH_CADENCE:
        return [{
            "rule": "H5",
            "refresh_at": at,
            "refresh_due": due,
            "expected_due": at + REFRESH_CADENCE,
            "msg": f"refresh-due ({due}) != refresh-at ({at}) + cadence ({REFRESH_CADENCE})",
        }]
    return []


def _rule_h6(labeled: dict[int, list[str]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for n, labels in labeled.items():
        if len(labels) > 1:
            out.append({
                "rule": "H6", "gate": n,
                "labels": labels,
                "msg": f"gate {n} is labeled by {len(labels)} dashboard suites (must be exactly 1)",
            })
    return out


def _rule_h7(labeled: dict[int, list[str]], n_suites: int) -> list[dict[str, Any]]:
    if not labeled:
        return [{
            "rule": "H7",
            "msg": "no PYTEST_SUITES label contains a (gate N) tag — cannot anchor newest gate",
        }]
    m = max(labeled.keys())
    if m != n_suites:
        return [{
            "rule": "H7",
            "max_labeled_gate": m,
            "actual_n": n_suites,
            "msg": f"max labeled gate ({m}) != PYTEST_SUITES count ({n_suites})",
        }]
    return []


# --- driver ----------------------------------------------------------

def audit() -> dict[str, Any]:
    text = HANDOFF_PATH.read_text()
    suites = _load_pytest_suites()
    handoff_refs = _harvest_handoff_gate_refs(text)
    labeled = _harvest_dashboard_labeled_gates(suites)
    n_suites = len(suites)

    violations: list[dict[str, Any]] = []
    violations.extend(_rule_h1(text))
    violations.extend(_rule_h2(handoff_refs, labeled))
    violations.extend(_rule_h3(text, n_suites))
    violations.extend(_rule_h4(text, n_suites))
    violations.extend(_rule_h5(text))
    violations.extend(_rule_h6(labeled))
    violations.extend(_rule_h7(labeled, n_suites))

    return {
        "status":      "active",
        "rules": {
            "H1": "every gate-ref token parses to a positive integer (or positive range)",
            "H2": "every (gate N) dashboard label is mentioned in HANDOFF",
            "H3": "headline `**N gates, all GREEN, exit 0**` equals len(PYTEST_SUITES)",
            "H4": "`Refresh complete at gate N` equals len(PYTEST_SUITES)",
            "H5": "`Next refresh due` equals refresh-at + cadence (5)",
            "H6": "each (gate N) tag labels at most one dashboard suite",
            "H7": "max labeled gate == len(PYTEST_SUITES)",
        },
        "refresh_cadence": REFRESH_CADENCE,
        "totals": {
            "handoff_gate_refs": len(handoff_refs),
            "labeled_dashboard_gates": len(labeled),
            "pytest_suites": n_suites,
            "violations":    len(violations),
        },
        "handoff_gate_refs":         sorted(handoff_refs),
        "labeled_dashboard_gates":   sorted(labeled.keys()),
        "max_labeled_dashboard_gate": max(labeled.keys()) if labeled else None,
        "violations": violations,
    }


def write_outputs(data: dict[str, Any], json_out: Path | None,
                  md_out: Path | None, csv_out: Path | None) -> None:
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if md_out:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append("# HANDOFF gate-reference registry — gate 253")
        lines.append("")
        lines.append(f"Status: `{data['status']}`")
        lines.append("")
        t = data["totals"]
        lines.append(
            f"Totals: handoff_refs={t['handoff_gate_refs']}  "
            f"labeled_dashboard_gates={t['labeled_dashboard_gates']}  "
            f"pytest_suites={t['pytest_suites']}  "
            f"violations={t['violations']}")
        lines.append("")
        lines.append(f"Max labeled dashboard gate: `{data['max_labeled_dashboard_gate']}`")
        lines.append("")
        if data["violations"]:
            lines.append("## Violations")
            lines.append("")
            for v in data["violations"][:50]:
                lines.append(f"- {v}")
            lines.append("")
        md_out.write_text("\n".join(lines))
    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["metric", "value"])
            for k, v in data["totals"].items():
                w.writerow([k, v])
            w.writerow(["max_labeled_dashboard_gate",
                        data["max_labeled_dashboard_gate"]])
            w.writerow(["refresh_cadence", data["refresh_cadence"]])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--md-out",   type=Path, default=None)
    ap.add_argument("--csv-out",  type=Path, default=None)
    args = ap.parse_args()
    a = audit()
    write_outputs(a, args.json_out, args.md_out, args.csv_out)
    print(
        f"[lit-faith-handoff-xref] status={a['status']} "
        f"handoff_refs={a['totals']['handoff_gate_refs']} "
        f"labeled_gates={a['totals']['labeled_dashboard_gates']} "
        f"suites={a['totals']['pytest_suites']} "
        f"violations={a['totals']['violations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
