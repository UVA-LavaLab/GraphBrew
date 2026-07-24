#!/usr/bin/env python3
"""Validate and render the HPCA contribution/claim gate."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LEDGER = PROJECT_ROOT / "research/ecg-hpca/claim_gate.json"
GATE_STATUSES = {"passed", "pending", "blocked"}
CLAIM_DECISIONS = {"allowed", "prohibited"}
COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


def evidence_error(evidence: str) -> str | None:
    if COMMIT_RE.fullmatch(evidence):
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", evidence, "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True)
        return None if result.returncode == 0 else (
            f"commit is missing or not reachable: {evidence}")
    path = PROJECT_ROOT / evidence
    return None if path.exists() else f"evidence path does not exist: {evidence}"


def validate(ledger: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    gates = ledger.get("gates")
    claims = ledger.get("claims")
    contributions = ledger.get("contribution_delta")
    if not isinstance(gates, list):
        return ["gates must be a list"]
    if not isinstance(claims, list):
        return ["claims must be a list"]
    if not isinstance(contributions, list):
        return ["contribution_delta must be a list"]

    gate_map: dict[str, dict[str, Any]] = {}
    for gate in gates:
        if not isinstance(gate, dict) or not isinstance(gate.get("id"), str):
            errors.append("every gate must have a string id")
            continue
        gate_id = gate["id"]
        if gate_id in gate_map:
            errors.append(f"duplicate gate id: {gate_id}")
        gate_map[gate_id] = gate
        if gate.get("status") not in GATE_STATUSES:
            errors.append(f"{gate_id}: invalid gate status")
        evidence = gate.get("evidence")
        if not isinstance(evidence, list):
            errors.append(f"{gate_id}: evidence must be a list")
        elif gate.get("status") == "passed" and not evidence:
            errors.append(f"{gate_id}: passed gate requires evidence")
        elif gate.get("status") == "passed":
            for entry in evidence:
                if not isinstance(entry, str):
                    errors.append(f"{gate_id}: evidence entries must be strings")
                    continue
                problem = evidence_error(entry)
                if problem:
                    errors.append(f"{gate_id}: {problem}")

    claim_ids: set[str] = set()
    for claim in claims:
        if not isinstance(claim, dict) or not isinstance(claim.get("id"), str):
            errors.append("every claim must have a string id")
            continue
        claim_id = claim["id"]
        if claim_id in claim_ids:
            errors.append(f"duplicate claim id: {claim_id}")
        claim_ids.add(claim_id)
        decision = claim.get("decision")
        if decision not in CLAIM_DECISIONS:
            errors.append(f"{claim_id}: invalid decision")
        requirements = claim.get("required_gates")
        if not isinstance(requirements, list) or not requirements:
            errors.append(f"{claim_id}: required_gates must be non-empty")
            continue
        missing_ids = [
            gate_id for gate_id in requirements if gate_id not in gate_map]
        if missing_ids:
            errors.append(
                f"{claim_id}: unknown gates: {', '.join(missing_ids)}")
            continue
        pending = [
            gate_id for gate_id in requirements
            if gate_map[gate_id].get("status") != "passed"]
        if decision == "allowed" and pending:
            errors.append(
                f"{claim_id}: allowed with pending gates: "
                f"{', '.join(pending)}")

    for index, contribution in enumerate(contributions):
        if not isinstance(contribution, dict):
            errors.append(f"contribution_delta[{index}] must be an object")
            continue
        gate_id = contribution.get("gate")
        if gate_id not in gate_map:
            errors.append(
                f"contribution_delta[{index}]: unknown gate {gate_id!r}")
    return errors


def resolved_claims(ledger: dict[str, Any]) -> list[dict[str, Any]]:
    gate_map = {gate["id"]: gate for gate in ledger["gates"]}
    rows = []
    for claim in ledger["claims"]:
        missing = [
            gate_id for gate_id in claim["required_gates"]
            if gate_map[gate_id]["status"] != "passed"]
        rows.append({
            **claim,
            "missing_gates": missing,
            "gate_complete": not missing,
        })
    return rows


def markdown(ledger: dict[str, Any]) -> str:
    lines = [
        "### Contribution delta",
        "",
        "| Prior ECG | HPCA successor | Gate | Status |",
        "|---|---|---|---|",
    ]
    gate_map = {gate["id"]: gate for gate in ledger["gates"]}
    for row in ledger["contribution_delta"]:
        status = gate_map[row["gate"]]["status"]
        lines.append(
            f"| {row['prior']} | {row['successor']} | "
            f"`{row['gate']}` | {status} |")
    lines.extend([
        "",
        "### Headline claim gate",
        "",
        "| Claim | Scope | Decision | Missing gates |",
        "|---|---|---|---|",
    ])
    for claim in resolved_claims(ledger):
        missing = ", ".join(
            f"`{gate}`" for gate in claim["missing_gates"]) or "-"
        lines.append(
            f"| {claim['text']} | {claim['scope']} | "
            f"**{claim['decision']}** | {missing} |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and render HPCA claim gates.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ledger = json.loads(args.ledger.read_text())
    errors = validate(ledger)
    if errors:
        for error in errors:
            print(f"[FAIL] {error}")
        return 1
    if args.json:
        print(json.dumps(resolved_claims(ledger), indent=2, sort_keys=True))
    else:
        print(markdown(ledger))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
