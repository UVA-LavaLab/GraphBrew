import copy
import json
import subprocess
from pathlib import Path

from scripts.experiments.ecg.analysis.claim_gate import (
    DEFAULT_LEDGER,
    markdown,
    validate,
)


ROOT = Path(__file__).resolve().parents[2]


def ledger():
    return json.loads(DEFAULT_LEDGER.read_text())


def test_current_claim_gate_is_consistent():
    data = ledger()
    assert validate(data) == []
    text = markdown(data)
    assert "K2-M generally speeds up full-graph workloads." in text
    assert "**prohibited**" in text
    assert "`matched_sniper_post_binding`" in text
    assert "K2 requires no live P-OPT rereference matrix" in text
    assert "1.329x packed K2-I-like model" in text
    assert "Zero reserved data ways means zero K2 hardware overhead" in text


def test_allowed_claim_fails_when_dependency_is_pending():
    data = copy.deepcopy(ledger())
    gate = next(
        gate for gate in data["gates"]
        if gate["id"] == "matrix_free_runtime")
    gate["status"] = "pending"
    errors = validate(data)
    assert any(
        "matrix_free_k2: allowed with pending gates" in error
        for error in errors)


def test_claim_gate_cli():
    script = ROOT / "scripts/experiments/ecg/analysis/claim_gate.py"
    result = subprocess.run(
        ["python3", str(script)],
        check=True, capture_output=True, text=True)
    assert "### Contribution delta" in result.stdout
    assert "### Headline claim gate" in result.stdout


def test_passed_gate_requires_resolvable_evidence():
    data = copy.deepcopy(ledger())
    gate = next(
        gate for gate in data["gates"]
        if gate["id"] == "matrix_free_runtime")
    gate["evidence"] = ["deadbee"]
    errors = validate(data)
    assert any("commit is missing or not reachable" in error for error in errors)
