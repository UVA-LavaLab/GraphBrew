import copy
import json
import subprocess
from pathlib import Path

from scripts.experiments.ecg.analysis.claim_gate import (
    DEFAULT_LEDGER,
    markdown,
    resolved_claims,
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
    assert "`full_graph_detailed_results`" in text
    assert "fresh computed-address K2-M gate passes all 15" in text
    assert "K2 requires no live P-OPT rereference matrix" in text
    assert "1.329x packed K2-I-like model" in text
    assert "Zero reserved data ways means zero K2 hardware overhead" in text
    assert "`semantic_work_infrastructure`" in text
    assert "policy-independent static graph-edge visits" in text
    gates = {gate["id"]: gate for gate in data["gates"]}
    assert gates["matched_sniper_post_binding"]["status"] == "passed"
    assert gates["hawkeye_real_pc_synthetic"]["status"] == "passed"
    assert gates["hawkeye_results"]["status"] == "pending"
    claims = {claim["id"]: claim for claim in resolved_claims(data)}
    speedup = claims["k2m_general_speedup"]
    assert speedup["decision"] == "prohibited"
    # A general speedup claim is blocked by the missing full-graph evidence and
    # by the two accounting defects the 2026-07-25 review found: an
    # oracle-classified prefetcher and an unpriced prefetch-traffic increase.
    # idealised_prefetch_free_evidence now passes: the oracle prefetcher is no
    # longer the default. The remaining blockers are the full-graph evidence and
    # the unpriced prefetch-traffic increase.
    assert speedup["missing_gates"] == [
        "full_graph_detailed_results",
        "prefetch_traffic_accounted",
    ]
    popt = claims["k2_beats_popt"]
    assert popt["decision"] == "prohibited"
    # K2-versus-P-OPT depends on the two metadata streams being priced
    # symmetrically. cache_sim now simulates P-OPT's column stream, but it does
    # so on the ordinary demand path rather than as a dedicated streaming
    # engine, so the gate is still open.
    assert "symmetric_metadata_accounting" in popt["required_gates"]
    assert "symmetric_metadata_accounting" in popt["missing_gates"]


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
