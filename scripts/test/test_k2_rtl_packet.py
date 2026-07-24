import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.experiments.ecg.analysis.k2_rtl_packet import emit  # noqa: E402


def test_rtl_packet_hashes_synthesis_inputs(tmp_path: Path):
    payload = emit(tmp_path)
    assert payload["status"] == "inputs_only_unmeasured"
    assert payload["technology_nm_required"] == 32
    replacement = payload["replacement_ranking_subcomponent"]
    assert replacement["parameters"]["WAYS"] == 16
    assert "Ranking and RRIP aging only" in replacement["scope"]
    assert payload["ecc"]["area_instances"] == {
        "encoders": 16,
        "decoders": 16,
    }
    for entry in (
            replacement["source"],
            replacement["policy_ssot"],
            payload["ecc"]["source"],
            payload["verification"]["testbench"]):
        path = ROOT / entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
    serialized = json.loads(
        (tmp_path / "k2_rtl_manifest.json").read_text())
    assert serialized == payload
