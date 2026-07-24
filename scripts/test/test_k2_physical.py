import json
import subprocess
from pathlib import Path

import pytest

from scripts.experiments.ecg.analysis.k2_physical import characterize, template


ROOT = Path(__file__).resolve().parents[2]


def measured_input():
    data = template()
    data.update({
        "technology_nm": 7,
        "cache_bytes": 8 * 1024 * 1024,
        "baseline_ways": 16,
        "metadata_access_fraction": 1.0,
    })
    data["baseline_cache"] = {
        "area_mm2": 4.0,
        "read_energy_nj": 1.0,
        "write_energy_nj": 1.2,
        "leakage_mw": 20.0,
        "delay_ns": 2.0,
    }
    data["k2_metadata_sram"] = {
        "area_mm2": 0.4,
        "read_energy_nj": 0.1,
        "write_energy_nj": 0.12,
        "leakage_mw": 2.0,
        "delay_ns": 1.0,
    }
    data["k2_replacement_logic"] = {
        "area_mm2": 0.04,
        "read_energy_nj": 0.02,
        "write_energy_nj": 0.02,
        "leakage_mw": 0.2,
        "delay_ns": 0.1,
    }
    data["provenance"] = {
        "cacti_version": "test-cacti",
        "synthesis_tool": "test-synth",
        "technology_library": "test-lib",
        "baseline_config_sha256": "a" * 64,
        "metadata_config_sha256": "b" * 64,
        "logic_report_sha256": "c" * 64,
    }
    return data


def test_characterize_measured_physical_inputs():
    result = characterize(measured_input())
    assert result["k2_total_area_mm2"] == pytest.approx(4.44)
    assert result["k2_area_overhead_percent"] == pytest.approx(11.0)
    assert result["k2_read_energy_nj"] == pytest.approx(1.12)
    assert result["parallel_lookup_delay_ns"] == pytest.approx(2.1)
    assert result["serialized_lookup_delay_ns"] == pytest.approx(3.1)
    assert result["linear_equal_area_fractional_ways"] == pytest.approx(14.4)
    assert result["linear_equal_area_integral_ways"] == 14
    assert result["linear_equal_area_integral_effective_bytes"] == 7340032


def test_characterize_rejects_missing_or_placeholder_values():
    with pytest.raises(ValueError, match="technology_nm"):
        characterize(template())
    data = measured_input()
    data["k2_metadata_sram"]["area_mm2"] = None
    with pytest.raises(ValueError, match="k2_metadata_sram.area_mm2"):
        characterize(data)
    data = measured_input()
    data["baseline_ways"] = 16.5
    with pytest.raises(ValueError, match="baseline_ways must be an integer"):
        characterize(data)
    data = measured_input()
    data["provenance"]["cacti_version"] = None
    with pytest.raises(ValueError, match="provenance.cacti_version"):
        characterize(data)


def test_cli_template_and_input(tmp_path: Path):
    script = ROOT / "scripts/experiments/ecg/analysis/k2_physical.py"
    template_result = subprocess.run(
        ["python3", str(script), "--template"],
        check=True, capture_output=True, text=True)
    assert json.loads(template_result.stdout)["baseline_cache"]["area_mm2"] is None

    input_path = tmp_path / "physical.json"
    input_path.write_text(json.dumps(measured_input()))
    measured_result = subprocess.run(
        ["python3", str(script), "--input", str(input_path)],
        check=True, capture_output=True, text=True)
    assert json.loads(measured_result.stdout)[
        "linear_equal_area_integral_ways"] == 14
