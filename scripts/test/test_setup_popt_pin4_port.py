from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/experiments/ecg/flows/setup_popt_pin4_port.py"


def test_setup_script_pins_sources_and_documents_only_compat_changes():
    text = SCRIPT.read_text()
    assert "POPT_COMMIT" in text and "GRASP_COMMIT" in text
    assert "PIN_ExitProcess(0)" in text
    assert "restore old Pin global C++ names" in text
    assert "official 3-bit insertion/hit rules" in text


def test_setup_script_records_built_binary_hashes():
    text = SCRIPT.read_text()
    assert "port_build_manifest.json" in text
    assert '"binaries": binaries' in text
