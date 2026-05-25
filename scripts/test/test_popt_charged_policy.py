#!/usr/bin/env python3
"""Regression tests for P-OPT charged-overhead policy expansion."""

from argparse import Namespace
import importlib.util
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROI_MATRIX_PATH = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "roi_matrix.py"
spec = importlib.util.spec_from_file_location("roi_matrix", ROI_MATRIX_PATH)
roi_matrix = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["roi_matrix"] = roi_matrix
spec.loader.exec_module(roi_matrix)


def test_popt_charged_reserves_matrix_columns_as_llc_ways():
    args = Namespace(
        options="-g 12 -k 16 -o 5 -n 1 -i 2",
        line_size="64",
        l3_ways="16",
        popt_property_bytes="4",
        popt_active_columns="2",
        popt_min_data_ways="1",
        popt_num_epochs="256",
    )

    spec = roi_matrix.parse_policy_spec("POPT_CHARGED")
    charge = roi_matrix.popt_charge_metadata(args, spec, "4kB")

    assert spec.policy == "POPT"
    assert spec.label == "POPT_CHARGED"
    assert spec.charge_popt_overhead
    assert charge["popt_estimated_vertices"] == 4096
    assert charge["popt_matrix_column_bytes"] == 256
    assert charge["popt_reserved_ways"] == 2
    assert charge["popt_reserved_bytes"] == 512
    assert charge["popt_effective_l3_ways"] == "14"
    assert charge["popt_effective_l3_size"] == "3584B"
    assert charge["popt_matrix_stream_cache_lines"] == 1024


def test_charged_ecg_policy_maps_to_underlying_ecg_mode():
    spec = roi_matrix.parse_policy_spec("ECG:DBG_PRIMARY_CHARGED")

    assert spec.label == "ECG_DBG_PRIMARY_CHARGED"
    assert spec.policy == "ECG"
    assert spec.ecg_mode == "DBG_PRIMARY"
    assert spec.charge_popt_overhead