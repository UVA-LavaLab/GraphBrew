#!/usr/bin/env python3
"""Emit hashed K2 replacement and SECDED synthesis inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.experiments.ecg.analysis.k2_cacti_packet import (
    PROJECT_ROOT,
    sha256_file,
    write_json_atomic,
)


RTL_ROOT = PROJECT_ROOT / "bench/src_rtl"
VICTIM_RTL = RTL_ROOT / "k2_victim_select.sv"
ECC_RTL = RTL_ROOT / "k2_secded_49.sv"
TESTBENCH = RTL_ROOT / "tb_k2_physical_logic.sv"
POLICY_SSOT = PROJECT_ROOT / "bench/include/ecg_victim_policy.h"


def source_entry(path: Path) -> dict[str, str]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(path),
    }


def manifest() -> dict[str, Any]:
    return {
        "version": 1,
        "status": "inputs_only_unmeasured",
        "technology_nm_required": 32,
        "replacement_ranking_subcomponent": {
            "top": "k2_victim_select",
            "source": source_entry(VICTIM_RTL),
            "policy_ssot": source_entry(POLICY_SSOT),
            "parameters": {
                "WAYS": 16,
                "RRPV_BITS": 3,
                "RECENCY_BITS": 4,
                "TIER_BITS": 2,
                "DIST_BITS": 15,
            },
            "variants": {
                "GRASP_ONLY": 0,
                "EPOCH_FIRST": 1,
                "RRIP_FIRST": 2,
                "EPOCH_ONLY": 3,
                "SHORTCIRCUIT": 4,
                "DEGREE_FIRST": 5,
                "LRU_ONLY": 6,
            },
            "scope": (
                "Ranking and RRIP aging only. Final replacement synthesis must "
                "also include epoch-pair distance, context/property "
                "qualification, variant/online selection, and any non-baseline "
                "recency-rank maintenance."),
        },
        "ecc": {
            "area_top": "k2_secded_49_parallel16",
            "read_delay_top": "k2_secded_49_decode",
            "source": source_entry(ECC_RTL),
            "data_bits_per_way": 49,
            "secded_bits_per_way": 7,
            "ways": 16,
            "area_instances": {
                "encoders": 16,
                "decoders": 16,
            },
        },
        "verification": {
            "testbench": source_entry(TESTBENCH),
            "commands": [
                "python3 -m "
                "scripts.experiments.ecg.analysis.k2_rtl_verify",
            ],
        },
        "limitations": [
            "No technology area, power, or delay result is embedded.",
            "Request/CSR/queue/MSHR storage and merge logic require a separate "
            "registered synthesis top before the physical gate can pass.",
            "k2_victim_select is not sufficient by itself for the final "
            "k2_replacement_logic component.",
            "The 4-bit recency input is baseline-provided 16-way age rank, not "
            "additional K2 line metadata.",
        ],
    }


def emit(out_dir: Path) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = manifest()
    write_json_atomic(out_dir / "k2_rtl_manifest.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit hashed K2 synthesis input provenance.")
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    payload = emit(parse_args().out_dir)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
