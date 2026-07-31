#!/usr/bin/env python3
"""Freeze the proposal K2-M correctness gate into hash-bound evidence."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from gem5_guest_receipt import stable_receipt_fingerprint  # noqa: E402


ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUT = (
    ROOT / "research" / "ecg-hpca" / "evidence" /
    "proposal_k2m_o3_20260730")
TARGET_POLICIES = {
    "ECG_K2_LRU_STREAMSHIELD",
    "ECG_K2_STREAMSHIELD",
}
INADMISSIBLE_EXACT = {
    "ipc", "cpi", "gem5_k2_accepts_per_traced_request",
}
INADMISSIBLE_EXCLUSIONS = {
    "l1d_size", "l1i_size", "l1_l2_policy",
    "l2_size", "l3_size", "l3_ways",
    "l3_effective_size", "l3_effective_ways",
    "dram_peak_bw_mibs",
}
REQUEST_RE = re.compile(
    r"\[ECG-K2-REQUEST sim=gem5 seq=(\d+) request_seq=(\d+) "
    r"dest=(\d+) tier=(\d+) epoch1=(\d+) epoch2=(\d+) "
    r"current=(\d+) context=(\d+)\]")
ACCEPT_RE = re.compile(
    r"\[ECG-K2-ACCEPT sim=gem5 seq=(\d+) request_seq=(\d+) "
    r"request_dest=(\d+) fill_dest=(\d+) source=(\w+) "
    r"tier=(\d+) epoch1=(\d+) epoch2=(\d+) current=(\d+) "
    r"context=(\d+) (?:property_elem_bytes|width)=(\d+)\]")
BYPASS_RE = re.compile(
    r"\[ECG-STREAM-BYPASS sim=gem5 [^\n]*size=(\d+) "
    r"source=([a-z-]+) [^\n]*allocate=0\]")
PR_RE = re.compile(
    r"\[ECG-PR-RESULT iterations=(\d+) semantic_edges=(\d+) "
    r"score_checksum=([0-9a-fA-F]+)\]")
VARIANT_RE = re.compile(
    r"\[ECG-VARIANT-RECEIPT sim=gem5 requested=([^ ]+) "
    r"effective=(\d+) dueling=(\d+)\]")


def descriptor(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    rows = None
    if path.suffix == ".csv":
        with path.open(newline="") as handle:
            rows = max(sum(1 for _ in csv.reader(handle)) - 1, 0)
    elif path.suffix == ".json":
        try:
            payload = json.loads(data)
            if isinstance(payload, list):
                rows = len(payload)
        except json.JSONDecodeError:
            rows = None
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "size": len(data),
        "rows": rows,
    }


def path_fingerprint(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.is_file():
        return descriptor(path)["sha256"]
    digest = hashlib.sha256()
    for child in sorted(
            item for item in path.rglob("*")
            if item.is_file() and
            "__pycache__" not in item.parts and
            item.suffix not in {".pyc", ".log"}):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(path_fingerprint(child).encode())
    return digest.hexdigest()


def git_capture(*args: str) -> bytes:
    try:
        result = subprocess.run(
            ["git", *args], cwd=ROOT,
            capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(
            f"cannot capture git state for proposal evidence: {error}") from error
    return result.stdout


def git_state() -> tuple[dict[str, Any], bytes, bytes]:
    status = git_capture("status", "--porcelain=v1")
    diff = git_capture("diff", "--binary", "HEAD")
    commit = git_capture("rev-parse", "HEAD").decode().strip()
    return ({
        "commit": commit,
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "status_lines": len(status.splitlines()),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "diff_bytes": len(diff),
    }, status, diff)


def paper_run_git_state_fingerprint() -> str:
    digest = hashlib.sha256()
    for args in (
            ("rev-parse", "HEAD"),
            ("diff", "--binary", "--no-ext-diff"),
            ("diff", "--cached", "--binary", "--no-ext-diff")):
        digest.update(git_capture(*args))
    return digest.hexdigest()


def descriptor_matches(path: Path, expected: dict[str, Any]) -> bool:
    if any(key not in expected for key in ("sha256", "size", "rows")):
        return False
    actual = descriptor(path)
    return all(
        actual.get(key) == expected.get(key)
        for key in ("sha256", "size", "rows"))


def is_inadmissible_column(name: str) -> bool:
    if name in INADMISSIBLE_EXCLUSIONS:
        return False
    return (
        name in INADMISSIBLE_EXACT or
        name.startswith(("sim_", "roi_", "l1_", "l2_", "l3_", "dram_")) or
        name.startswith("popt_charged_")
    )


def derive_inadmissible_columns(paths: list[Path]) -> list[str]:
    columns: set[str] = set()
    for path in paths:
        with path.open(newline="") as handle:
            columns.update(next(csv.reader(handle)))
    return sorted(name for name in columns if is_inadmissible_column(name))


def validate_probe_claims(
        payload: dict[str, Any], log_dir: Path | None = None) -> None:
    if not payload.get("overall_pass"):
        raise SystemExit("real-decoder proposal probe did not pass")
    required_probe_checks = (
        "atomic_all_modes",
        "atomic_compact_stream_to_k2m",
        "atomic_wrong_width_teeth",
        "atomic_proposal_wrong_format_teeth",
        "o3_exact_request_binding",
        "o3_request_flag_bypass",
    )
    if not all(
            payload.get("checks", {}).get(check) is True
            for check in required_probe_checks):
        raise SystemExit("decoder probe receipt is missing required checks")
    expected_probe = payload.get("expected", {})
    if (
            expected_probe.get("canonical") != "0x3a004700000025" or
            expected_probe.get("payload") != [37, 3, 17, 29, 11, 7] or
            expected_probe.get("property_value_bits") != "0x41234567" or
            expected_probe.get("record_request_bytes") != 4):
        raise SystemExit("decoder probe expected payload is not canonical")
    o3_run = payload.get("runs", {}).get("o3", {})
    if o3_run.get("exit_code") != 0 or o3_run.get("timed_out") is not False:
        raise SystemExit("decoder O3 probe did not complete successfully")
    atomic_runs = payload.get("runs", {}).get("atomic", [])
    if (
            {run.get("label") for run in atomic_runs} !=
            {
                "atomic_normal", "atomic_teeth",
                "atomic_proposal_format_teeth",
            } or
            any(
                run.get("exit_code") != 0 or
                run.get("timed_out") is not False
                for run in atomic_runs)):
        raise SystemExit("decoder Atomic probes did not exit cleanly")
    if log_dir is None:
        return
    normal_text = (log_dir / "atomic_normal.log").read_text(errors="ignore")
    teeth_text = (log_dir / "atomic_teeth.log").read_text(errors="ignore")
    proposal_teeth_text = (
        log_dir / "atomic_proposal_format_teeth.log").read_text(
            errors="ignore")
    o3_text = (log_dir / "o3_proposal.log").read_text(errors="ignore")
    if not (
            "[test_ecg_load_modes] RESULT: PASS" in normal_text and
            re.search(r"K2-C-SS-MLOAD[^\n]*\[OK\]", normal_text) and
            "canonical=0x3a004700000025" in normal_text):
        raise SystemExit("atomic normal proposal log did not pass")
    if "[test_ecg_load_modes] RESULT: FAIL" not in teeth_text:
        raise SystemExit("legacy decoder teeth did not fail")
    proposal_teeth = re.search(
        r"K2-C-SS-MLOAD[^\n]*canonical=(0x[0-9a-fA-F]+)[^\n]*\[FAIL\]",
        proposal_teeth_text)
    if (
            proposal_teeth is None or
            proposal_teeth.group(1).lower() == "0x3a004700000025" or
            "[test_ecg_load_modes] RESULT: FAIL" not in proposal_teeth_text):
        raise SystemExit("proposal record-format teeth did not fail")
    if "[test_ecg_load_modes] RESULT: PASS" not in o3_text:
        raise SystemExit("O3 proposal guest did not pass")
    requests: dict[int, tuple[int, ...]] = {}
    for match in REQUEST_RE.finditer(o3_text):
        groups = tuple(map(int, match.groups()))
        requests[groups[1]] = groups[2:]
    matching_accept = False
    for match in ACCEPT_RE.finditer(o3_text):
        groups = match.groups()
        request_sequence = int(groups[1])
        request_dest = int(groups[2])
        source = groups[4]
        accept_payload = tuple(map(int, groups[5:10]))
        elem_bytes = int(groups[10])
        request_payload = requests.get(request_sequence)
        if (
                source == "request" and elem_bytes == 4 and
                request_payload is not None and
                request_dest == request_payload[0] and
                accept_payload == request_payload[1:]):
            matching_accept = True
            break
    if not requests or not matching_accept:
        raise SystemExit("O3 proposal request/accept evidence is missing")
    if not any(
            match.group(1) == "4" and match.group(2) == "request-flag"
            for match in BYPASS_RE.finditer(o3_text)):
        raise SystemExit("O3 proposal size-4 request-flag bypass is missing")


def validate_raw_policy(
        policy: str, row: dict[str, str], raw: dict[str, Any],
        cell_env: dict[str, str]) -> None:
    pr_receipt = (
        row["pr_iterations"], row["pr_semantic_edges"],
        row["pr_score_checksum"])
    if raw["pr_receipt"] != pr_receipt:
        raise SystemExit(f"raw PR receipt mismatch for {policy}")
    if policy in TARGET_POLICIES:
        comparisons = {
            "request_trace_events": "gem5_k2_request_trace_events",
            "request_receipts": "gem5_k2_request_receipts",
            "request_trace_max_seq": "gem5_k2_request_trace_max_seq",
            "duplicate_request_receipts":
                "gem5_k2_duplicate_request_receipts",
            "request_conflicts": "gem5_k2_request_conflicts",
            "accepts": "gem5_k2_request_accepts",
            "duplicate_accepts": "gem5_k2_duplicate_accepts",
            "bad_accepts": "gem5_k2_request_bad_receipts",
            "exact_accepts": "gem5_k2_exact_vertex_accepts",
            "coalesced_accepts": "gem5_k2_coalesced_line_accepts",
            "mailbox_accepts": "gem5_k2_mailbox_accepts",
            "accept_record_epoch_pairs":
                "gem5_k2_accept_record_epoch_pairs",
            "request_record_epoch_pairs":
                "gem5_k2_request_record_epoch_pairs",
            "request_epoch_states": "gem5_k2_request_epoch_states",
            "accept_epoch_states": "gem5_k2_accept_epoch_states",
            "nonzero_epoch_accepts":
                "gem5_k2_nonzero_epoch_accepts",
            "payload_discriminating":
                "gem5_k2_payload_discriminating",
            "all_bypasses": "gem5_stream_bypass_all_events",
            "request_flag_bypasses":
                "gem5_stream_bypass_request_flag_events",
            "size4_bypasses":
                "gem5_stream_bypass_request_flag_size4_events",
            "range_bypasses": "gem5_stream_bypass_range_events",
        }
        for raw_key, row_key in comparisons.items():
            if raw[raw_key] != int(row.get(row_key) or 0):
                raise SystemExit(
                    f"raw receipt mismatch for {policy}: {raw_key}")
        if int(row.get("gem5_k2_request_line_bytes") or 0) != 64:
            raise SystemExit(f"unexpected request line size for {policy}")
        if abs(
                raw["accepts_per_traced_request"] -
                float(row.get(
                    "gem5_k2_accepts_per_traced_request") or 0.0)
                ) > 1e-12:
            raise SystemExit(
                f"raw receipt mismatch for {policy}: "
                "accepts_per_traced_request")
        if not raw["proposal_active"]:
            raise SystemExit(f"proposal activation missing for {policy}")
        required_env = {
            "GEM5_ECG_COMPACT_K2M_SS": "1",
            "GEM5_ECG_PRODUCER": "1",
            "GEM5_ECG_STREAM_REQUEST_BOUND": "1",
            "ECG_K2_DELIVERY_TRACE": "2048",
            "ECG_STREAM_BYPASS_TRACE": "2048",
            "ECG_STREAM_BYPASS": "1",
            "ECG_RECORD_VARIABLE_WIDTH": "1",
            "ECG_EXPECT_BYTES_PER_EDGE": "4",
            "ECG_VARIANT": row["gem5_variant_requested_receipt"],
        }
        for key, expected in required_env.items():
            if cell_env.get(key) != expected:
                raise SystemExit(
                    f"cell environment mismatch for {policy}: {key}")
    elif raw["proposal_active"]:
        raise SystemExit("semantic anchor activated the proposal path")
    elif cell_env.get("GEM5_ECG_COMPACT_K2M_SS") == "1":
        raise SystemExit("semantic anchor requested the proposal path")
    expected_variant = (
        row.get("gem5_variant_requested_receipt"),
        row.get("gem5_variant_effective_receipt"),
        row.get("gem5_variant_dueling_receipt"),
    )
    if raw["variant_receipt"] != expected_variant:
        raise SystemExit(f"raw variant receipt mismatch for {policy}")


def parse_raw_policy_receipts(
        log_path: Path, stderr_path: Path) -> dict[str, Any]:
    log_text = log_path.read_text(errors="ignore")
    stderr_text = stderr_path.read_text(errors="ignore")
    requests: dict[int, tuple[int, ...]] = {}
    raw_requests = 0
    max_trace_seq = -1
    request_conflicts = 0
    for match in REQUEST_RE.finditer(log_text):
        values = tuple(map(int, match.groups()))
        raw_requests += 1
        max_trace_seq = max(max_trace_seq, values[0])
        request_sequence = values[1]
        payload = values[2:]
        previous = requests.setdefault(request_sequence, payload)
        request_conflicts += previous != payload

    accepts = 0
    duplicate_accepts = 0
    accepted_sequences: set[int] = set()
    accepted_record_epoch_pairs: set[tuple[int, int]] = set()
    accepted_epoch_states: set[tuple[int, int, int]] = set()
    bad_accepts = 0
    exact_accepts = 0
    coalesced_accepts = 0
    mailbox_accepts = 0
    nonzero_epoch_accepts = 0
    for match in ACCEPT_RE.finditer(log_text):
        groups = match.groups()
        request_sequence = int(groups[1])
        request_dest = int(groups[2])
        fill_dest = int(groups[3])
        source = groups[4]
        payload = tuple(map(int, groups[5:10]))
        elem_bytes = int(groups[10])
        expected = requests.get(request_sequence)
        same_line = (
            elem_bytes > 0 and
            (request_dest * elem_bytes) // 64 ==
            (fill_dest * elem_bytes) // 64)
        valid = (
            expected is not None and source == "request" and
            request_dest == expected[0] and payload == expected[1:] and
            elem_bytes == 4 and same_line)
        duplicate_accepts += request_sequence in accepted_sequences
        accepted_sequences.add(request_sequence)
        accepts += 1
        bad_accepts += not valid
        mailbox_accepts += source == "mailbox"
        if valid and request_dest == fill_dest:
            exact_accepts += 1
        elif valid:
            coalesced_accepts += 1
        if valid:
            accepted_record_epoch_pairs.add((payload[1], payload[2]))
            accepted_epoch_states.add((payload[1], payload[2], payload[3]))
            nonzero_epoch_accepts += payload[1] != 0 or payload[2] != 0

    bypasses = list(BYPASS_RE.finditer(log_text))
    all_bypasses = len(bypasses)
    request_flag_bypasses = sum(
        match.group(2) == "request-flag" for match in bypasses)
    size4_bypasses = sum(
        match.group(1) == "4" and match.group(2) == "request-flag"
        for match in bypasses)
    range_bypasses = sum(
        match.group(2) == "range" for match in bypasses)
    pr_match = PR_RE.search(stderr_text)
    variant_match = VARIANT_RE.search(log_text)
    request_epoch_states = {
        (payload[2], payload[3], payload[4])
        for payload in requests.values()
    }
    request_record_epoch_pairs = {
        (payload[2], payload[3])
        for payload in requests.values()
    }
    payload_discriminating = (
        len(request_epoch_states) > 1 and
        len(accepted_epoch_states) > 1 and
        len(request_record_epoch_pairs) > 1 and
        len(accepted_record_epoch_pairs) > 1 and
        any(first != 0 or second != 0
            for first, second in accepted_record_epoch_pairs)
    )
    return {
        "request_trace_events": raw_requests,
        "request_receipts": len(requests),
        "request_trace_max_seq": max_trace_seq,
        "duplicate_request_receipts": raw_requests - len(requests),
        "request_conflicts": request_conflicts,
        "accepts": accepts,
        "duplicate_accepts": duplicate_accepts,
        "bad_accepts": bad_accepts,
        "exact_accepts": exact_accepts,
        "coalesced_accepts": coalesced_accepts,
        "mailbox_accepts": mailbox_accepts,
        "accept_record_epoch_pairs": len(accepted_record_epoch_pairs),
        "request_record_epoch_pairs": len(request_record_epoch_pairs),
        "request_epoch_states": len(request_epoch_states),
        "accept_epoch_states": len(accepted_epoch_states),
        "nonzero_epoch_accepts": nonzero_epoch_accepts,
        "payload_discriminating": int(payload_discriminating),
        "accepts_per_traced_request": (
            accepts / raw_requests if raw_requests else 0.0),
        "all_bypasses": all_bypasses,
        "request_flag_bypasses": request_flag_bypasses,
        "size4_bypasses": size4_bypasses,
        "range_bypasses": range_bypasses,
        "proposal_active": "[ECG_K2_MLOAD_C_SS]" in stderr_text,
        "variant_receipt": (
            variant_match.groups() if variant_match else None),
        "pr_receipt": (
            tuple(pr_match.groups()) if pr_match else None),
    }


def validate_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 3 or any(row.get("status") != "ok" for row in rows):
        raise SystemExit("proposal evidence requires exactly three ok rows")
    semantic_receipts = {
        (
            row.get("pr_iterations"),
            row.get("pr_semantic_edges"),
            row.get("pr_score_checksum"),
        )
        for row in rows
    }
    if (
            len(semantic_receipts) != 1 or
            any(not value for value in next(iter(semantic_receipts)))):
        raise SystemExit("proposal rows do not share one complete PR receipt")
    by_policy = {row.get("policy_label"): row for row in rows}
    if set(by_policy) != {"ECG_K2", *TARGET_POLICIES}:
        raise SystemExit(f"unexpected proposal policies: {sorted(by_policy)}")
    for policy in TARGET_POLICIES:
        row = by_policy[policy]
        required = {
            "proposal_path_active": "1",
            "gem5_k2_exact_request_bound": "1",
            "gem5_k2_payload_discriminating": "1",
            "gem5_k2_duplicate_accepts": "0",
            "gem5_k2_request_bad_receipts": "0",
            "gem5_k2_request_conflicts": "0",
            "gem5_k2_duplicate_request_receipts": "0",
            "gem5_k2_mailbox_accepts": "0",
            "gem5_k2_delivery_trace_limit": "2048",
            "gem5_k2_request_trace_events": "2048",
            "gem5_k2_request_receipts": "2048",
            "gem5_k2_request_trace_max_seq": "2047",
            "gem5_k2_delivery_trace_saturated": "1",
            "gem5_stream_bypass_range_events": "0",
            "gem5_stream_bypass_trace_saturated": "0",
            "gem5_stream_bypass_request_flag_bad_size_events": "0",
            "ecg_record_bytes": "4",
            "gem5_cpu_type": "O3",
            "timing_valid_for_speedup": "0",
        }
        for key, expected in required.items():
            if row.get(key) != expected:
                raise SystemExit(
                    f"{policy} failed {key}: {row.get(key)!r} != {expected!r}")
        if int(row.get("gem5_k2_accept_record_epoch_pairs") or 0) < 2:
            raise SystemExit(f"{policy} lacks record-epoch discrimination")
        if int(row.get("gem5_k2_nonzero_epoch_accepts") or 0) < 8:
            raise SystemExit(f"{policy} lacks enough nonzero epoch accepts")
        if int(row.get("gem5_k2_coalesced_line_accepts") or 0) < 1:
            raise SystemExit(f"{policy} lacks live same-line coalescing")
        size4_events = int(
            row.get("gem5_stream_bypass_request_flag_size4_events") or 0)
        if size4_events <= 0:
            raise SystemExit(f"{policy} lacks size-4 StreamShield evidence")
        if (
                row.get("gem5_stream_bypass_all_events") !=
                row.get("gem5_stream_bypass_request_flag_events")):
            raise SystemExit(f"{policy} has non-request-flag bypass events")
        if (
                int(row.get("gem5_stream_bypass_all_events") or 0) !=
                size4_events):
            raise SystemExit(f"{policy} bypass census is not entirely size 4")
    anchor = by_policy["ECG_K2"]
    if (
            anchor.get("ecg_record_bytes") != "8" or
            anchor.get("timing_model") != "mechanism_semantic_anchor" or
            anchor.get("gem5_cpu_type") != "O3" or
            anchor.get("gem5_compact_k2m_streamshield_active") != "0" or
            anchor.get("proposal_path_active") != "0" or
            anchor.get("timing_valid_for_speedup") != "0"):
        raise SystemExit("semantic anchor is not the declared 8-byte control")
    return rows


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"missing evidence input: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def verify_bundle(bundle: Path) -> None:
    manifest_path = bundle / "manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"invalid proposal evidence manifest: {error}") from error
    if manifest.get("schema") != "graphbrew-proposal-k2m-o3-evidence-v1":
        raise SystemExit("unexpected proposal evidence schema")
    actual_files = {
        str(path.relative_to(bundle))
        for path in bundle.rglob("*")
        if path.is_file() and path.name != "manifest.json"
    }
    if actual_files != set(manifest.get("files", {})):
        raise SystemExit("proposal evidence file roster mismatch")
    for relative, expected in manifest.get("files", {}).items():
        if not descriptor_matches(bundle / relative, expected):
            raise SystemExit(f"proposal evidence file mismatch: {relative}")
    generator = manifest.get("generators", {}).get("freeze_script", {})
    if not descriptor_matches(
            bundle / "source" / "freeze_proposal_k2m.py", generator):
        raise SystemExit("proposal evidence generator mismatch")
    source_diff = bundle / "source" / "source.diff.gz"
    try:
        uncompressed_diff = gzip.decompress(source_diff.read_bytes())
    except (OSError, EOFError) as error:
        raise SystemExit(f"invalid compressed source diff: {error}") from error
    if (
            hashlib.sha256(uncompressed_diff).hexdigest() !=
            manifest.get("git_state", {}).get("diff_sha256")):
        raise SystemExit("proposal evidence source diff mismatch")
    status_path = bundle / "source" / "status.porcelain"
    status_data = status_path.read_bytes()
    if (
            hashlib.sha256(status_data).hexdigest() !=
            manifest.get("git_state", {}).get("status_sha256") or
            len(status_data.splitlines()) !=
            manifest.get("git_state", {}).get("status_lines")):
        raise SystemExit("proposal evidence source status mismatch")
    inputs = manifest.get("inputs", {})
    if (
            stable_receipt_fingerprint(
                bundle / "source" / "pr_riscv_m5ops.build.json") !=
            inputs.get("proposal_guest_receipt_stable_sha256")):
        raise SystemExit("proposal guest stable receipt mismatch")
    if (
            stable_receipt_fingerprint(
                bundle / "source" /
                "test_ecg_load_modes_riscv_m5ops.build.json") !=
            inputs.get("probe_guest_receipt_stable_sha256")):
        raise SystemExit("probe guest stable receipt mismatch")
    rows = validate_rows(bundle / "run" / "combined_roi_matrix.csv")
    by_policy = {row["policy_label"]: row for row in rows}
    run_marker = json.loads(
        (bundle / "run" / "run.complete.json").read_text())
    matrix_marker = json.loads(
        (bundle / "run" / "matrix" /
         "roi_matrix.complete.json").read_text())
    resolved_manifest = json.loads(
        (bundle / "run" / "resolved_manifest.json").read_text())
    if not run_marker.get("complete"):
        raise SystemExit("frozen proposal run is incomplete")
    if not (
            matrix_marker.get("complete") and
            matrix_marker.get("all_rows_ok")):
        raise SystemExit("frozen proposal matrix is incomplete")
    if not descriptor_matches(
            bundle / "run" / "combined_roi_matrix.csv",
            run_marker.get("outputs", {}).get(
                "combined_roi_matrix.csv", {})):
        raise SystemExit("frozen run marker output mismatch")
    for name, expected in matrix_marker.get("outputs", {}).items():
        if not descriptor_matches(
                bundle / "run" / "matrix" / name, expected):
            raise SystemExit("frozen matrix marker output mismatch")
    jobs = resolved_manifest.get("jobs", [])
    if len(jobs) != 1:
        raise SystemExit("frozen resolved manifest job count mismatch")
    fingerprints = jobs[0].get("metadata", {}).get(
        "input_fingerprints", {})
    inputs = manifest.get("inputs", {})
    if (
            fingerprints.get("gem5_binary") !=
            inputs.get("gem5_opt", {}).get("sha256")):
        raise SystemExit("frozen run/probe gem5 hash mismatch")
    if (
            fingerprints.get("gem5_guest_build_receipt_stable") !=
            inputs.get("proposal_guest_receipt_stable_sha256")):
        raise SystemExit("frozen run guest receipt mismatch")
    proposal_guest_sha = inputs.get("proposal_guest_sha256")
    if (
            run_marker.get("gem5_guest_gate", {}).get("detail") !=
            proposal_guest_sha or
            {
                row.get("gem5_guest_expected_sha256") for row in rows
            } != {proposal_guest_sha}):
        raise SystemExit("frozen run guest binary mismatch")
    if {
            row.get("gem5_opt_expected_sha256") for row in rows
            } != {inputs.get("gem5_opt", {}).get("sha256")}:
        raise SystemExit("frozen row gem5 hash mismatch")
    for policy, row in by_policy.items():
        logs = list((bundle / "raw" / "logs").glob(
            f"gem5_pr_{policy}_L3*.log"))
        stderr_path = bundle / "raw" / "stderr" / f"{policy}.txt"
        environment_path = (
            bundle / "raw" / "environments" / f"{policy}.json")
        if len(logs) != 1 or not (
                stderr_path.is_file() and environment_path.is_file()):
            raise SystemExit(f"frozen raw artifact roster mismatch: {policy}")
        validate_raw_policy(
            policy, row,
            parse_raw_policy_receipts(logs[0], stderr_path),
            json.loads(environment_path.read_text()))
    probe_receipt = json.loads(
        (bundle / "probe" / "decoder_probe_receipt.json").read_text())
    validate_probe_claims(probe_receipt, bundle / "probe")
    expected_probe_outputs = {
        "atomic_normal.log", "atomic_teeth.log",
        "atomic_proposal_format_teeth.log", "o3_proposal.log"}
    for name in expected_probe_outputs:
        expected = probe_receipt.get("outputs", {}).get(name)
        if not expected or not descriptor_matches(
                bundle / "probe" / name, expected):
            raise SystemExit(f"frozen probe output mismatch: {name}")
    matrix_paths = [
        bundle / "pipeline" / "roi_matrix_all.csv",
        bundle / "run" / "combined_roi_matrix.csv",
        bundle / "run" / "matrix" / "roi_matrix.csv",
    ]
    derived_inadmissible = derive_inadmissible_columns(matrix_paths)
    if manifest.get("inadmissible_columns") != derived_inadmissible:
        raise SystemExit("proposal inadmissible-column declaration is stale")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="")
    parser.add_argument("--pipeline-dir", default="")
    parser.add_argument("--probe-dir", default="")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument(
        "--verify", default="",
        help="Verify an existing frozen bundle without live /tmp inputs.")
    args = parser.parse_args(argv)

    if args.verify:
        verify_bundle(Path(args.verify).resolve())
        print(f"[ok] {Path(args.verify).resolve()}")
        return 0
    if not (args.run_dir and args.pipeline_dir and args.probe_dir):
        parser.error(
            "--run-dir, --pipeline-dir, and --probe-dir are required "
            "unless --verify is used")

    run_dir = Path(args.run_dir).resolve()
    pipeline_dir = Path(args.pipeline_dir).resolve()
    probe_dir = Path(args.probe_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        raise SystemExit(f"refusing to overwrite evidence directory: {out_dir}")
    source_git_state, source_status, source_diff = git_state()

    run_marker = json.loads((run_dir / "run.complete.json").read_text())
    if not run_marker.get("complete"):
        raise SystemExit("proposal run is not complete")
    rows_path = run_dir / "combined_roi_matrix.csv"
    rows = validate_rows(rows_path)
    rows_descriptor = descriptor(rows_path)
    run_rows_descriptor = run_marker.get("outputs", {}).get(
        "combined_roi_matrix.csv")
    if run_rows_descriptor != rows_descriptor:
        raise SystemExit("run marker does not bind combined_roi_matrix.csv")
    resolved_manifest_path = run_dir / "resolved_manifest.json"
    resolved_manifest = json.loads(resolved_manifest_path.read_text())
    jobs = resolved_manifest.get("jobs", [])
    if len(jobs) != 1:
        raise SystemExit("proposal evidence requires exactly one resolved job")
    job = jobs[0]
    expected_fingerprints = {
        "git_state": paper_run_git_state_fingerprint(),
        "manifest": path_fingerprint(
            ROOT / "scripts" / "experiments" / "ecg" /
            "final_paper_manifest.json"),
        "paper_run": path_fingerprint(
            ROOT / "scripts" / "experiments" / "ecg" / "flows" /
            "paper_run.py"),
        "roi_matrix": path_fingerprint(
            ROOT / "scripts" / "experiments" / "ecg" /
            "roi_matrix.py"),
        "policy_specs": path_fingerprint(
            ROOT / "scripts" / "experiments" / "ecg" /
            "policy_specs.py"),
        "gem5_binary": path_fingerprint(
            ROOT / "bench" / "include" / "gem5_sim" / "gem5" /
            "build" / "RISCV" / "gem5.opt"),
        "gem5_config": path_fingerprint(
            ROOT / "bench" / "include" / "gem5_sim" / "configs" /
            "graphbrew"),
        "gem5_benchmark_binary": path_fingerprint(
            ROOT / "bench" / "bin_gem5" / "pr_riscv_m5ops"),
        "gem5_guest_build_receipt_stable": stable_receipt_fingerprint(
            ROOT / "bench" / "bin_gem5" /
            "pr_riscv_m5ops.build.json"),
    }
    actual_fingerprints = job.get("metadata", {}).get(
        "input_fingerprints", {})
    for name, expected in expected_fingerprints.items():
        if actual_fingerprints.get(name) != expected:
            raise SystemExit(
                f"resolved run source fingerprint mismatch: {name}")

    probe_receipt = json.loads(
        (probe_dir / "decoder_probe_receipt.json").read_text())
    validate_probe_claims(probe_receipt, probe_dir)
    expected_probe_inputs = {
        "atomic_config", "decoder_overlay", "gem5_opt",
        "guest", "o3_config", "verifier",
    }
    if set(probe_receipt.get("inputs", {})) != expected_probe_inputs:
        raise SystemExit("decoder probe input set is incomplete")
    for expected in probe_receipt["inputs"].values():
        if not descriptor_matches(Path(expected["path"]), expected):
            raise SystemExit("decoder probe input descriptor mismatch")
    expected_probe_outputs = {
        "atomic_normal.log", "atomic_teeth.log",
        "atomic_proposal_format_teeth.log", "o3_proposal.log"}
    if set(probe_receipt.get("outputs", {})) != expected_probe_outputs:
        raise SystemExit("decoder probe output set is incomplete")
    for name, expected in probe_receipt["outputs"].items():
        if not descriptor_matches(probe_dir / name, expected):
            raise SystemExit("decoder probe output descriptor mismatch")

    pipeline_manifest = json.loads(
        (pipeline_dir / "paper_pipeline_manifest.json").read_text())
    if not pipeline_manifest.get("inputs") or not pipeline_manifest.get(
            "outputs"):
        raise SystemExit("paper pipeline manifest is not hash-bound")
    pipeline_rows = pipeline_manifest["inputs"].get(
        "run_0/combined_roi_matrix.csv", {})
    if (
            pipeline_rows.get("sha256") != rows_descriptor["sha256"] or
            pipeline_rows.get("size") != rows_descriptor["size"] or
            pipeline_rows.get("rows") != rows_descriptor["rows"]):
        raise SystemExit("paper pipeline input does not match proposal rows")
    pipeline_run_marker = pipeline_manifest["inputs"].get(
        "run_0/run.complete.json", {})
    run_marker_descriptor = descriptor(run_dir / "run.complete.json")
    if (
            pipeline_run_marker.get("sha256") !=
            run_marker_descriptor["sha256"] or
            pipeline_run_marker.get("size") !=
            run_marker_descriptor["size"]):
        raise SystemExit("paper pipeline input does not match run marker")
    if pipeline_manifest.get("git_state") != source_git_state:
        raise SystemExit("paper pipeline and freeze source states differ")
    expected_pipeline_inputs = {
        "run_0/combined_roi_matrix.csv",
        "run_0/resolved_manifest.json",
        "run_0/run.complete.json",
    }
    if set(pipeline_manifest["inputs"]) != expected_pipeline_inputs:
        raise SystemExit("paper pipeline input set is not exact")
    for expected in pipeline_manifest["inputs"].values():
        if not descriptor_matches(Path(expected["path"]), expected):
            raise SystemExit("paper pipeline input descriptor mismatch")
    expected_pipeline_outputs = {
        "aggregate/roi_matrix_all.csv",
        "aggregate/roi_policy_summary.csv",
        "tables/roi_policy_summary.tex",
    }
    if set(pipeline_manifest["outputs"]) != expected_pipeline_outputs:
        raise SystemExit("paper pipeline output set is not exact")
    for expected in pipeline_manifest["outputs"].values():
        if not descriptor_matches(Path(expected["path"]), expected):
            raise SystemExit("paper pipeline output descriptor mismatch")
    for expected in pipeline_manifest.get("scripts", {}).values():
        if not descriptor_matches(Path(expected["path"]), expected):
            raise SystemExit("paper pipeline script descriptor mismatch")
    probe_gem5_sha = probe_receipt["inputs"]["gem5_opt"]["sha256"]
    row_gem5_shas = {
        row.get("gem5_opt_expected_sha256") for row in rows
    }
    if row_gem5_shas != {probe_gem5_sha}:
        raise SystemExit("probe and proposal run used different gem5 binaries")
    proposal_guest_sha = run_marker["gem5_guest_gate"]["detail"]
    if {
            row.get("gem5_guest_expected_sha256") for row in rows
            } != {proposal_guest_sha}:
        raise SystemExit("proposal rows do not match the run guest gate")

    matrix_dirs = list((run_dir / "matrices").glob(
        "60_gem5_proposal_k2m_o3/kron_s12_k4/pr"))
    if len(matrix_dirs) != 1:
        raise SystemExit("proposal matrix directory is missing")
    matrix_dir = matrix_dirs[0]
    matrix_marker = json.loads(
        (matrix_dir / "roi_matrix.complete.json").read_text())
    if not matrix_marker.get("complete") or not matrix_marker.get(
            "all_rows_ok"):
        raise SystemExit("proposal matrix completion marker failed")
    for name, expected in matrix_marker.get("outputs", {}).items():
        if not descriptor_matches(matrix_dir / name, expected):
            raise SystemExit("proposal matrix output descriptor mismatch")
    by_policy = {row["policy_label"]: row for row in rows}
    raw_artifacts: dict[str, Path] = {}
    for policy, row in by_policy.items():
        logs = list((matrix_dir / "logs").glob(
            f"gem5_pr_{policy}_L3*.log"))
        command_receipts = list((matrix_dir / "logs").glob(
            f"gem5_pr_{policy}_L3*.log.cmd"))
        environment_receipts = list((matrix_dir / "logs").glob(
            f"gem5_pr_{policy}_L3*.log.env.json"))
        stderr_files = list((matrix_dir / "gem5").glob(
            f"gem5_pr_{policy}_L3*/benchmark_stderr.txt"))
        if not (
                len(logs) == len(command_receipts) ==
                len(environment_receipts) ==
                len(stderr_files) == 1):
            raise SystemExit(f"raw artifact roster mismatch for {policy}")
        raw = parse_raw_policy_receipts(logs[0], stderr_files[0])
        cell_env = json.loads(environment_receipts[0].read_text())
        validate_raw_policy(policy, row, raw, cell_env)
        raw_artifacts.update({
            f"raw/logs/{logs[0].name}": logs[0],
            f"raw/commands/{command_receipts[0].name}":
                command_receipts[0],
            f"raw/environments/{policy}.json":
                environment_receipts[0],
            f"raw/stderr/{policy}.txt": stderr_files[0],
        })

    copies = {
        "run/run.complete.json": run_dir / "run.complete.json",
        "run/resolved_manifest.json": run_dir / "resolved_manifest.json",
        "run/combined_roi_matrix.csv": rows_path,
        "probe/decoder_probe_receipt.json":
            probe_dir / "decoder_probe_receipt.json",
        "probe/atomic_normal.log": probe_dir / "atomic_normal.log",
        "probe/atomic_teeth.log": probe_dir / "atomic_teeth.log",
        "probe/atomic_proposal_format_teeth.log":
            probe_dir / "atomic_proposal_format_teeth.log",
        "probe/o3_proposal.log": probe_dir / "o3_proposal.log",
        "pipeline/paper_pipeline_manifest.json":
            pipeline_dir / "paper_pipeline_manifest.json",
        "pipeline/roi_policy_summary.csv":
            pipeline_dir / "aggregate" / "roi_policy_summary.csv",
        "pipeline/roi_policy_summary.tex":
            pipeline_dir / "tables" / "roi_policy_summary.tex",
        "pipeline/roi_matrix_all.csv":
            pipeline_dir / "aggregate" / "roi_matrix_all.csv",
        "source/pr_riscv_m5ops.build.json":
            ROOT / "bench" / "bin_gem5" /
            "pr_riscv_m5ops.build.json",
        "source/test_ecg_load_modes_riscv_m5ops.build.json":
            ROOT / "bench" / "bin_gem5" /
            "test_ecg_load_modes_riscv_m5ops.build.json",
        "source/freeze_proposal_k2m.py": Path(__file__),
        "run/matrix/roi_matrix.complete.json":
            matrix_dir / "roi_matrix.complete.json",
        "run/matrix/roi_matrix.csv": matrix_dir / "roi_matrix.csv",
        "run/matrix/roi_matrix.json": matrix_dir / "roi_matrix.json",
    }
    copies.update(raw_artifacts)
    for relative, source in copies.items():
        copy_file(source, out_dir / relative)
    (out_dir / "source").mkdir(parents=True, exist_ok=True)
    (out_dir / "source" / "status.porcelain").write_bytes(source_status)
    with (out_dir / "source" / "source.diff.gz").open("wb") as handle:
        with gzip.GzipFile(
                filename="", mode="wb", fileobj=handle, mtime=0) as archive:
            archive.write(source_diff)
    copies.update({
        "source/status.porcelain": out_dir / "source" / "status.porcelain",
        "source/source.diff.gz": out_dir / "source" / "source.diff.gz",
    })

    inadmissible_columns = derive_inadmissible_columns([
        pipeline_dir / "aggregate" / "roi_matrix_all.csv",
        rows_path,
        matrix_dir / "roi_matrix.csv",
    ])
    manifest = {
        "schema": "graphbrew-proposal-k2m-o3-evidence-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "claim_scope": (
            "Mechanism correctness only: compact 4-byte StreamShield record "
            "load followed by K2-M property load. Exact request binding is "
            "attested for accepted LLC deliveries observed within the first "
            "2048 traced K2 requests; request-count, performance, and SOTA "
            "coverage are not claimed. Accept traces are emitted only after "
            "the simulator dest-line guard; non-accepted traced requests are "
            "unclassified."),
        "inadmissible_columns": inadmissible_columns,
        "raw_matrix_note": (
            "Raw ROI matrices are retained for auditability but contain "
            "timing and cache counters that are inadmissible for comparison "
            "because the 8-byte semantic anchor is width-unmatched."),
        "git_state": source_git_state,
        "source_paths": {
            "run_dir": str(run_dir),
            "pipeline_dir": str(pipeline_dir),
            "probe_dir": str(probe_dir),
        },
        "semantic_receipt": {
            "rows": len(rows),
            "pr_iterations": rows[0]["pr_iterations"],
            "pr_semantic_edges": rows[0]["pr_semantic_edges"],
            "pr_score_checksum": rows[0]["pr_score_checksum"],
            "proposal_rows": {
                row["policy_label"]: {
                    "timing_caveat": row["timing_caveat"],
                    "gem5_k2_delivery_trace_saturated":
                        row["gem5_k2_delivery_trace_saturated"],
                    "gem5_k2_request_trace_events":
                        row["gem5_k2_request_trace_events"],
                    "gem5_k2_request_trace_max_seq":
                        row["gem5_k2_request_trace_max_seq"],
                    "gem5_k2_coalesced_line_accepts":
                        row["gem5_k2_coalesced_line_accepts"],
                }
                for row in rows
                if row["policy_label"] in TARGET_POLICIES
            },
        },
        "inputs": {
            "gem5_opt": probe_receipt["inputs"]["gem5_opt"],
            "probe_guest": probe_receipt["inputs"]["guest"],
            "proposal_guest_sha256":
                proposal_guest_sha,
            "proposal_guest_receipt_stable_sha256":
                stable_receipt_fingerprint(
                    ROOT / "bench" / "bin_gem5" /
                    "pr_riscv_m5ops.build.json"),
            "probe_guest_receipt_stable_sha256":
                stable_receipt_fingerprint(
                    ROOT / "bench" / "bin_gem5" /
                    "test_ecg_load_modes_riscv_m5ops.build.json"),
            "decoder_overlay": probe_receipt["inputs"]["decoder_overlay"],
            "o3_config": probe_receipt["inputs"]["o3_config"],
        },
        "files": {
            relative: descriptor(out_dir / relative)
            for relative in copies
        },
        "generators": {
            "freeze_script": descriptor(
                out_dir / "source" / "freeze_proposal_k2m.py"),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(f"[write] {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
