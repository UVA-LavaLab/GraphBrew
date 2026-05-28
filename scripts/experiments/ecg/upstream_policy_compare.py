#!/usr/bin/env python3
"""Compare GraphBrew cache policy replay against official upstream artifacts."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_UPSTREAM = Path("/tmp/graphbrew-faithfulness-upstream")
TRACE_REPLAY_SRC = PROJECT_ROOT / "scripts" / "experiments" / "ecg" / "graphbrew_trace_replay.cc"


def run(cmd: list[str], cwd: Path | None = None, stdout: Path | None = None) -> subprocess.CompletedProcess[str]:
    if stdout:
        stdout.parent.mkdir(parents=True, exist_ok=True)
        with stdout.open("w") as fh:
            return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True,
                                  stdout=fh, stderr=subprocess.STDOUT, check=False)
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)


def ensure_grasp_upstream(upstream_root: Path) -> Path:
    grasp = upstream_root / "grasp"
    if grasp.exists():
        patch_grasp_portability(grasp)
        return grasp
    upstream_root.mkdir(parents=True, exist_ok=True)
    result = run(["git", "clone", "https://github.com/faldupriyank/grasp.git", str(grasp)])
    if result.returncode != 0:
        raise SystemExit(result.stdout)
    result = run(["git", "checkout", "6e3814430265fc4f2513c95ef131a6522bc9d389"], cwd=grasp)
    if result.returncode != 0:
        raise SystemExit(result.stdout)
    patch_grasp_portability(grasp)
    return grasp


def patch_grasp_portability(grasp: Path) -> None:
    common = grasp / "trace-based-simulators" / "common.h"
    text = common.read_text()
    needle = '    printf("\\n");\n}\n\n#endif\n'
    replacement = '    printf("\\n");\n    return 0;\n}\n\n#endif\n'
    if needle in text:
        common.write_text(text.replace(needle, replacement))


def build_upstream_grasp(grasp: Path, policy: str, log_path: Path) -> Path:
    sim_dir = grasp / "trace-based-simulators"
    binary = sim_dir / f"{policy}.bin"
    result = run(["make", "clean"], cwd=sim_dir, stdout=log_path)
    if result.returncode != 0:
        raise SystemExit(f"failed to clean upstream GRASP simulator, see {log_path}")
    result = run(["make", f"POLICY={policy}", f"{policy}.bin"], cwd=sim_dir, stdout=log_path)
    if result.returncode != 0 or not binary.exists():
        raise SystemExit(f"failed to build upstream GRASP {policy}, see {log_path}")
    return binary


def build_graphbrew_replay(out_dir: Path) -> Path:
    binary = out_dir / "graphbrew_trace_replay"
    cmd = [
        "g++", "-std=c++17", "-O2", "-fopenmp",
        "-I", str(PROJECT_ROOT / "bench" / "include"),
        str(TRACE_REPLAY_SRC), "-o", str(binary),
    ]
    result = run(cmd, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        raise SystemExit(result.stdout)
    return binary


def parse_upstream_output(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "total-accesses:" in line:
            out["total_accesses"] = line.split()[-1]
        elif "total-misses:" in line:
            out["total_misses"] = line.split()[-1]
        elif "miss-rate:" in line:
            out["miss_rate"] = line.split()[-1]
    return out


def parse_graphbrew_output(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in text.splitlines():
        if "," not in line:
            continue
        key, value = line.strip().split(",", 1)
        out[key.replace("-", "_")] = value
    return out


def run_compare(args: argparse.Namespace) -> list[dict[str, Any]]:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    grasp = ensure_grasp_upstream(args.upstream_root)
    replay = build_graphbrew_replay(out_dir)

    rows: list[dict[str, Any]] = []
    for policy in args.policies:
        upstream_binary = build_upstream_grasp(grasp, policy, out_dir / "logs" / f"build_grasp_{policy}.log")
        for trace in args.traces:
            trace_path = Path(trace)
            if not trace_path.is_absolute():
                trace_path = grasp / "datasets" / trace
            upstream_result = run([str(upstream_binary), str(trace_path), str(args.cache_mb)])
            graphbrew_result = run([str(replay), str(trace_path), str(args.cache_mb), policy])
            upstream = parse_upstream_output(upstream_result.stdout or "")
            graphbrew = parse_graphbrew_output(graphbrew_result.stdout or "")
            upstream_misses = int(upstream.get("total_misses", "-1")) if upstream.get("total_misses") else None
            graphbrew_misses = int(graphbrew.get("total_misses", "-1")) if graphbrew.get("total_misses") else None
            delta = None
            if upstream_misses is not None and graphbrew_misses is not None:
                delta = graphbrew_misses - upstream_misses
            rows.append({
                "artifact": "grasp_trace_simulator",
                "trace": trace_path.name,
                "policy": policy,
                "cache_mb": args.cache_mb,
                "upstream_status": "ok" if upstream_result.returncode == 0 else "error",
                "graphbrew_status": "ok" if graphbrew_result.returncode == 0 else "error",
                "upstream_total_accesses": upstream.get("total_accesses", ""),
                "graphbrew_total_accesses": graphbrew.get("total_accesses", ""),
                "upstream_total_misses": upstream.get("total_misses", ""),
                "graphbrew_total_misses": graphbrew.get("total_misses", ""),
                "miss_delta_graphbrew_minus_upstream": "" if delta is None else delta,
                "upstream_miss_rate": upstream.get("miss_rate", ""),
                "graphbrew_miss_rate": graphbrew.get("miss_rate", ""),
                "aligned_addresses": graphbrew.get("aligned_addresses", ""),
                "grasp_regions": graphbrew.get("grasp_regions", ""),
                "notes": "same official LLC trace replayed by upstream artifact and GraphBrew cache_sim",
            })
    rows.append(popt_toolchain_row(args.upstream_root))
    return rows


def popt_toolchain_row(upstream_root: Path) -> dict[str, Any]:
    popt = upstream_root / "POPT-CacheSim-HPCA21"
    local_shim = popt / "toolchain" / "g++-4.9"
    gpp49 = shutil.which("g++-4.9")
    if gpp49 is None and local_shim.exists():
        gpp49 = str(local_shim)
    missing = []
    if not (popt / "pin-2.14").exists():
        missing.append("pin-2.14")
    if gpp49 is None:
        missing.append("g++-4.9")
    notes = "official P-OPT requires " + ", ".join(missing) if missing else "Pin 2.14 and a g++-4.9-compatible command are present; dynamic targets may still require the static-app workaround on modern loaders"
    return {
        "artifact": "popt_pin_toolchain",
        "trace": "",
        "policy": "popt",
        "cache_mb": "",
        "upstream_status": "blocked" if missing else "ready",
        "graphbrew_status": "not_run",
        "upstream_total_accesses": "",
        "graphbrew_total_accesses": "",
        "upstream_total_misses": "",
        "graphbrew_total_misses": "",
        "miss_delta_graphbrew_minus_upstream": "",
        "upstream_miss_rate": "",
        "graphbrew_miss_rate": "",
        "aligned_addresses": "",
        "grasp_regions": "",
        "notes": notes,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], args: argparse.Namespace) -> None:
    lines = [
        "# Upstream Policy Comparison",
        "",
        f"Upstream root: `{args.upstream_root}`",
        f"Cache size: `{args.cache_mb}MB`",
        "",
        "## GRASP Trace Simulator",
        "",
        "| Trace | Policy | Upstream Misses | GraphBrew Misses | Delta | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in [row for row in rows if row.get("artifact") == "grasp_trace_simulator"]:
        status = "pass" if row.get("miss_delta_graphbrew_minus_upstream") == 0 else "diff"
        lines.append(
            f"| `{row['trace']}` | `{row['policy']}` | {row['upstream_total_misses']} | "
            f"{row['graphbrew_total_misses']} | {row['miss_delta_graphbrew_minus_upstream']} | {status} |"
        )

    popt_row = next((row for row in rows if row.get("artifact") == "popt_pin_toolchain"), None)
    lines.extend([
        "",
        "## P-OPT Original Artifact",
        "",
        "The official P-OPT simulator is a Pin 2.14 pintool and cannot be run by this local check unless both `pin-2.14` and `g++-4.9` are available.",
        f"Status: `{popt_row['upstream_status'] if popt_row else 'unknown'}`",
        f"Notes: {popt_row['notes'] if popt_row else ''}",
        "",
    ])
    path.write_text("\n".join(lines) + "\n")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare GraphBrew policy replay with upstream artifacts.")
    parser.add_argument("--upstream-root", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--cache-mb", type=int, default=1)
    parser.add_argument("--policies", nargs="+", default=["lru", "grasp"],
                        choices=["lru", "pin", "grasp", "belady"])
    parser.add_argument("--traces", nargs="+", default=["PageRankOpt.web-Google.cvgr.dbg.lru.llc.trace"])
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/graphbrew-upstream-policy-compare"))
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    if not args.out_dir.is_absolute():
        args.out_dir = PROJECT_ROOT / args.out_dir
    rows = run_compare(args)
    write_csv(args.out_dir / "comparison.csv", rows)
    write_markdown(args.out_dir / "comparison.md", rows, args)
    print(f"[write] {args.out_dir / 'comparison.csv'} rows={len(rows)}")
    print(f"[write] {args.out_dir / 'comparison.md'}")
    mismatches = [
        row for row in rows
        if row.get("artifact") == "grasp_trace_simulator"
        and row.get("miss_delta_graphbrew_minus_upstream") not in (0, "0")
    ]
    if mismatches:
        print(f"[warn] mismatched rows={len(mismatches)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))