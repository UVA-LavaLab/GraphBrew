#!/usr/bin/env python3
"""Reproduce the public P-OPT PageRank cache-miss direction.

The claimable gate is deliberately narrow:

  LRU, DRRIP, P-OPT; 24 MiB/16-way LLC; no prefetch; one PR sweep.

It does not reproduce the paper's 8-core Sniper speedups or Figure 12(a)'s
P-OPT-vs-GRASP result. The optional direct-DBG mode is a non-claimable
diagnostic using official GRASP RRIP rules on the P-OPT workload's registered
regions, not official GRASP's PageRank property/frontier mapping.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable


PINNED_ARTIFACT_COMMIT = "53b5021846690d0f3445428c6380e877ecf7a10e"
PINNED_GRASP_COMMIT = "6e3814430265fc4f2513c95ef131a6522bc9d389"
GRASP_PROXY = "grasp-rules-proxy"
PROXY_GATE = "dbg-grasp-rules-proxy"
POLICIES = {
    "lru": ("baseline", "lru"),
    "drrip": ("baseline", "drrip"),
    "popt-8b": ("popt", "popt-8b"),
    "opt-ideal": ("opt-ideal", "opt-ideal"),
    GRASP_PROXY: ("baseline", GRASP_PROXY),
}
PUBLIC_POLICIES = ("lru", "drrip", "popt-8b")
DEFAULT_GRAPHS = (
    "uk-2002",
    "hugebubbles-00020",
    "kron25-d4",
    "urand25-d4",
)
DBG_GRAPHS = tuple(f"{graph}-dbg-direct" for graph in DEFAULT_GRAPHS)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_tree(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return sha256(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(sha256(child).encode())
    return digest.hexdigest()


def parse_total_llc_misses(text: str) -> int:
    values = []
    for line in text.splitlines():
        if "[LLC-STAT] Total Misses" not in line:
            continue
        values.append(int(float(line.rsplit(" ", 1)[-1])))
    if not values:
        raise ValueError("artifact output contains no LLC Total Misses")
    return sum(values)


def parse_app_error(text: str) -> float:
    values = [
        float(line.split("=", 1)[1].strip())
        for line in text.splitlines()
        if line.startswith("[APP] Error =")
    ]
    if len(values) != 1 or not math.isfinite(values[0]):
        raise ValueError("output lacks one finite [APP] Error receipt")
    return values[0]


def validate_completed_output(text: str) -> tuple[int, float]:
    markers = (
        "~~~ PINTOOL STATS BEGIN ~~~",
        "~~~ PINTOOL STATS END ~~~",
        "[APP] Error =",
        "[PIN-FINI] App Exit Code = 0",
    )
    if any(marker not in text for marker in markers):
        raise ValueError("output lacks normal-completion markers")
    if not (
            text.index(markers[0]) < text.index(markers[1]) <
            text.index(markers[2]) < text.index(markers[3])):
        raise ValueError("normal-completion markers are out of order")
    return parse_total_llc_misses(text), parse_app_error(text)


def artifact_head(root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root,
        capture_output=True, text=True, check=True)
    return result.stdout.strip()


def expected_paths(
        root: Path, pin_root: Path, tool_root: Path, app_root: Path,
        graphs: Iterable[str], policies: Iterable[str]) -> list[Path]:
    paths = [pin_root / "pin"]
    for graph in graphs:
        paths.append(root / "input-graphs" / f"{graph}.sg")
    for policy in policies:
        app_version, tool_name = POLICIES[policy]
        paths.extend((
            app_root / app_version / "pr",
            tool_root / tool_name / "cache_pinsim.so",
        ))
    return paths


def build_command(
        root: Path, pin_root: Path, tool_root: Path, app_root: Path,
        graph: str, policy: str, setarch: str) -> list[str]:
    app_version, tool_name = POLICIES[policy]
    return [
        setarch, "x86_64", "-R",
        str(pin_root / "pin"),
        "-t", str(tool_root / tool_name / "cache_pinsim.so"),
        "--",
        str(app_root / app_version / "pr"),
        "-f", str(root / "input-graphs" / f"{graph}.sg"),
        "-n", "1", "-i", "1",
    ]


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_graph_provenance(
        path: Path, root: Path, graphs: Iterable[str],
        artifact_commit: str) -> dict:
    data = json.loads(path.read_text())
    if data.get("artifact_commit") != artifact_commit:
        raise SystemExit("graph provenance artifact commit mismatch")
    graph_entries = data.get("graphs")
    if not isinstance(graph_entries, dict):
        raise SystemExit("graph provenance lacks a graphs object")
    for graph in graphs:
        name = f"{graph}.sg"
        if name not in graph_entries:
            raise SystemExit(f"graph provenance lacks {name}")
        entry = graph_entries[name]
        declared = entry.get("sha256") if isinstance(entry, dict) else entry
        actual = sha256(root / "input-graphs" / name)
        if declared != actual:
            raise SystemExit(
                f"graph provenance hash mismatch for {name}: "
                f"{declared} != {actual}")
    return data


def verify_port_build_manifest(
        path: Path, port_source: Path, artifact_commit: str,
        pin_root: Path, tool_root: Path, app_root: Path,
        policies: Iterable[str]) -> dict:
    data = json.loads(path.read_text())
    if data.get("schema_version") != 2:
        raise SystemExit("unsupported port build manifest schema")
    if data.get("setup_script_sha256") != sha256(port_source):
        raise SystemExit("port build manifest does not match setup script")
    if data.get("popt_repository", {}).get("commit") != artifact_commit:
        raise SystemExit("port build manifest P-OPT commit mismatch")
    expected_tool_root = (path.parent / "bin").resolve()
    expected_app_root = (path.parent / "apps").resolve()
    if tool_root != expected_tool_root:
        raise SystemExit(
            f"--tool-root must be manifest output {expected_tool_root}")
    if app_root != expected_app_root:
        raise SystemExit(
            f"--app-root must be manifest output {expected_app_root}")
    pin = data.get("pin", {})
    pin_checks = {
        "pin_sha256": sha256(pin_root / "pin"),
        "intel64_tree_sha256": hash_tree(pin_root / "intel64"),
        "config_tree_sha256": hash_tree(pin_root / "source/tools/Config"),
    }
    for field, actual in pin_checks.items():
        if pin.get(field) != actual:
            raise SystemExit(f"port build manifest Pin mismatch: {field}")
    if data.get("smoke", {}).get("passed") is not True:
        raise SystemExit("port build manifest lacks a passing smoke test")
    if data.get("smoke", {}).get("normal_application_completion") is not True:
        raise SystemExit("port smoke did not prove normal app completion")
    if data.get("smoke", {}).get("semantic_error_match") is not True:
        raise SystemExit("port smoke did not prove semantic error parity")

    binaries = data.get("binaries", {})
    source_trees = data.get("generated_source_trees", {})
    applications = data.get("applications", {})
    app_source_trees = data.get("application_source_trees", {})
    app_versions = {POLICIES[policy][0] for policy in policies}
    for policy in policies:
        tool_name = POLICIES[policy][1]
        binary = tool_root / tool_name / "cache_pinsim.so"
        if binaries.get(tool_name) != sha256(binary):
            raise SystemExit(f"manifest binary mismatch for {tool_name}")
        source = path.parent / "src" / tool_name
        if source_trees.get(tool_name) != hash_tree(source):
            raise SystemExit(f"manifest source-tree mismatch for {tool_name}")
    for app_version in app_versions:
        binary = app_root / app_version / "pr"
        if applications.get(app_version, {}).get("pr") != sha256(binary):
            raise SystemExit(f"manifest app mismatch for {app_version}/pr")
        source = path.parent / "app-src" / app_version
        if app_source_trees.get(app_version) != hash_tree(source):
            raise SystemExit(
                f"manifest app-source mismatch for {app_version}")
    if GRASP_PROXY in policies:
        if data.get("grasp_repository", {}).get("commit") != \
                PINNED_GRASP_COMMIT:
            raise SystemExit("GRASP proxy source commit mismatch")
        proxy = data.get("grasp_rules_proxy", {})
        if proxy.get("claimable_as_official_grasp") is not False or \
                proxy.get("official_workload_mapping") is not False:
            raise SystemExit("GRASP proxy manifest overclaims workload fidelity")
        preserved = path.parent / "provenance" / "grasp"
        if proxy.get("preserved_source_sha256") != hash_tree(preserved):
            raise SystemExit("preserved GRASP source hash mismatch")
    return data


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--pin-root", type=Path)
    parser.add_argument("--tool-root", type=Path)
    parser.add_argument("--app-root", type=Path)
    parser.add_argument("--port-source-root", type=Path)
    parser.add_argument("--port-build-manifest", type=Path)
    parser.add_argument("--graph-provenance", type=Path)
    parser.add_argument("--port-label", default="pin2-exact")
    parser.add_argument(
        "--gate", choices=("public", PROXY_GATE), default="public")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--graphs", nargs="+", default=list(DEFAULT_GRAPHS))
    parser.add_argument(
        "--policies", nargs="+", choices=sorted(POLICIES),
        default=None)
    parser.add_argument("--timeout", type=int, default=86400)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--exploratory", action="store_true",
        help="Allow graph/policy subsets that do not evaluate the public gate.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    public_gate = args.gate == "public"
    if args.policies is None:
        args.policies = list(
            PUBLIC_POLICIES if public_gate
            else ("lru", GRASP_PROXY, "popt-8b"))

    root = args.artifact_root.resolve()
    pin_root = (args.pin_root or root / "pin-2.14").resolve()
    tool_root = (args.tool_root or root / "simulators").resolve()
    app_root = (args.app_root or root / "applications").resolve()
    setarch = shutil.which("setarch")
    if not setarch:
        raise SystemExit(
            "setarch is required: artifact addresses are hashed into cache sets")
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    head = artifact_head(root)
    if head != PINNED_ARTIFACT_COMMIT:
        raise SystemExit(
            f"artifact commit mismatch: {head} != {PINNED_ARTIFACT_COMMIT}")
    required_graphs = set(DEFAULT_GRAPHS if public_gate else DBG_GRAPHS)
    required_policies = set(
        PUBLIC_POLICIES if public_gate
        else ("lru", GRASP_PROXY, "popt-8b"))
    if not args.exploratory and (
            set(args.graphs) != required_graphs or
            set(args.policies) != required_policies):
        raise SystemExit(
            f"{args.gate} requires graphs={sorted(required_graphs)} and "
            f"policies={sorted(required_policies)}; use --exploratory for a "
            "non-gating subset")

    inputs = expected_paths(
        root, pin_root, tool_root, app_root,
        args.graphs, args.policies)
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise SystemExit("missing reproduction inputs:\n  " + "\n  ".join(missing))

    port_source = (
        args.port_source_root.resolve()
        if args.port_source_root else None)
    build_manifest_path = (
        args.port_build_manifest.resolve()
        if args.port_build_manifest else None)
    if args.port_label != "pin2-exact":
        if port_source is None or not port_source.is_file():
            raise SystemExit(
                "compatibility ports require a setup script file via "
                "--port-source-root")
        if build_manifest_path is None or not build_manifest_path.is_file():
            raise SystemExit(
                "compatibility ports require --port-build-manifest")
        build_receipt = verify_port_build_manifest(
            build_manifest_path, port_source, head, pin_root,
            tool_root, app_root, args.policies)
    else:
        if build_manifest_path is not None:
            raise SystemExit("pin2-exact must not use a port build manifest")
        build_receipt = {}

    graph_provenance = (
        args.graph_provenance.resolve()
        if args.graph_provenance else None)
    if graph_provenance is None or not graph_provenance.is_file():
        raise SystemExit("--graph-provenance JSON is required")
    verify_graph_provenance(
        graph_provenance, root, args.graphs, head)

    port_source_hash = sha256(port_source) if port_source else ""
    build_manifest_hash = (
        sha256(build_manifest_path) if build_manifest_path else "")
    manifest = {
        "artifact_commit": head,
        "port_label": args.port_label,
        "gate": args.gate,
        "scope": (
            "public-artifact PageRank LLC-miss direction"
            if public_gate else
            "non-claimable direct-DBG GRASP-rules proxy diagnostic"),
        "claimable": {
            "popt_vs_drrip": public_gate,
            "popt_vs_grasp_direction": False,
            "popt_vs_grasp_figure12_exact": False,
            "grasp_rules_proxy_direction": False,
            "execution_time": False,
        },
        "grasp_workload_mapping_faithful": False if not public_gate else None,
        "graphs": list(args.graphs),
        "policies": list(args.policies),
        "aslr_disabled": True,
        "normal_application_completion_required": True,
        "semantic_error_match_required": True,
        "port_source_sha256": port_source_hash,
        "port_build_manifest_sha256": build_manifest_hash,
        "graph_provenance_sha256": sha256(graph_provenance),
        "inputs": {
            str(path): sha256(path)
            for path in (*inputs, graph_provenance)
        },
    }
    if port_source:
        manifest["inputs"][str(port_source)] = port_source_hash
    if build_manifest_path:
        manifest["inputs"][str(build_manifest_path)] = build_manifest_hash
        manifest["port_build"] = {
            "schema_version": build_receipt["schema_version"],
            "popt_tree": build_receipt["popt_repository"]["tree"],
            "pin_version": build_receipt["pin"]["version"],
            "smoke_passed": build_receipt["smoke"]["passed"],
        }
        shutil.copy2(
            build_manifest_path, out_dir / "port_build_manifest.json")
        shutil.copy2(port_source, out_dir / "setup_popt_pin4_port.py")
    shutil.copy2(graph_provenance, out_dir / "graph_provenance.json")
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    rows = []
    results_path = out_dir / "results.csv"
    if args.resume and results_path.exists():
        with results_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if row.get("status") == "ok":
                row["exit_code"] = int(row["exit_code"])
                row["llc_demand_misses"] = int(row["llc_demand_misses"])
                row["app_error"] = float(row["app_error"])
    runner_hash = sha256(Path(__file__).resolve())
    fingerprints = {}
    for graph in args.graphs:
        for policy in args.policies:
            command = build_command(
                root, pin_root, tool_root, app_root,
                graph, policy, setarch)
            app_version, tool_name = POLICIES[policy]
            payload = {
                "command": command,
                "runner_sha256": runner_hash,
                "pin_sha256": sha256(pin_root / "pin"),
                "tool_sha256": sha256(
                    tool_root / tool_name / "cache_pinsim.so"),
                "app_sha256": sha256(
                    app_root / app_version / "pr"),
                "graph_sha256": sha256(
                    root / "input-graphs" / f"{graph}.sg"),
                "port_label": args.port_label,
                "port_source_sha256": port_source_hash,
                "port_build_manifest_sha256": build_manifest_hash,
                "graph_provenance_sha256": manifest[
                    "graph_provenance_sha256"],
            }
            fingerprints[(graph, policy)] = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()).hexdigest()
    completed = {
        (row["graph"], row["policy"]) for row in rows
        if row.get("status") == "ok" and
        row.get("execution_fingerprint") ==
        fingerprints.get((row["graph"], row["policy"]))
    }
    env = dict(os.environ, OMP_NUM_THREADS="1")
    for graph in args.graphs:
        for policy in args.policies:
            if (graph, policy) in completed:
                print(f"[resume] {graph}/{policy}", flush=True)
                continue
            command = build_command(
                root, pin_root, tool_root, app_root,
                graph, policy, setarch)
            print("$", " ".join(command), flush=True)
            if args.dry_run:
                continue
            log = out_dir / f"{graph}__{policy}.stdout"
            err = out_dir / f"{graph}__{policy}.stderr"
            with log.open("w") as stdout, err.open("w") as stderr:
                result = subprocess.run(
                    command, cwd=root / "scripts", env=env,
                    stdout=stdout, stderr=stderr, timeout=args.timeout,
                    check=False)
            text = log.read_text(errors="ignore")
            status = "ok"
            failure = ""
            misses = ""
            app_error = ""
            if result.returncode != 0:
                status = "error"
                failure = f"nonzero exit code {result.returncode}"
            else:
                try:
                    misses, app_error = validate_completed_output(text)
                except ValueError as error:
                    status = "error"
                    failure = str(error)
            row = {
                "graph": graph,
                "policy": policy,
                "exit_code": result.returncode,
                "llc_demand_misses": misses,
                "app_error": app_error,
                "execution_fingerprint": fingerprints[(graph, policy)],
                "stdout_sha256": sha256(log),
                "stderr_sha256": sha256(err),
                "status": status,
                "failure": failure,
            }
            rows.append(row)
            write_csv(results_path, rows)
            if status != "ok":
                raise SystemExit(f"{graph}/{policy} failed: {failure}")

    if args.dry_run:
        return 0

    by_graph = {
        graph: {
            row["policy"]: row for row in rows
            if row["graph"] == graph and row["status"] == "ok"
        }
        for graph in args.graphs
    }
    complete = all(
        set(by_graph[graph]) == set(args.policies)
        for graph in args.graphs)
    semantic_match = {}
    for graph, policies in by_graph.items():
        values = [row["app_error"] for row in policies.values()]
        semantic_match[graph] = bool(values) and all(
            math.isclose(value, values[0], rel_tol=1e-9, abs_tol=1e-12)
            for value in values[1:])

    reference = "drrip" if public_gate else GRASP_PROXY
    direction = {}
    ratios = {}
    for graph, policies in by_graph.items():
        if "popt-8b" in policies and reference in policies:
            popt_misses = policies["popt-8b"]["llc_demand_misses"]
            reference_misses = policies[reference]["llc_demand_misses"]
            direction[graph] = popt_misses < reference_misses
            ratios[graph] = {
                "popt_over_reference": popt_misses / reference_misses,
                "miss_reduction_pct": (
                    1.0 - popt_misses / reference_misses) * 100.0,
            }
    direction_evaluated = {"popt-8b", reference} <= set(args.policies)
    semantic_complete = (
        len(semantic_match) == len(args.graphs) and
        all(semantic_match.values()))
    public_passed = (
        public_gate and complete and semantic_complete and
        direction_evaluated and len(direction) == len(args.graphs) and
        all(direction.values()))
    proxy_complete = (
        not public_gate and complete and semantic_complete and
        direction_evaluated and len(direction) == len(args.graphs))
    (out_dir / "complete.json").write_text(json.dumps({
        "complete": complete,
        "normal_application_completion": complete,
        "semantic_error_match": semantic_match,
        "semantic_complete": semantic_complete,
        "direction_evaluated": direction_evaluated,
        "reference_policy": reference,
        "passed_popt_better_than_reference_every_graph": (
            public_passed if public_gate else None),
        "proxy_diagnostic_complete": proxy_complete if not public_gate else None,
        "popt_better_than_reference": direction,
        "popt_vs_reference": ratios,
        "popt_vs_grasp_direction_claimable": False,
        "popt_vs_grasp_figure12_exact": False,
    }, indent=2, sort_keys=True) + "\n")
    if public_gate:
        return 0 if (
            complete and semantic_complete and
            ((args.exploratory and not direction_evaluated) or public_passed)
        ) else 1
    return 0 if proxy_complete else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
