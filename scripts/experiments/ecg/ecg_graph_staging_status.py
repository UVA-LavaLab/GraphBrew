#!/usr/bin/env python3
"""Report graph staging status for ECG final-run profiles."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import final_paper_run
from scripts.lib.pipeline.download import get_graph_info


FIELDNAMES = [
    "graph",
    "expected_path",
    "exists",
    "size_bytes",
    "profiles",
    "stages",
    "jobs",
    "status",
    "detail",
    "source_url",
    "source_size_mb",
    "source_nodes",
    "source_edges",
    "source_symmetric",
    "source_category",
    "download_command",
    "convert_command",
]


def graph_jobs(manifest: dict[str, Any], profiles: list[str]) -> dict[str, dict[str, Any]]:
    selected_profiles = set(profiles)
    graph_sets = manifest.get("graph_sets", {})
    out: dict[str, dict[str, Any]] = {}
    for stage in manifest.get("stages", []):
        if str(stage.get("kind", "")) != "roi_matrix":
            continue
        stage_profiles = set(str(profile) for profile in stage.get("profiles", []))
        profiles_for_stage = sorted(selected_profiles.intersection(stage_profiles))
        if not profiles_for_stage:
            continue
        settings = final_paper_run.merged_defaults(manifest, stage)
        graph_set_name = str(settings.get("graph_set", ""))
        for graph in graph_sets.get(graph_set_name, []):
            if final_paper_run.graph_uses_synthetic_options(graph):
                continue
            graph_name = str(graph["name"])
            record = out.setdefault(graph_name, {"graph": graph, "profiles": set(), "stages": set(), "jobs": 0})
            record["profiles"].update(profiles_for_stage)
            record["stages"].add(str(stage["name"]))
            record["jobs"] += len(settings.get("benchmarks", [])) * len(settings.get("policies", [])) * len(profiles_for_stage)
    return out


def expected_graph_path(graph: dict[str, Any], graph_dir: Path) -> Path:
    if "path" in graph:
        return final_paper_run.resolve_path(str(graph["path"]))
    return graph_dir / str(graph["name"]) / f"{graph['name']}.sg"


def catalog_info(graph_name: str):
    aliases = {
        "com-orkut": "com-Orkut",
    }
    for candidate in (graph_name, aliases.get(graph_name, "")):
        if not candidate:
            continue
        info = get_graph_info(candidate)
        if info is not None:
            return info
    return None


def staging_rows(manifest: dict[str, Any], profiles: list[str], graph_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for graph_name, record in sorted(graph_jobs(manifest, profiles).items()):
        path = expected_graph_path(record["graph"], graph_dir)
        exists = path.exists()
        info = catalog_info(graph_name)
        mtx_path = path.parent / f"{graph_name}.mtx"
        rows.append({
            "graph": graph_name,
            "expected_path": str(path),
            "exists": int(exists),
            "size_bytes": path.stat().st_size if exists else 0,
            "profiles": ";".join(sorted(record["profiles"])),
            "stages": ";".join(sorted(record["stages"])),
            "jobs": int(record["jobs"]),
            "status": "ok" if exists else "missing",
            "detail": "staged" if exists else f"stage graph at {path}",
            "source_url": info.url if info else "",
            "source_size_mb": info.size_mb if info else "",
            "source_nodes": info.nodes if info else "",
            "source_edges": info.edges if info else "",
            "source_symmetric": int(info.symmetric) if info else "",
            "source_category": info.category if info else "",
            "download_command": f"python3 -m scripts.lib.pipeline.download --graph {graph_name} --dest {graph_dir}",
            "convert_command": f"make converter && bench/bin/converter -f {mtx_path} -s -b {path}",
        })
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Report ECG final-run graph staging status.")
    parser.add_argument("--manifest", default=str(final_paper_run.DEFAULT_MANIFEST))
    parser.add_argument("--profile", nargs="+", default=["final_replacement", "final_droplet", "final_cache_sim", "final_cache_sim_ecg_pfx"])
    parser.add_argument("--graph-dir", default=str(final_paper_run.PROJECT_ROOT / "results" / "graphs"))
    parser.add_argument("--out", default="-", help="Output CSV path, or '-' for stdout.")
    parser.add_argument("--fail-on-missing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    manifest = final_paper_run.load_manifest(final_paper_run.resolve_path(str(args.manifest)))
    rows = staging_rows(manifest, [str(profile) for profile in args.profile], final_paper_run.resolve_path(str(args.graph_dir)))
    if args.out == "-":
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    else:
        out_path = final_paper_run.resolve_path(str(args.out))
        write_csv(out_path, rows)
        missing = sum(1 for row in rows if row["status"] == "missing")
        print(f"[write] {out_path} rows={len(rows)} missing={missing}")
    if args.fail_on_missing and any(row["status"] == "missing" for row in rows):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))