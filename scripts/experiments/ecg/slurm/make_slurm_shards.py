#!/usr/bin/env python3
"""Generate one-policy Slurm shards from the final-paper manifest."""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path


ECG_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ECG_DIR.parents[2]
sys.path.insert(0, str(ECG_DIR))

from flows import paper_run  # noqa: E402


def build_rows(args: argparse.Namespace) -> list[tuple[str, ...]]:
    manifest = paper_run.load_manifest(Path(args.manifest))
    graph_sets = manifest.get("graph_sets", {})
    rows: list[tuple[str, ...]] = []

    for profile in args.profile:
        for stage in manifest.get("stages", []):
            if stage.get("kind") != "roi_matrix":
                continue
            if profile not in stage.get("profiles", []):
                continue
            if args.only and not any(
                    token in str(stage["name"]) for token in args.only):
                continue

            graph_set_name = str(stage["graph_set"])
            if graph_set_name not in graph_sets:
                raise SystemExit(
                    f"unknown graph_set={graph_set_name!r} "
                    f"in stage {stage['name']}")
            policies = paper_run.filter_policy_specs(
                [str(policy) for policy in stage.get("policies", [])],
                args.policy,
            )
            for graph in graph_sets[graph_set_name]:
                graph_name = str(graph["name"])
                if not paper_run.token_matches(graph_name, args.graph):
                    continue
                if not paper_run.graph_uses_synthetic_options(graph):
                    paper_run.find_graph_path(
                        graph, Path(args.graph_dir),
                        args.allow_missing_graphs)
                for benchmark in stage.get("benchmarks", []):
                    if not paper_run.token_matches(
                            str(benchmark), args.benchmark):
                        continue
                    for policy in policies:
                        rows.append((
                            profile,
                            str(stage["name"]),
                            graph_name,
                            str(benchmark),
                            policy,
                            args.run_tag,
                        ))

    if args.smoke and rows:
        return rows[:1]
    return rows


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate Slurm shard TSV rows from final_paper_manifest.json")
    parser.add_argument("--profile", nargs="+", default=["ecg_smoke"])
    parser.add_argument("--manifest", default=str(paper_run.DEFAULT_MANIFEST))
    parser.add_argument(
        "--graph-dir", default=str(PROJECT_ROOT / "results" / "graphs"))
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--graph", nargs="*", default=[])
    parser.add_argument("--benchmark", nargs="*", default=[])
    parser.add_argument("--policy", nargs="*", default=[])
    parser.add_argument(
        "--run-tag",
        default=f"ecg_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    parser.add_argument("--out", required=True)
    parser.add_argument("--allow-missing-graphs", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    args.run_tag = paper_run.sanitize(args.run_tag)
    rows = build_rows(args)
    if not rows:
        raise SystemExit("no shard rows matched the requested filters")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as handle:
        csv.writer(handle, delimiter="\t", lineterminator="\n").writerows(rows)
    print(f"[slurm-shards] wrote {len(rows)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
