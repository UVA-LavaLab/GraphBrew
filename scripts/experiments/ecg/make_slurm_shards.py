#!/usr/bin/env python3
"""Generate final_paper_run.py Slurm shard TSV rows from the manifest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import final_paper_run


DEFAULT_MANIFEST = final_paper_run.DEFAULT_MANIFEST
DEFAULT_GRAPH_DIR = final_paper_run.PROJECT_ROOT / "results" / "graphs"
SMOKE_PROFILE = "final_replacement"
SMOKE_GRAPH = "cit-Patents"
SMOKE_BENCHMARK = "pr"
SMOKE_POLICY = "LRU"


@dataclass(frozen=True)
class ShardRow:
    profile: str
    stage: str
    graph: str
    benchmark: str
    policy: str
    run_tag: str

    def to_tsv(self) -> str:
        return "\t".join((self.profile, self.stage, self.graph, self.benchmark, self.policy, self.run_tag))


def matching_profiles(stage: dict[str, Any], requested_profiles: list[str]) -> list[str]:
    stage_profiles = {str(profile) for profile in stage.get("profiles", [])}
    return [profile for profile in requested_profiles if profile in stage_profiles]


def generate_shards(
    manifest: dict[str, Any],
    profiles: list[str],
    run_tag: str,
    graph_dir: Path,
    stage_filters: list[str],
    graph_filters: list[str],
    benchmark_filters: list[str],
    policy_filters: list[str],
    allow_missing_graphs: bool,
) -> list[ShardRow]:
    graph_sets = manifest.get("graph_sets", {})
    rows: list[ShardRow] = []

    for stage in manifest.get("stages", []):
        if str(stage.get("kind", "")) != "roi_matrix":
            continue
        profiles_for_stage = matching_profiles(stage, profiles)
        if not profiles_for_stage:
            continue
        stage_name = str(stage["name"])
        if stage_filters and not any(token in stage_name for token in stage_filters):
            continue

        settings = final_paper_run.merged_defaults(manifest, stage)
        graph_set_name = str(settings["graph_set"])
        if graph_set_name not in graph_sets:
            raise SystemExit(f"unknown graph_set={graph_set_name!r} in stage {stage_name}")

        for graph in graph_sets[graph_set_name]:
            graph_name = str(graph["name"])
            if not final_paper_run.token_matches(graph_name, graph_filters):
                continue
            if not final_paper_run.graph_uses_synthetic_options(graph):
                final_paper_run.find_graph_path(graph, graph_dir, allow_missing_graphs)

            for benchmark in settings.get("benchmarks", []):
                benchmark = str(benchmark)
                if not final_paper_run.token_matches(benchmark, benchmark_filters):
                    continue
                policies = final_paper_run.filter_policy_specs(
                    [str(policy) for policy in settings.get("policies", [])],
                    policy_filters,
                )
                for profile in profiles_for_stage:
                    for policy in policies:
                        rows.append(ShardRow(profile, stage_name, graph_name, benchmark, policy, run_tag))

    return rows


def write_shards(path: Path, rows: list[ShardRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row.to_tsv()}\n" for row in rows))


def apply_smoke_defaults(args: argparse.Namespace) -> None:
    if not args.smoke:
        return
    if not args.profile:
        args.profile = [SMOKE_PROFILE]
    if not args.graph:
        args.graph = [SMOKE_GRAPH]
    if not args.benchmark:
        args.benchmark = [SMOKE_BENCHMARK]
    if not args.policy:
        args.policy = [SMOKE_POLICY]


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Slurm shard TSV rows for ECG final-paper runs.")
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST), help="JSON final-run manifest.")
    parser.add_argument("--profile", nargs="+", default=[], help="Profile(s) to expand, e.g. final_replacement final_droplet.")
    parser.add_argument("--run-tag", required=True, help="Run tag written into each shard row.")
    parser.add_argument("--out", default="-", help="Output TSV path, or '-' for stdout.")
    parser.add_argument("--graph-dir", default=str(DEFAULT_GRAPH_DIR), help="Graph root for manifest graph names without explicit paths.")
    parser.add_argument("--smoke", action="store_true", help="Generate the canonical one-row cluster smoke shard unless filters are overridden.")
    parser.add_argument("--stage", nargs="+", default=[], help="Only stages whose name contains one of these tokens.")
    parser.add_argument("--graph", nargs="+", default=[], help="Only exact normalized graph names.")
    parser.add_argument("--benchmark", nargs="+", default=[], help="Only exact normalized benchmark names.")
    parser.add_argument("--policy", nargs="+", default=[], help="Only exact normalized policy labels.")
    parser.add_argument("--allow-missing-graphs", action="store_true", help="Allow shard generation before all graph files are staged.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    apply_smoke_defaults(args)
    if not args.profile:
        raise SystemExit("--profile is required unless --smoke supplies the canonical final_replacement smoke profile")
    manifest = final_paper_run.load_manifest(final_paper_run.resolve_path(args.manifest))
    rows = generate_shards(
        manifest=manifest,
        profiles=[str(profile) for profile in args.profile],
        run_tag=str(args.run_tag),
        graph_dir=final_paper_run.resolve_path(args.graph_dir),
        stage_filters=[str(value) for value in args.stage],
        graph_filters=[str(value) for value in args.graph],
        benchmark_filters=[str(value) for value in args.benchmark],
        policy_filters=[str(value) for value in args.policy],
        allow_missing_graphs=bool(args.allow_missing_graphs),
    )
    if not rows:
        raise SystemExit("no shards selected; check --profile/--stage/--graph/--benchmark/--policy filters")

    if args.out == "-":
        for row in rows:
            print(row.to_tsv())
    else:
        out_path = final_paper_run.resolve_path(str(args.out))
        write_shards(out_path, rows)
        print(f"[write] {out_path} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))