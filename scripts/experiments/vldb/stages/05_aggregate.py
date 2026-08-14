#!/usr/bin/env python3
"""Stage 05 — Aggregate JSON results into LaTeX tables + PNG figures.

Reads everything under results/vldb_paper/ and emits paper-ready
artefacts via vldb_generate_figures.py. Pure I/O, no benchmarks.

Examples:
    python3 scripts/experiments/vldb/stages/05_aggregate.py --exp 0
    # (--exp is required by the common parser but ignored here; use any value.)
"""
from __future__ import annotations
import argparse
import os
import sys
import json
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.experiments.vldb.stages._common import (  # noqa: F401
    add_common_args, resolve_config, V,
)
from scripts.experiments.vldb.config import (
    PAPER_ARTIFACT_ROOT, PAPER_GRAPH_ROOT,
)

def build_paper_package(
    paper_dir: Path,
    manifest_path: Path | None = None,
) -> Path:
    paper_dir = paper_dir.resolve()
    manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else paper_dir / "paper-package-manifest.json"
    )
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Private paper manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    package_name = manifest.get("package_name")
    package_files = manifest.get("files")
    if (
        not isinstance(package_name, str)
        or not package_name.endswith(".zip")
        or Path(package_name).name != package_name
        or not isinstance(package_files, list)
        or not package_files
        or any(
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            for relative in package_files
        )
    ):
        raise ValueError("Private paper package manifest is invalid")
    if len(package_files) != len(set(package_files)):
        raise ValueError("Private paper manifest contains duplicates")
    resolved_files = {
        relative: (paper_dir / relative).resolve()
        for relative in package_files
    }
    if any(
        not path.is_relative_to(paper_dir)
        for path in resolved_files.values()
    ):
        raise ValueError(
            "Private paper manifest escapes the paper workspace")
    missing = [
        relative for relative in package_files
        if not resolved_files[relative].is_file()
    ]
    if missing:
        raise RuntimeError(
            "Cannot package paper; missing: " + ", ".join(missing)
        )
    package = paper_dir / package_name
    with zipfile.ZipFile(
        package,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for relative in sorted(package_files):
            data = resolved_files[relative].read_bytes()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, data)
    return package


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # We accept --exp for argparse uniformity but this stage aggregates ALL exps.
    p.add_argument("--exp", type=int, default=0,
                   help="Ignored — figure generator aggregates all exps under results/vldb_paper/.")
    p.add_argument("--no-figures", action="store_true",
                   help="Skip figure generation; only rebuild INDEX.json.")
    p.add_argument("--artifact-root", type=str, default=str(PAPER_ARTIFACT_ROOT),
                   help="Root containing vldb_paper/, vldb_mappings/, and vldb_runs/.")
    p.add_argument("--graph-dir", type=str, default=str(PAPER_GRAPH_ROOT),
                   help="External graph root to include in INDEX.json.")
    p.add_argument("--publish-paper-figures", action="store_true",
                   help="Copy generated figures/tables into the private paper workspace.")
    p.add_argument("--package-paper", action="store_true",
                   help="Build a deterministic source ZIP after publishing.")
    p.add_argument(
        "--paper-dir",
        type=Path,
        default=(
            Path(os.environ["GRAPHBREW_PRIVATE_PAPER_ROOT"])
            if os.environ.get("GRAPHBREW_PRIVATE_PAPER_ROOT")
            else None
        ),
        help="Private paper workspace (required for publish/package).",
    )
    p.add_argument(
        "--paper-package-manifest",
        type=Path,
        help="Private package manifest (default: <paper-dir>/paper-package-manifest.json).",
    )
    p.add_argument("--dry-run", action="store_true",
                   help="Print the aggregation plan without writing files.")
    args, ignored = p.parse_known_args()
    if ignored:
        print(f"Stage 05 ignoring selection/runtime flags: {' '.join(ignored)}")
    V.configure_artifact_root(args.artifact_root)
    if args.dry_run:
        print(
            f"DRY RUN — would aggregate {args.artifact_root} "
            f"with graph root {args.graph_dir}"
        )
        return
    if args.publish_paper_figures:
        if args.paper_dir is None:
            p.error(
                "--publish-paper-figures requires --paper-dir or "
                "GRAPHBREW_PRIVATE_PAPER_ROOT"
            )
        os.environ["GRAPHBREW_PRIVATE_PAPER_ROOT"] = str(
            args.paper_dir.resolve())
        os.environ["GRAPHBREW_PUBLISH_PAPER_FIGURES"] = "1"
    print("STAGE 05 — rebuilding results/INDEX.json")
    try:
        from scripts.lib.analysis.results_index import write_index
        out = write_index(
            Path(args.artifact_root),
            graph_root=Path(args.graph_dir) if args.graph_dir else None,
        )
        print(f"  wrote {out}")
    except Exception as e:
        print(f"  WARN: could not build INDEX.json ({e})")

    if not args.no_figures:
        print("STAGE 05 — generating figures + LaTeX tables from results/vldb_paper/")
        V._generate_figures()
    if args.package_paper:
        if args.paper_dir is None:
            p.error(
                "--package-paper requires --paper-dir or "
                "GRAPHBREW_PRIVATE_PAPER_ROOT"
            )
        package = build_paper_package(
            args.paper_dir,
            args.paper_package_manifest,
        )
        print(f"STAGE 05 — wrote paper package {package}")
    print("STAGE 05 COMPLETE.")


if __name__ == "__main__":
    main()
