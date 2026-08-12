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
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from scripts.experiments.vldb.stages._common import (  # noqa: F401
    add_common_args, resolve_config, V,
)
from scripts.experiments.vldb.config import (
    PAPER_ARTIFACT_ROOT, PAPER_GRAPH_ROOT,
)

PAPER_PACKAGE_NAME = (
    "GraphBrew__Multilayered_Graph_Reordering_Techniques_for_"
    "Accelerated_Graph_Processing__VLDB_2026_.zip"
)
PAPER_PACKAGE_FILES = [
    "VLDB_2026.tex",
    "references_sync.bib",
    "ACM-Reference-Format.bst",
    "acmart.cls",
    "paperFigures/Fig1c.png",
    "paperFigures/Fig2.png",
    "paperFigures/Fig3.png",
    "paperFigures/Fig4.png",
    "paperFigures/Fig5.png",
    "paperFigures/Fig6.png",
    "paperFigures/Fig7.png",
    "dataCharts/cache/cacheGM.png",
    "dataCharts/cache/Cit-Patents.png",
    "dataCharts/cache/Com-Orkut.png",
    "dataCharts/cache/Hollywood.png",
    "dataCharts/cache/road.png",
    "dataCharts/speedup/aggregateSpeedups.png",
    "dataCharts/speedup/overheadReorder.png",
    "dataCharts/speedup/BFS.png",
    "dataCharts/speedup/PR.png",
    "dataCharts/speedup/SpMV.png",
    "dataCharts/speedup/SSSP.png",
    "dataCharts/speedup/CC.png",
    "dataCharts/speedup/CC_SV.png",
    "dataCharts/speedup/BC.png",
    "dataCharts/trials/aggregateTrials.png",
    "dataCharts/trials/BFS.png",
    "dataCharts/trials/PR.png",
    "dataCharts/trials/SpMV.png",
    "dataCharts/trials/SSSP.png",
    "dataCharts/trials/CC.png",
    "dataCharts/trials/CC_SV.png",
    "dataCharts/trials/BC.png",
    "dataCharts/tables/table_kernel_speedup.tex",
    "dataCharts/tables/table_end_to_end.tex",
    "dataCharts/tables/table_ablation.tex",
    "dataCharts/tables/table_sensitivity.tex",
    "dataCharts/tables/table_chained.tex",
    "dataCharts/tables/table_scalability.tex",
]
PAPER_PACKAGE_FILES += [
    Path(relative).with_suffix(".svg").as_posix()
    for relative in PAPER_PACKAGE_FILES
    if relative.startswith("dataCharts/")
    and relative.endswith(".png")
]


def build_paper_package(paper_dir: Path | None = None) -> Path:
    paper_dir = (
        paper_dir.resolve()
        if paper_dir is not None
        else Path(__file__).resolve().parents[4] / "research"
    )
    missing = [
        relative for relative in PAPER_PACKAGE_FILES
        if not (paper_dir / relative).is_file()
    ]
    if missing:
        raise RuntimeError(
            "Cannot package paper; missing: " + ", ".join(missing)
        )
    package = paper_dir / PAPER_PACKAGE_NAME
    with zipfile.ZipFile(
        package,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for relative in sorted(PAPER_PACKAGE_FILES):
            data = (paper_dir / relative).read_bytes()
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
                   help="Copy generated figures/tables into research/dataCharts/.")
    p.add_argument("--package-paper", action="store_true",
                   help="Build a deterministic source ZIP after publishing.")
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
        package = build_paper_package()
        print(f"STAGE 05 — wrote paper package {package}")
    print("STAGE 05 COMPLETE.")


if __name__ == "__main__":
    main()
