#!/usr/bin/env python3
"""Generate Slurm TSV rows for ECG_PFX BFS scale-proof runs."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import final_paper_run


DEFAULT_OUT_ROOT = "results/ecg_experiments/ecg_pfx_scale_proof"


@dataclass(frozen=True)
class ScaleShardRow:
    scale: int
    root: int
    backend: str
    out_root: str

    def to_tsv(self) -> str:
        return "\t".join((str(self.scale), str(self.root), self.backend, self.out_root))


def parse_range_token(token: str) -> list[int]:
    if ":" in token:
        pieces = token.split(":")
        if len(pieces) not in (2, 3):
            raise SystemExit(f"invalid range token {token!r}; expected start:end[:step]")
        start = int(pieces[0])
        end = int(pieces[1])
        step = int(pieces[2]) if len(pieces) == 3 else 1
        if step <= 0:
            raise SystemExit(f"invalid non-positive range step in {token!r}")
        return list(range(start, end + 1, step))
    return [int(token)]


def expand_ints(tokens: list[str]) -> list[int]:
    values: list[int] = []
    for token in tokens:
        values.extend(parse_range_token(str(token)))
    return sorted(dict.fromkeys(values))


def generate_rows(scales: list[int], roots: list[int], backends: list[str], run_tag: str, out_root: str) -> list[ScaleShardRow]:
    rows: list[ScaleShardRow] = []
    for scale in scales:
        for root in roots:
            for backend in backends:
                shard_out_root = f"{out_root.rstrip('/')}/{run_tag}/g{scale}_r{root}_{backend}"
                rows.append(ScaleShardRow(scale=scale, root=root, backend=backend, out_root=shard_out_root))
    return rows


def generate_rows_for_pairs(pairs: list[tuple[int, int]], backends: list[str], run_tag: str, out_root: str) -> list[ScaleShardRow]:
    rows: list[ScaleShardRow] = []
    for scale, root in pairs:
        for backend in backends:
            shard_out_root = f"{out_root.rstrip('/')}/{run_tag}/g{scale}_r{root}_{backend}"
            rows.append(ScaleShardRow(scale=scale, root=root, backend=backend, out_root=shard_out_root))
    return rows


def selected_pairs_from_combined(path: Path, source_backend: str, min_pf_useful: int) -> list[tuple[int, int]]:
    pairs: set[tuple[int, int]] = set()
    with path.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if source_backend and str(row.get("backend", "")) != source_backend:
                continue
            if str(row.get("status", "")) != "ok":
                continue
            try:
                pf_useful = int(float(str(row.get("pf_useful", "0") or "0")))
            except ValueError:
                pf_useful = 0
            if pf_useful < min_pf_useful:
                continue
            scale = row.get("scale") or row.get("shard_scale")
            root = row.get("root") or row.get("shard_root")
            if scale in (None, "") or root in (None, ""):
                continue
            pairs.add((int(scale), int(root)))
    return sorted(pairs)


def write_rows(path: Path, rows: list[ScaleShardRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{row.to_tsv()}\n" for row in rows))


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate ECG_PFX BFS scale-proof Slurm shard TSV rows.")
    parser.add_argument("--scale", nargs="+", default=[], help="Scale(s), e.g. 11 12 or 11:13.")
    parser.add_argument("--root", nargs="+", default=[], help="Root(s), e.g. 0 20 or 0:31.")
    parser.add_argument("--backend", nargs="+", choices=["sniper", "gem5-riscv", "both"], default=["both"])
    parser.add_argument("--from-combined", default="", help="Select scale/root pairs from an ecg_pfx_scale_status.py --combined CSV.")
    parser.add_argument("--source-backend", default="sniper-sift", help="Backend to select from --from-combined rows.")
    parser.add_argument("--min-pf-useful", type=int, default=1, help="Minimum useful prefetches required when using --from-combined.")
    parser.add_argument("--run-tag", required=True, help="Run tag embedded into each out_root.")
    parser.add_argument("--out-root", default=DEFAULT_OUT_ROOT, help="Base output root for shard run directories.")
    parser.add_argument("--out", default="-", help="Output TSV path, or '-' for stdout.")
    return parser.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    backends = [str(value) for value in args.backend]
    if args.from_combined:
        pairs = selected_pairs_from_combined(
            final_paper_run.resolve_path(str(args.from_combined)),
            source_backend=str(args.source_backend),
            min_pf_useful=int(args.min_pf_useful),
        )
        rows = generate_rows_for_pairs(pairs, backends, str(args.run_tag), str(args.out_root))
    else:
        if not args.scale or not args.root:
            raise SystemExit("--scale and --root are required unless --from-combined is provided")
        rows = generate_rows(
            scales=expand_ints([str(value) for value in args.scale]),
            roots=expand_ints([str(value) for value in args.root]),
            backends=backends,
            run_tag=str(args.run_tag),
            out_root=str(args.out_root),
        )
    if not rows:
        raise SystemExit("no scale-proof shard rows generated")

    if args.out == "-":
        for row in rows:
            print(row.to_tsv())
    else:
        out_path = final_paper_run.resolve_path(str(args.out))
        write_rows(out_path, rows)
        print(f"[write] {out_path} rows={len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))