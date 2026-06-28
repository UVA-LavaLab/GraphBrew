#!/usr/bin/env python3
"""experiments.py -- THE single, config-driven entry point for ECG experiments.

Anti-bloat contract: scripts/experiments/ecg/ once held 177 ad-hoc scripts. It
now holds a small fixed set of runners, and EVERY experiment is DEFINED in
experiments.json and RUN through this orchestrator. The actual simulation is
delegated to roi_matrix.py (the cache_sim/gem5/Sniper driver).

   >>> To add an experiment: add a named entry under "experiments" in
   >>> experiments.json. Do NOT create a new top-level script. <<<

An experiment is a cartesian product expanded into resumable roi_matrix cells:
    graphs x l3_sizes x policies x benchmarks
Every field falls back to "defaults"; "graphs": "@eval" resolves a graph_set.

Usage:
  experiments.py list                          # experiments defined in the config
  experiments.py show  <name>                  # the fully-resolved cells of one
  experiments.py run   <name> [opts]           # run it (resumable)
  experiments.py verify [--pfx|--equiv|--kernels] # correctness (ecg/pfx/equiv/multi-kernel gate)
  experiments.py analyze                       # aggregate the latest scale run

run options:
  --dry-run            print the roi_matrix commands, do not execute
  --only SUBSTR        only cells whose "graph/benchmark/l3/policy" contains SUBSTR
  --force              re-run cells whose output already exists
  --config PATH        use a different config (default: experiments.json)
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PY = sys.executable
ROI = HERE / "roi_matrix.py"
GRAPHS = ROOT / "results" / "graphs"
RESULTS = ROOT / "results" / "ecg_experiments"
DEFAULT_CONFIG = HERE / "experiments.json"

# config field -> roi_matrix scalar flag
FLAG_MAP = {
    "suite": "--suite", "l1d_size": "--l1d-size", "l2_size": "--l2-size",
    "l3_ways": "--l3-ways", "line_size": "--line-size",
    "ecg_epoch_pack_bits": "--ecg-epoch-pack-bits", "ecg_epochs": "--ecg-epochs",
    "ecg_charged": "--ecg-charged", "cache_stream_prefetch_degree": "--cache-stream-prefetch-degree",
    "structure_prefetch_degree": "--structure-prefetch-degree",
    "popt_reserve_model": "--popt-reserve-model", "popt_active_columns": "--popt-active-columns",
    "timeout_cache": "--timeout-cache", "timeout_gem5": "--timeout-gem5",
    "timeout_sniper": "--timeout-sniper",
    "sniper_workload": "--sniper-workload", "sniper_memory_limit_gb": "--sniper-memory-limit-gb",
}

# config field -> roi_matrix store_true (boolean) flag, appended only when truthy
BOOL_FLAG_MAP = {
    "allow_sniper_sg_kernel_workload": "--allow-sniper-sg-kernel-workload",
    "allow_sniper_benchmark_workload": "--allow-sniper-benchmark-workload",
}


def sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in ".-" else "_" for c in str(s))


def load_config(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        raise SystemExit(f"config not found: {path}")
    except json.JSONDecodeError as e:
        raise SystemExit(f"invalid JSON in {path}: {e}")


def resolve_graphs(spec, cfg) -> list[str]:
    if isinstance(spec, str) and spec.startswith("@"):
        s = cfg.get("graph_sets", {}).get(spec[1:])
        if s is None:
            raise SystemExit(f"unknown graph_set {spec}")
        return list(s)
    return list(spec) if isinstance(spec, (list, tuple)) else [spec]


def _as_list(v):
    return list(v) if isinstance(v, (list, tuple)) else [v]


def cell_tag(cell: dict) -> str:
    """graph/bench/l3/policy[/prefetcher] — the prefetcher is shown only when not 'none'."""
    t = f"{cell['graph']}/{cell['benchmark']}/{cell['l3']}/{cell['policy']}"
    pf = cell.get("prefetcher", "none")
    return t + (f"/{pf}" if pf != "none" else "")


def expand(name: str, cfg: dict) -> tuple[dict, list[dict]]:
    """Return (settings, cells) for an experiment. Each cell is a dict of the
    concrete (graph, benchmark, l3, policy) + the merged settings."""
    exp = cfg.get("experiments", {}).get(name)
    if exp is None:
        raise SystemExit(f"unknown experiment '{name}'. Run: experiments.py list")
    s = {**cfg.get("defaults", {}), **exp}
    graphs = resolve_graphs(s["graphs"], cfg)
    benches = _as_list(s["benchmark"])
    l3s = _as_list(s["l3_sizes"])
    policies = _as_list(s["policies"])
    prefetchers = _as_list(s.get("prefetchers", "none"))
    cells = []
    for g, b, l3, pol, pf in itertools.product(graphs, benches, l3s, policies, prefetchers):
        cells.append({"graph": g, "benchmark": b, "l3": l3, "policy": pol, "prefetcher": pf})
    return s, cells


def cell_cmd(name: str, s: dict, cell: dict) -> tuple[list[str], Path, dict]:
    g, b, l3, pol = cell["graph"], cell["benchmark"], cell["l3"], cell["policy"]
    pf = cell.get("prefetcher", "none")
    gpath = GRAPHS / g / f"{g}.sg"
    out = RESULTS / sanitize(name) / sanitize(g) / sanitize(b) / sanitize(l3) / sanitize(pol)
    if pf != "none":
        out = out / f"pf_{sanitize(pf)}"
    cmd = [PY, str(ROI), "--benchmark", b, "--policies", pol,
           "--options", f"-f {gpath} {s.get('bench_options', {}).get(b, s.get('options', ''))}".strip(),
           "--l3-sizes", str(l3), "--prefetcher", pf, "--out-dir", str(out)]
    for key, flag in FLAG_MAP.items():
        if key in s and s[key] is not None:
            cmd += [flag, str(s[key])]
    for key, flag in BOOL_FLAG_MAP.items():
        if s.get(key):
            cmd.append(flag)
    if s.get("no_build"):
        cmd.append("--no-build")
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    # Per-benchmark ECG eviction variant (epoch confidence). Sweep kernels (PR)
    # trust the epoch (epoch_only = epoch vetoes recency); frontier kernels
    # (BFS/SSSP/CC/BC) use rrip_first (degree-first, epoch only breaks ties among
    # GRASP's equal-RRPV victims) which is do-no-harm vs GRASP. Falls back to the
    # experiment-wide ecg_variant when a benchmark is not listed.
    variant = s.get("bench_variant", {}).get(b, s.get("ecg_variant"))
    if variant:
        env["ECG_VARIANT"] = variant
    return cmd, out, env


def cmd_list(cfg: dict) -> int:
    print(__doc__.split("Usage:")[0].rstrip())
    print(f"\nExperiments (in {DEFAULT_CONFIG.name}):")
    for nm, e in cfg.get("experiments", {}).items():
        print(f"  {nm:<22} {e.get('description','')}")
    print("\nGraph sets:", ", ".join(cfg.get("graph_sets", {})))
    return 0


def cmd_show(name: str, cfg: dict) -> int:
    s, cells = expand(name, cfg)
    print(f"# {name}: {s.get('description','')}")
    npf = len(set(c.get('prefetcher', 'none') for c in cells))
    print(f"# suite={s['suite']}  {len(cells)} cells "
          f"({len(set(c['graph'] for c in cells))} graphs x "
          f"{len(set(c['benchmark'] for c in cells))} bench x "
          f"{len(set(c['l3'] for c in cells))} L3 x "
          f"{len(set(c['policy'] for c in cells))} policies"
          f"{f' x {npf} prefetchers' if npf > 1 else ''})")
    for c in cells:
        pf = c.get('prefetcher', 'none')
        print(f"  {c['graph']:18} {c['benchmark']:5} {c['l3']:>5} {c['policy']:<22}"
              f"{'' if pf == 'none' else '  +' + pf}")
    return 0


def cmd_run(name: str, cfg: dict, dry: bool, only: str | None, force: bool) -> int:
    s, cells = expand(name, cfg)
    if only:
        cells = [c for c in cells if only in cell_tag(c)]
    print(f"[experiments] {name}: {len(cells)} cells, suite={s['suite']}"
          f"{' (DRY-RUN)' if dry else ''}")
    done = run = skip = fail = 0
    for i, cell in enumerate(cells, 1):
        cmd, out, env = cell_cmd(name, s, cell)
        tag = cell_tag(cell)
        if not force and (out / "roi_matrix.json").exists():
            skip += 1
            continue
        if dry:
            print(f"  [{i}/{len(cells)}] {tag}\n    $ {' '.join(cmd)}")
            continue
        out.mkdir(parents=True, exist_ok=True)
        print(f"  [{i}/{len(cells)}] {tag} ...", flush=True)
        rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
        if rc == 0:
            run += 1
        else:
            fail += 1
            print(f"      FAILED (rc={rc})")
    print(f"[experiments] {name}: ran={run} skipped={skip} failed={fail}")
    return 1 if fail else 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("command", nargs="?", default="list")
    ap.add_argument("name", nargs="?")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--pfx", action="store_true")
    ap.add_argument("--equiv", action="store_true",
                    help="run the behavioral cross-sim equivalence + insertion-RRPV gate")
    ap.add_argument("--kernels", action="store_true",
                    help="run the multi-kernel (PR/BFS/BC) x 3-sim eviction-spec + debug equivalence")
    args, extra = ap.parse_known_args(argv)

    if args.command in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if args.command == "verify":
        if args.pfx:
            target = "verify/pfx.py"
        elif args.equiv:
            target = "verify/equiv.py"
        elif args.kernels:
            target = "verify/equiv_kernels.py"
        else:
            target = "verify/ecg.py"
        return subprocess.run([PY, str(HERE / target), *extra], cwd=str(ROOT)).returncode
    if args.command == "analyze":
        return subprocess.run([PY, str(HERE / "analysis/scale.py"), *extra],
                              cwd=str(ROOT)).returncode

    cfg = load_config(Path(args.config))
    if args.command == "list":
        return cmd_list(cfg)
    if args.command == "show":
        return cmd_show(args.name, cfg)
    if args.command == "run":
        return cmd_run(args.name, cfg, args.dry_run, args.only, args.force)
    print(f"unknown command '{args.command}'. Run: experiments.py list", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
