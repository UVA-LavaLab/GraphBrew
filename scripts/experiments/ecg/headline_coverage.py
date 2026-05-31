#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Headline coverage proof tool (gate 282).

PROOF GATE — not an AST audit. Backed by real experimental measurements.

Reports per (simulator × graph × app × l3 × policy × prefetcher) cell:
  - ok       — cell has a measurement row in the on-disk artifact
  - missing  — no row exists yet (paper-headline gap)

Required cell set is derived from the LITERATURE_BASELINES claim registry
(gate 281 frozen), narrowed to a configurable scope. The scope MUST be
literature-grounded — no hand-tuned cell lists. As literature_baselines.py
grows new claims, this coverage scope expands automatically.

Inputs:
  cache_sim: wiki/data/literature_faithfulness_postfix.csv
  gem5:      wiki/data/gem5_anchor.json
  Sniper:    wiki/data/sniper_anchor.json

Scopes:
  headline_1MB:  the GRASP HPCA'20 + POPT HPCA'21 canonical 1MB row,
                 baseline policies (LRU/SRRIP/GRASP/POPT) + ECG_DBG_PRIMARY,
                 no_pfx only, across 3 sims.
  with_droplet:  same as headline_1MB but adds the DROPLET prefetcher
                 column for the prior-method prefetch comparison.
  full_sweep:    headline_1MB + the 8MB convergence row from
                 LITERATURE_CACHE_ORGS["fits_8MB"].

Ratchet behavior:
  The companion gate 282 pytest asserts present_cell_count ≥ baseline,
  where baseline lives in wiki/data/headline_coverage_baseline.json.
  When real coverage exceeds the baseline, a notice is emitted and the
  user can bump the baseline via --bump-baseline. The baseline can never
  retreat without explicit removal.

Twenty-ninth in the gate series — first PROOF gate (gates 273-281 were
vocabulary-lock AST audits; 282 is the first gate backed by simulator
measurements).
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
LIT_BASELINES = REPO_ROOT / "scripts/experiments/ecg/literature_baselines.py"


# ---------------------------------------------------------------------------
# Sim + policy + prefetcher rosters
# ---------------------------------------------------------------------------

SIMS = ("cache_sim", "gem5", "sniper")

POLICIES_BASELINE = ("LRU", "SRRIP", "GRASP", "POPT")
POLICIES_ECG = ("ECG_DBG_PRIMARY",)
POLICIES_HEADLINE = POLICIES_BASELINE + POLICIES_ECG

PREFETCHERS_NONE = ("no_pfx",)
PREFETCHERS_BASELINE_PLUS_DROPLET = ("no_pfx", "DROPLET")


# Workstation feasibility per graph. Based on ECG-Final-Runs.md /
# ECG-Sniper-Runs.md sizing notes — graphs with >100M edges or graphs
# where Sniper SIFT same-graph runs are documented to need ~50 GiB RSS
# go to the SLURM tier.
WORKSTATION_TIERS = {
    "email-Eu-core":    "LOCAL",          # 1K vertices, 26K edges (sanity)
    "cit-Patents":      "LOCAL",          # 3.8M vertices, 16.5M edges
    "web-Google":       "LOCAL_TIGHT",    # 876K vertices, 5.1M edges
    "soc-pokec":        "LOCAL_TIGHT",    # 1.6M vertices, 30.6M edges
    "soc-LiveJournal1": "SLURM",          # 4.8M vertices, 42.8M edges
    "com-orkut":        "SLURM",          # 3M vertices, 117M edges
}


# ---------------------------------------------------------------------------
# Literature scope derivation (lives in literature_baselines.py — gate 281)
# ---------------------------------------------------------------------------

def _load_literature_baselines():
    """Import literature_baselines.py. Mirrors the pattern used by
    literature_faithfulness.py so a single sys.modules entry is shared
    when both are loaded in the same Python process (test sessions)."""
    if "literature_baselines" in sys.modules:
        return sys.modules["literature_baselines"]
    spec = importlib.util.spec_from_file_location(
        "literature_baselines", LIT_BASELINES
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["literature_baselines"] = mod  # required BEFORE exec_module
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


def literature_concrete_cells(l3_filter: tuple[str, ...] | None = None
                              ) -> set[tuple[str, str, str]]:
    """Return the distinct concrete (graph, app, l3_size) cells the
    literature claims direct results on. Skips pattern claims (graph
    starts with ``*``) — those are tier-1 invariants, not headline cells.

    Optionally filter to a specific L3 set (e.g. only ``("1MB",)``).
    """
    lit = _load_literature_baselines()
    all_claims = list(lit.INVARIANT_CLAIMS)
    if hasattr(lit, "PER_GRAPH_CLAIMS"):
        all_claims.extend(lit.PER_GRAPH_CLAIMS)
    cells: set[tuple[str, str, str]] = set()
    for c in all_claims:
        if c.graph.startswith("*"):
            continue
        if l3_filter is not None and c.l3_size not in l3_filter:
            continue
        cells.add((c.graph, c.app, c.l3_size))
    return cells


# ---------------------------------------------------------------------------
# Cell key
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CellKey:
    sim: str
    graph: str
    app: str
    l3_size: str
    policy: str
    prefetcher: str = "no_pfx"


# ---------------------------------------------------------------------------
# Presence loaders
# ---------------------------------------------------------------------------

def load_cache_sim(csv_path: Path) -> set[CellKey]:
    """Load cache_sim lit-faith rows. The lit-faith CSV is the no_pfx
    baseline sweep — DROPLET rows live in a separate sweep root."""
    out: set[CellKey] = set()
    if not csv_path.exists():
        return out
    with csv_path.open(newline="") as fh:
        for r in csv.DictReader(fh):
            graph = (r.get("graph") or "").strip()
            app = (r.get("app") or "").strip()
            l3 = (r.get("l3_size") or "").strip()
            pol = (r.get("policy") or "").strip()
            miss = (r.get("miss_rate") or "").strip()
            if not (graph and app and l3 and pol and miss):
                continue
            out.add(CellKey("cache_sim", graph, app, l3, pol, "no_pfx"))
    return out


def load_anchor_json(json_path: Path, sim_name: str) -> set[CellKey]:
    """Load gem5/sniper anchor JSON cells."""
    out: set[CellKey] = set()
    if not json_path.exists():
        return out
    try:
        d = json.loads(json_path.read_text())
    except Exception:
        return out
    for c in d.get("cells", []):
        graph = (c.get("graph") or "").strip()
        app = (c.get("app") or "").strip()
        l3 = (c.get("l3_size") or "").strip()
        for pol, miss in (c.get("miss_rate_by_policy") or {}).items():
            if graph and app and l3 and pol:
                out.add(CellKey(sim_name, graph, app, l3, pol, "no_pfx"))
    return out


def load_all_presence(cache_sim_csv: Path, gem5_json: Path,
                      sniper_json: Path) -> set[CellKey]:
    return (load_cache_sim(cache_sim_csv)
            | load_anchor_json(gem5_json, "gem5")
            | load_anchor_json(sniper_json, "sniper"))


# ---------------------------------------------------------------------------
# Required cells
# ---------------------------------------------------------------------------

SCOPE_HEADLINE_1MB = "headline_1MB"
SCOPE_WITH_DROPLET = "with_droplet"
SCOPE_FULL_SWEEP   = "full_sweep"
VALID_SCOPES = (SCOPE_HEADLINE_1MB, SCOPE_WITH_DROPLET, SCOPE_FULL_SWEEP)


def required_cells(scope: str = SCOPE_HEADLINE_1MB) -> set[CellKey]:
    if scope not in VALID_SCOPES:
        raise ValueError(f"unknown scope: {scope}; must be one of {VALID_SCOPES}")
    if scope == SCOPE_HEADLINE_1MB:
        lit_cells = literature_concrete_cells(l3_filter=("1MB",))
        pfx_roster = PREFETCHERS_NONE
    elif scope == SCOPE_WITH_DROPLET:
        lit_cells = literature_concrete_cells(l3_filter=("1MB",))
        pfx_roster = PREFETCHERS_BASELINE_PLUS_DROPLET
    else:  # SCOPE_FULL_SWEEP
        lit_cells = literature_concrete_cells(l3_filter=("1MB", "8MB"))
        pfx_roster = PREFETCHERS_BASELINE_PLUS_DROPLET
    out: set[CellKey] = set()
    for sim in SIMS:
        for graph, app, l3 in lit_cells:
            for pol in POLICIES_HEADLINE:
                for pfx in pfx_roster:
                    out.add(CellKey(sim, graph, app, l3, pol, pfx))
    return out


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def graph_tier(graph: str) -> str:
    return WORKSTATION_TIERS.get(graph, "UNKNOWN")


def render_summary(required: set[CellKey], present: set[CellKey]) -> dict:
    """Return summary dict (totals, per-sim, missing-by-tier)."""
    inscope_present = required & present
    missing = required - present
    per_sim = {}
    for s in SIMS:
        r = {c for c in required if c.sim == s}
        p = {c for c in inscope_present if c.sim == s}
        per_sim[s] = {
            "required": len(r), "present": len(p),
            "missing": len(r) - len(p),
            "coverage_pct": round(100.0 * len(p) / len(r), 2) if r else 0.0,
        }
    by_tier = defaultdict(int)
    for c in missing:
        by_tier[graph_tier(c.graph)] += 1
    return {
        "totals": {
            "required": len(required),
            "present_in_scope": len(inscope_present),
            "missing": len(missing),
            "extra_out_of_scope": len(present - required),
            "coverage_pct": round(100.0 * len(inscope_present) / len(required), 2)
                            if required else 0.0,
        },
        "per_sim": per_sim,
        "missing_by_workstation_tier": dict(by_tier),
    }


def render_per_sim_per_graph(required: set[CellKey],
                              present: set[CellKey]) -> list[dict]:
    rows = []
    graphs_present = sorted({c.graph for c in required})
    for s in SIMS:
        for g in graphs_present:
            r = {c for c in required if c.sim == s and c.graph == g}
            p = {c for c in present if c.sim == s and c.graph == g and c in required}
            if not r:
                continue
            rows.append({
                "sim": s, "graph": g, "tier": graph_tier(g),
                "required": len(r), "present": len(p),
                "missing": len(r) - len(p),
                "coverage_pct": round(100.0 * len(p) / len(r), 1),
            })
    return rows


def render_missing_cells(required: set[CellKey], present: set[CellKey],
                          workstation_only: bool = False) -> list[dict]:
    rows = []
    for c in sorted(required - present,
                    key=lambda x: (x.sim, x.graph, x.app, x.l3_size, x.policy)):
        tier = graph_tier(c.graph)
        if workstation_only and tier not in ("LOCAL", "LOCAL_TIGHT"):
            continue
        rows.append({
            "sim": c.sim, "graph": c.graph, "app": c.app,
            "l3_size": c.l3_size, "policy": c.policy,
            "prefetcher": c.prefetcher, "workstation_tier": tier,
        })
    return rows


# ---------------------------------------------------------------------------
# Writers
# ---------------------------------------------------------------------------

def write_json(path: Path, *, scope: str, summary: dict,
               per_sim_graph: list[dict], missing: list[dict]) -> None:
    payload = {
        "gate": 282,
        "scope": scope,
        "status": "active",
        "summary": summary,
        "per_sim_graph": per_sim_graph,
        "missing_cells": missing,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


def write_md(path: Path, *, scope: str, summary: dict,
             per_sim_graph: list[dict], missing_workstation: list[dict]) -> None:
    t = summary["totals"]
    lines = []
    lines.append(f"# Headline coverage proof (gate 282)")
    lines.append("")
    lines.append(f"**Scope:** `{scope}` — literature-derived from "
                 f"`literature_baselines.{{INVARIANT,PER_GRAPH}}_CLAIMS`.")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- Required cells: **{t['required']}**")
    lines.append(f"- Present (in scope): **{t['present_in_scope']}**")
    lines.append(f"- Missing: **{t['missing']}**")
    lines.append(f"- Coverage: **{t['coverage_pct']:.1f} %**")
    lines.append(f"- Extra rows on disk (out of scope): {t['extra_out_of_scope']}")
    lines.append("")
    lines.append("## Per-simulator coverage")
    lines.append("")
    lines.append("| sim | present | required | coverage |")
    lines.append("|---|---:|---:|---:|")
    for s, d in summary["per_sim"].items():
        lines.append(f"| `{s}` | {d['present']} | {d['required']} | "
                     f"{d['coverage_pct']:.1f} % |")
    lines.append("")
    lines.append("## Missing cells by workstation tier")
    lines.append("")
    lines.append("| tier | missing |")
    lines.append("|---|---:|")
    for tier, n in sorted(summary["missing_by_workstation_tier"].items()):
        lines.append(f"| {tier} | {n} |")
    lines.append("")
    lines.append("## Per (simulator, graph) coverage")
    lines.append("")
    lines.append("| sim | graph | tier | present/req | coverage |")
    lines.append("|---|---|---|---:|---:|")
    for r in per_sim_graph:
        lines.append(f"| `{r['sim']}` | `{r['graph']}` | {r['tier']} | "
                     f"{r['present']}/{r['required']} | {r['coverage_pct']:.1f} % |")
    lines.append("")
    if missing_workstation:
        lines.append(f"## Workstation-runnable missing cells "
                     f"({len(missing_workstation)} cells)")
        lines.append("")
        lines.append("| sim | graph | app | L3 | policy | prefetcher | tier |")
        lines.append("|---|---|---|---|---|---|---|")
        for c in missing_workstation:
            lines.append(f"| {c['sim']} | {c['graph']} | {c['app']} | "
                         f"{c['l3_size']} | {c['policy']} | {c['prefetcher']} | "
                         f"{c['workstation_tier']} |")
        lines.append("")
    else:
        lines.append("## ✅ No workstation-runnable missing cells")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    # Strip trailing empty strings so the file ends with exactly one newline.
    while lines and lines[-1] == "":
        lines.pop()
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(path: Path, *, missing: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sim", "graph", "app", "l3_size", "policy",
                    "prefetcher", "workstation_tier", "status"])
        for c in missing:
            w.writerow([c["sim"], c["graph"], c["app"], c["l3_size"],
                        c["policy"], c["prefetcher"], c["workstation_tier"],
                        "missing"])


# ---------------------------------------------------------------------------
# Baseline ratchet
# ---------------------------------------------------------------------------

DEFAULT_BASELINE_PATH = (REPO_ROOT / "scripts/experiments/ecg"
                          / "headline_coverage_baseline.json")


def read_baseline(path: Path = DEFAULT_BASELINE_PATH) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_baseline(path: Path, *, scope: str, present_count: int,
                   per_sim: dict) -> None:
    payload = {
        "schema": 1,
        "scope": scope,
        "present_count": present_count,
        "per_sim_present_count": {s: per_sim[s]["present"] for s in per_sim},
        "note": ("Ratchet floor. Gate 282 asserts current coverage >= these "
                 "values per scope and per sim. Bump via "
                 "`python3 -m scripts.experiments.ecg.headline_coverage "
                 "--scope <scope> --bump-baseline`."),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--scope", choices=VALID_SCOPES, default=SCOPE_HEADLINE_1MB)
    p.add_argument("--cache-sim-csv", type=Path,
                   default=REPO_ROOT / "wiki/data/literature_faithfulness_postfix.csv")
    p.add_argument("--gem5-anchor", type=Path,
                   default=REPO_ROOT / "wiki/data/gem5_anchor.json")
    p.add_argument("--sniper-anchor", type=Path,
                   default=REPO_ROOT / "wiki/data/sniper_anchor.json")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "wiki/data/headline_coverage.json")
    p.add_argument("--md-out", type=Path,
                   default=REPO_ROOT / "wiki/data/headline_coverage.md")
    p.add_argument("--csv-out", type=Path,
                   default=REPO_ROOT / "wiki/data/headline_coverage.csv")
    p.add_argument("--baseline-path", type=Path,
                   default=DEFAULT_BASELINE_PATH)
    p.add_argument("--bump-baseline", action="store_true",
                   help="Overwrite the baseline file with current coverage.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress stdout summary; only write artifacts.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    present = load_all_presence(args.cache_sim_csv, args.gem5_anchor,
                                  args.sniper_anchor)
    required = required_cells(args.scope)
    summary = render_summary(required, present)
    per_sim_graph = render_per_sim_per_graph(required, present)
    missing = render_missing_cells(required, present, workstation_only=False)
    missing_ws = render_missing_cells(required, present, workstation_only=True)

    write_json(args.json_out, scope=args.scope, summary=summary,
               per_sim_graph=per_sim_graph, missing=missing)
    write_md(args.md_out, scope=args.scope, summary=summary,
             per_sim_graph=per_sim_graph, missing_workstation=missing_ws)
    write_csv(args.csv_out, missing=missing)

    if args.bump_baseline:
        write_baseline(args.baseline_path, scope=args.scope,
                       present_count=summary["totals"]["present_in_scope"],
                       per_sim=summary["per_sim"])

    if not args.quiet:
        t = summary["totals"]
        print(f"[headline-coverage] scope={args.scope} "
              f"present={t['present_in_scope']}/{t['required']} "
              f"({t['coverage_pct']:.1f}%) missing={t['missing']}")
        for s, d in summary["per_sim"].items():
            print(f"  {s:10s}  {d['present']:>3d} / {d['required']:>3d}  "
                  f"({d['coverage_pct']:5.1f} %)")
        baseline = read_baseline(args.baseline_path)
        if baseline:
            delta = t["present_in_scope"] - baseline.get("present_count", 0)
            tag = "ratchet floor" if delta == 0 else f"+{delta} above floor"
            print(f"  baseline={baseline.get('present_count', '?')} cells "
                  f"(scope={baseline.get('scope', '?')})  → {tag}")
        else:
            print(f"  no baseline at {args.baseline_path} — run "
                  f"--bump-baseline to create one")
    return 0


if __name__ == "__main__":
    sys.exit(main())
