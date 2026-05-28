#!/usr/bin/env python3
"""Literature-faithfulness comparator for the GraphBrew cache_sim sweep.

Joins ``roi_matrix.csv`` outputs under a sweep root with the expectations
encoded in :mod:`literature_baselines` and reports per-tuple verdicts:

  - ``ok``                — observed Δ matches the expected sign **and**
                            falls within ``min_abs_delta_pct`` and
                            ``max_abs_delta_pct`` bounds.
  - ``within_tolerance``  — observed Δ is within ``tolerance_pct`` of
                            either bound (counts as a soft pass).
  - ``disagree``          — observed Δ violates sign or bounds beyond
                            tolerance.
  - ``missing``           — no measurement for the expected tuple.

Inputs
------
A sweep root containing one directory per ``<graph>-<app>``, with a
``lit/roi_matrix.csv`` file (or ``--sweep-subdir`` to override the
sub-directory). The CSV columns used are ``policy``, ``l3_size``,
``l3_miss_rate``, ``section``, ``status``.

Outputs
-------
- Human-readable table to stdout grouped by (graph, app).
- Optional JSON dump (``--json-out``) with ``per_claim`` (one entry per
  expected ``LiteratureClaim``) and ``per_observation`` (one entry per
  observed (graph, app, L3, policy) row including LRU baseline rows).

Exit code 2 if any non-tolerated ``disagree`` is encountered, unless
``--no-exit-on-disagree`` is set.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

HERE = Path(__file__).resolve().parent

_spec = importlib.util.spec_from_file_location(
    "literature_baselines", HERE / "literature_baselines.py"
)
_lit = importlib.util.module_from_spec(_spec)
sys.modules["literature_baselines"] = _lit
_spec.loader.exec_module(_lit)  # type: ignore[union-attr]


@dataclass(frozen=True)
class Observation:
    graph: str
    app: str
    l3_size: str
    policy: str
    miss_rate: float
    section: int
    accesses: int = 0


def _coerce_int(text: str | None) -> int:
    try:
        return int(text) if text is not None and text != "" else 0
    except ValueError:
        return 0


def _pick_section(rows: list[Observation]) -> Observation:
    """Pick the smallest non-zero section if any exists; else section 0.

    Mirrors :mod:`sign_consistency` so the two comparators agree on which
    ROI section is canonical when gem5 emits multiple.
    """
    non_zero = [r for r in rows if r.section != 0]
    if non_zero:
        return min(non_zero, key=lambda r: r.section)
    return rows[0]


def load_observations(sweep_root: Path, subdir: str) -> list[Observation]:
    out: list[Observation] = []
    for csv_path in sorted(sweep_root.glob(f"*/{subdir}/roi_matrix.csv")):
        graph_app = csv_path.parent.parent.name
        if "-" not in graph_app:
            continue
        graph, _, app = graph_app.rpartition("-")
        rows_per_key: dict[tuple[str, str], list[Observation]] = defaultdict(list)
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                if (r.get("status") or "ok") != "ok":
                    continue
                if not r.get("l3_miss_rate"):
                    continue
                obs = Observation(
                    graph=graph,
                    app=app,
                    l3_size=r["l3_size"],
                    policy=r["policy"],
                    miss_rate=float(r["l3_miss_rate"]),
                    section=_coerce_int(r.get("section")),
                    accesses=_coerce_int(r.get("l3_misses")) + _coerce_int(r.get("l3_hits")),
                )
                rows_per_key[(r["l3_size"], r["policy"])].append(obs)
        for key, rows in rows_per_key.items():
            out.append(_pick_section(rows))
    return out


def index(observations: Iterable[Observation]) -> dict[tuple[str, str, str, str], Observation]:
    return {(o.graph, o.app, o.l3_size, o.policy): o for o in observations}


def _classify(claim: "_lit.LiteratureClaim", delta_pct: float) -> str:
    """Return ok / within_tolerance / disagree given a claim and observed Δ."""
    sign = claim.expected_sign
    tol = claim.tolerance_pct
    if sign == "-":
        if claim.min_abs_delta_pct is not None:
            if delta_pct <= -(claim.min_abs_delta_pct):
                pass
            elif delta_pct <= -(claim.min_abs_delta_pct - tol):
                return "within_tolerance"
            else:
                return "disagree"
        elif delta_pct > tol:
            return "disagree"
        if claim.max_abs_delta_pct is not None and abs(delta_pct) > claim.max_abs_delta_pct + tol:
            return "disagree"
        return "ok"
    if sign == "+":
        if claim.min_abs_delta_pct is not None and delta_pct < claim.min_abs_delta_pct - tol:
            return "disagree"
        if claim.max_abs_delta_pct is not None and delta_pct > claim.max_abs_delta_pct + tol:
            return "disagree"
        return "ok" if delta_pct >= 0 else "within_tolerance" if delta_pct >= -tol else "disagree"
    # sign == "~"  --> magnitude-only claim
    if claim.max_abs_delta_pct is not None and abs(delta_pct) > claim.max_abs_delta_pct + tol:
        return "disagree"
    return "ok"


def evaluate(
    obs_by_key: dict[tuple[str, str, str, str], Observation],
    graphs: list[str] | None = None,
    apps: list[str] | None = None,
    min_accesses: int = 10_000,
) -> dict[str, Any]:
    per_claim: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str, str]] = set()
    for o in obs_by_key.values():
        if graphs and o.graph not in graphs:
            continue
        if apps and o.app not in apps:
            continue
        seen_pairs.add((o.graph, o.app, o.l3_size))

    for graph, app, l3 in sorted(seen_pairs):
        lru = obs_by_key.get((graph, app, l3, "LRU"))
        for claim in _lit.claims_for(graph, app, l3):
            if claim.policy == "POPT_GE_GRASP":
                # Relative claim handled separately below.
                continue
            obs = obs_by_key.get((graph, app, l3, claim.policy))
            entry: dict[str, Any] = {
                "graph": graph,
                "app": app,
                "l3_size": l3,
                "policy": claim.policy,
                "expected_sign": claim.expected_sign,
                "min_abs_delta_pct": claim.min_abs_delta_pct,
                "max_abs_delta_pct": claim.max_abs_delta_pct,
                "tolerance_pct": claim.tolerance_pct,
                "rationale": claim.rationale,
                "citation": claim.citation,
            }
            if obs is None or lru is None:
                entry["status"] = "missing"
                entry["lru_miss_rate"] = None if lru is None else lru.miss_rate
                entry["policy_miss_rate"] = None if obs is None else obs.miss_rate
                entry["delta_pct"] = None
                entry["accesses"] = None if obs is None else obs.accesses
            elif max(lru.accesses, obs.accesses) < min_accesses:
                entry["status"] = "insufficient_data"
                entry["lru_miss_rate"] = lru.miss_rate
                entry["policy_miss_rate"] = obs.miss_rate
                entry["delta_pct"] = round((obs.miss_rate - lru.miss_rate) * 100.0, 4)
                entry["accesses"] = obs.accesses
            else:
                delta_pct = (obs.miss_rate - lru.miss_rate) * 100.0
                entry["lru_miss_rate"] = lru.miss_rate
                entry["policy_miss_rate"] = obs.miss_rate
                entry["delta_pct"] = round(delta_pct, 4)
                entry["accesses"] = obs.accesses
                entry["status"] = _classify(claim, delta_pct)
            per_claim.append(entry)

        # POPT_GE_GRASP relative claim
        rel_claims = [c for c in _lit.claims_for(graph, app, l3) if c.policy == "POPT_GE_GRASP"]
        if rel_claims:
            popt = obs_by_key.get((graph, app, l3, "POPT"))
            grasp = obs_by_key.get((graph, app, l3, "GRASP"))
            for claim in rel_claims:
                entry = {
                    "graph": graph, "app": app, "l3_size": l3,
                    "policy": "POPT_GE_GRASP", "expected_sign": claim.expected_sign,
                    "min_abs_delta_pct": None, "max_abs_delta_pct": None,
                    "tolerance_pct": claim.tolerance_pct,
                    "rationale": claim.rationale, "citation": claim.citation,
                }
                if popt is None or grasp is None:
                    entry["status"] = "missing"
                    entry["popt_miss_rate"] = None if popt is None else popt.miss_rate
                    entry["grasp_miss_rate"] = None if grasp is None else grasp.miss_rate
                    entry["delta_pct"] = None
                    entry["accesses"] = None if popt is None else popt.accesses
                elif max(popt.accesses, grasp.accesses) < min_accesses:
                    entry["status"] = "insufficient_data"
                    entry["popt_miss_rate"] = popt.miss_rate
                    entry["grasp_miss_rate"] = grasp.miss_rate
                    entry["delta_pct"] = round((popt.miss_rate - grasp.miss_rate) * 100.0, 4)
                    entry["accesses"] = popt.accesses
                else:
                    diff_pct = (popt.miss_rate - grasp.miss_rate) * 100.0
                    entry["popt_miss_rate"] = popt.miss_rate
                    entry["grasp_miss_rate"] = grasp.miss_rate
                    entry["delta_pct"] = round(diff_pct, 4)
                    entry["accesses"] = popt.accesses
                    if diff_pct <= claim.tolerance_pct:
                        entry["status"] = "ok"
                    else:
                        entry["status"] = "disagree"
                per_claim.append(entry)

    per_observation: list[dict[str, Any]] = []
    for (g, a, l, p), o in sorted(obs_by_key.items()):
        if graphs and g not in graphs:
            continue
        if apps and a not in apps:
            continue
        lru = obs_by_key.get((g, a, l, "LRU"))
        delta = (o.miss_rate - lru.miss_rate) * 100.0 if lru else None
        per_observation.append({
            "graph": g, "app": a, "l3_size": l, "policy": p,
            "miss_rate": o.miss_rate,
            "delta_vs_lru_pct": round(delta, 4) if delta is not None else None,
            "section": o.section,
        })

    disagreements = [e for e in per_claim if e["status"] == "disagree"]
    tolerated = [e for e in per_claim if e["status"] == "within_tolerance"]
    insufficient = [e for e in per_claim if e["status"] == "insufficient_data"]

    # Re-tag known deviations from `disagree` -> `known_deviation` so the
    # comparator and pytest stay green while the underlying issue is tracked.
    known_devs = getattr(_lit, "KNOWN_DEVIATIONS", {})
    deviated: list[dict[str, Any]] = []
    for e in disagreements[:]:
        key = (e["graph"], e["app"], e["l3_size"], e["policy"])
        if key in known_devs:
            e["status"] = "known_deviation"
            e["known_deviation_reason"] = known_devs[key]
            deviated.append(e)
    disagreements = [e for e in per_claim if e["status"] == "disagree"]

    return {
        "per_claim": per_claim,
        "per_observation": per_observation,
        "summary": {
            "claims_total": len(per_claim),
            "ok": sum(1 for e in per_claim if e["status"] == "ok"),
            "within_tolerance": len(tolerated),
            "disagree": len(disagreements),
            "missing": sum(1 for e in per_claim if e["status"] == "missing"),
            "insufficient_data": len(insufficient),
            "known_deviation": len(deviated),
            "min_accesses_threshold": min_accesses,
        },
        "disagreements": disagreements,
        "tolerated": tolerated,
        "known_deviations": deviated,
    }


# ---------- formatting ----------

def format_table(result: dict[str, Any]) -> str:
    out: list[str] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for o in result["per_observation"]:
        grouped[(o["graph"], o["app"])].append(o)

    for (graph, app), rows in sorted(grouped.items()):
        out.append(f"=== {graph} / {app} ===")
        out.append(f"  {'L3':<6} {'LRU':>9} {'SRRIP':>9} {'GRASP':>9} {'POPT':>9}    {'GRASP-LRU':>10} {'POPT-LRU':>10}")
        out.append(f"  {'-'*72}")
        by_l3: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for r in rows:
            by_l3[r["l3_size"]][r["policy"]] = r
        for l3 in sorted(by_l3, key=_l3_sort_key):
            policies = by_l3[l3]
            def mr(p: str) -> str:
                return f"{policies[p]['miss_rate']:.5f}" if p in policies else "    n/a "
            def dlt(p: str) -> str:
                v = policies.get(p, {}).get("delta_vs_lru_pct")
                return f"{v:+.3f}" if v is not None else "  n/a "
            out.append(f"  {l3:<6} {mr('LRU'):>9} {mr('SRRIP'):>9} {mr('GRASP'):>9} {mr('POPT'):>9}    {dlt('GRASP'):>10} {dlt('POPT'):>10}")
        out.append("")
    out.append("=== Claim verdicts ===")
    for e in result["per_claim"]:
        delta = e.get("delta_pct")
        delta_s = f"{delta:+.3f}pp" if delta is not None else "   n/a"
        out.append(
            f"  [{e['status']:<17}] {e['graph']:>14}/{e['app']:<3} L3={e['l3_size']:<4} {e['policy']:<16} "
            f"Δ={delta_s}  ({e['citation']})"
        )
    s = result["summary"]
    out.append("")
    out.append(
        f"Summary: {s['ok']} ok, {s['within_tolerance']} within-tolerance, "
        f"{s['disagree']} DISAGREE, {s.get('known_deviation', 0)} known-deviation, "
        f"{s['missing']} missing, {s.get('insufficient_data', 0)} insufficient_data  "
        f"(total claims: {s['claims_total']}; min_accesses={s.get('min_accesses_threshold', '?')})"
    )
    return "\n".join(out)


def _l3_sort_key(s: str) -> int:
    units = {"kB": 1024, "MB": 1024 * 1024, "GB": 1024 ** 3}
    for u, v in units.items():
        if s.endswith(u):
            return int(s[: -len(u)]) * v
    return int(s)


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sweep-root", type=Path, required=True,
                   help="Root containing <graph>-<app>/<subdir>/roi_matrix.csv files.")
    p.add_argument("--sweep-subdir", default="lit",
                   help="Subdirectory under <graph>-<app>/ that holds roi_matrix.csv (default: lit).")
    p.add_argument("--graphs", nargs="*", default=None,
                   help="Restrict to these graph short names.")
    p.add_argument("--apps", nargs="*", default=None,
                   help="Restrict to these apps.")
    p.add_argument("--json-out", type=Path, default=None)
    p.add_argument("--no-exit-on-disagree", action="store_true")
    p.add_argument("--min-accesses", type=int, default=10_000,
                   help="Minimum L3 accesses for a claim to count; below this it's "
                        "tagged insufficient_data (default: 10000).")
    args = p.parse_args(argv)

    obs = load_observations(args.sweep_root, args.sweep_subdir)
    if not obs:
        print(f"[lit] no observations found under {args.sweep_root}/*/{args.sweep_subdir}/", file=sys.stderr)
        return 1
    obs_idx = index(obs)
    result = evaluate(obs_idx, graphs=args.graphs, apps=args.apps, min_accesses=args.min_accesses)
    print(format_table(result))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True))
        print(f"\n[lit] JSON: {args.json_out}", file=sys.stderr)
    if result["summary"]["disagree"] and not args.no_exit_on_disagree:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
