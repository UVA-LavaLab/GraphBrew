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


# Canonical literature-comparator policy roster. Restricts the
# comparator output to the four baseline policies that downstream
# cross-tool parity gates assume. ECG variants (POPT_CHARGED,
# ECG_DBG_PRIMARY, etc.) live in the same on-disk roi_matrix.csv
# but are siphoned off here into a SEPARATE companion artifact so
# the cross-tool gates that consume literature_faithfulness_postfix.*
# see a stable shape (4 policies x N L3 x N (graph, app) cells).
CANONICAL_POLICY_ROSTER = ("LRU", "SRRIP", "GRASP", "POPT")


def load_observations(sweep_root: Path, subdir: str,
                       policy_filter: set[str] | None = None) -> list[Observation]:
    """Load per-cell observations from a sweep root.

    Args:
      sweep_root: directory containing <graph>-<app>/<subdir>/roi_matrix.csv
      subdir:    typically 'lit'
      policy_filter: if provided, restrict to policy_label values in
                     this set. If None, return ALL policies (used by
                     the ECG-extension companion).
    """
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
                # Prefer policy_label (specific variant, e.g. ECG_DBG_PRIMARY)
                # over policy (family, e.g. ECG). For non-ECG rows the two
                # are identical so existing LRU/SRRIP/GRASP/POPT behavior
                # is preserved.
                pol = r.get("policy_label") or r["policy"]
                if policy_filter is not None and pol not in policy_filter:
                    continue
                obs = Observation(
                    graph=graph,
                    app=app,
                    l3_size=r["l3_size"],
                    policy=pol,
                    miss_rate=float(r["l3_miss_rate"]),
                    section=_coerce_int(r.get("section")),
                    accesses=_coerce_int(r.get("l3_misses")) + _coerce_int(r.get("l3_hits")),
                )
                rows_per_key[(r["l3_size"], pol)].append(obs)
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
            if claim.policy == "POPT_NEAR_GRASP_IF_BIG_GAP":
                # Cross-policy invariant handled separately below.
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

        # POPT_NEAR_GRASP_IF_BIG_GAP cross-policy invariant. Fires only when
        # GRASP-LRU shows a >10pp improvement (phase-transition regime); in
        # that regime POPT must agree with GRASP within tolerance, otherwise
        # one of the two policies is misbehaving.
        near_claims = [c for c in _lit.claims_for(graph, app, l3) if c.policy == "POPT_NEAR_GRASP_IF_BIG_GAP"]
        if near_claims:
            popt = obs_by_key.get((graph, app, l3, "POPT"))
            grasp = obs_by_key.get((graph, app, l3, "GRASP"))
            lru_ref = obs_by_key.get((graph, app, l3, "LRU"))
            for claim in near_claims:
                entry = {
                    "graph": graph, "app": app, "l3_size": l3,
                    "policy": "POPT_NEAR_GRASP_IF_BIG_GAP",
                    "expected_sign": claim.expected_sign,
                    "min_abs_delta_pct": None,
                    "max_abs_delta_pct": claim.max_abs_delta_pct,
                    "tolerance_pct": claim.tolerance_pct,
                    "rationale": claim.rationale, "citation": claim.citation,
                }
                if popt is None or grasp is None or lru_ref is None:
                    entry["status"] = "missing"
                    entry["popt_miss_rate"] = None if popt is None else popt.miss_rate
                    entry["grasp_miss_rate"] = None if grasp is None else grasp.miss_rate
                    entry["delta_pct"] = None
                    entry["accesses"] = None
                elif max(popt.accesses, grasp.accesses, lru_ref.accesses) < min_accesses:
                    entry["status"] = "insufficient_data"
                    entry["popt_miss_rate"] = popt.miss_rate
                    entry["grasp_miss_rate"] = grasp.miss_rate
                    entry["delta_pct"] = round((popt.miss_rate - grasp.miss_rate) * 100.0, 4)
                    entry["accesses"] = popt.accesses
                else:
                    grasp_gain_pp = (lru_ref.miss_rate - grasp.miss_rate) * 100.0
                    signed_pp = (popt.miss_rate - grasp.miss_rate) * 100.0
                    diff_pct = abs(signed_pp)
                    entry["popt_miss_rate"] = popt.miss_rate
                    entry["grasp_miss_rate"] = grasp.miss_rate
                    entry["delta_pct"] = round(diff_pct, 4)
                    entry["signed_delta_pct"] = round(signed_pp, 4)
                    entry["grasp_gain_vs_lru_pct"] = round(grasp_gain_pp, 4)
                    entry["accesses"] = popt.accesses
                    # Trigger threshold: only assert when GRASP outperforms
                    # LRU by >10pp (the phase-transition regime).
                    if grasp_gain_pp <= 10.0:
                        entry["status"] = "ok"
                        entry["note"] = "not in phase-transition regime; assertion not triggered"
                    elif signed_pp <= claim.max_abs_delta_pct + claim.tolerance_pct:
                        # POPT either better than GRASP (signed_pp <= 0) or
                        # within tolerance worse - both are literature-faithful.
                        entry["status"] = "ok"
                        if signed_pp < 0:
                            entry["note"] = "POPT outperforms GRASP (oracle dominates heuristic)"
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


# ---------- summary emitters ----------

def format_markdown(result: dict[str, Any], sweep_root: Path) -> str:
    """Render a self-contained Markdown report for the audit doc."""
    s = result["summary"]
    out: list[str] = []
    out.append(f"# Literature-faithfulness summary")
    out.append("")
    out.append(f"- Sweep root: `{sweep_root}`")
    out.append(f"- Claims total: **{s['claims_total']}**")
    out.append(
        f"- Verdict mix: **{s['ok']} ok**, {s['within_tolerance']} within-tolerance, "
        f"**{s['disagree']} DISAGREE**, {s.get('known_deviation', 0)} known-deviation, "
        f"{s['missing']} missing, {s.get('insufficient_data', 0)} insufficient_data"
    )
    out.append(f"- min_accesses threshold: {s.get('min_accesses_threshold', '?')}")
    out.append("")

    out.append("## Observed L3 miss-rates")
    out.append("")
    out.append("| graph | app | L3 | LRU | SRRIP | GRASP | POPT | Δ GRASP-LRU | Δ POPT-LRU |")
    out.append("|---|---|---|---:|---:|---:|---:|---:|---:|")
    grouped: dict[tuple[str, str], dict[str, dict[str, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for o in result["per_observation"]:
        grouped[(o["graph"], o["app"])][o["l3_size"]][o["policy"]] = o
    for (graph, app), by_l3 in sorted(grouped.items()):
        for l3 in sorted(by_l3, key=_l3_sort_key):
            pmap = by_l3[l3]
            def fmt(p: str) -> str:
                return f"{pmap[p]['miss_rate']:.4f}" if p in pmap else "—"
            def fmt_d(p: str) -> str:
                v = pmap.get(p, {}).get("delta_vs_lru_pct")
                return f"{v:+.3f}pp" if v is not None else "—"
            out.append(
                f"| {graph} | {app} | {l3} | {fmt('LRU')} | {fmt('SRRIP')} | "
                f"{fmt('GRASP')} | {fmt('POPT')} | {fmt_d('GRASP')} | {fmt_d('POPT')} |"
            )
    out.append("")

    out.append("## Per-claim verdicts")
    out.append("")
    out.append("| status | graph | app | L3 | policy | Δ | citation |")
    out.append("|---|---|---|---|---|---:|---|")
    for e in result["per_claim"]:
        delta = e.get("delta_pct")
        delta_s = f"{delta:+.3f}pp" if delta is not None else "—"
        out.append(
            f"| {e['status']} | {e['graph']} | {e['app']} | {e['l3_size']} | "
            f"{e['policy']} | {delta_s} | {e['citation']} |"
        )
    out.append("")

    if result.get("known_deviations"):
        out.append("## Known deviations (registered)")
        out.append("")
        for d in result["known_deviations"]:
            reason = d.get("known_deviation_reason") or d.get("reason") or ""
            line = (
                f"- **{d['graph']}/{d['app']} L3={d['l3_size']} {d['policy']}** "
                f"({d.get('delta_pct', 0):+.3f}pp)"
            )
            if reason:
                line += f": {reason}"
            out.append(line)
        out.append("")

    if result.get("disagreements"):
        out.append("## ⚠ Disagreements (need investigation)")
        out.append("")
        for d in result["disagreements"]:
            out.append(
                f"- **{d['graph']}/{d['app']} L3={d['l3_size']} {d['policy']}** "
                f"Δ={d.get('delta_pct', 0):+.3f}pp — {d['citation']}"
            )
        out.append("")
    return "\n".join(out)


def format_summary_csv(result: dict[str, Any]) -> str:
    """One row per (graph, app, L3, policy) with miss-rate + Δ + verdict."""
    rows: list[dict[str, Any]] = []
    claim_lookup: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for e in result["per_claim"]:
        claim_lookup[(e["graph"], e["app"], e["l3_size"], e["policy"])] = e
    for o in result["per_observation"]:
        c = claim_lookup.get((o["graph"], o["app"], o["l3_size"], o["policy"]))
        rows.append({
            "graph": o["graph"],
            "app": o["app"],
            "l3_size": o["l3_size"],
            "policy": o["policy"],
            "miss_rate": f"{o['miss_rate']:.6f}",
            "delta_vs_lru_pct": (
                f"{o['delta_vs_lru_pct']:+.4f}" if o.get("delta_vs_lru_pct") is not None else ""
            ),
            "l3_accesses": o.get("l3_accesses", ""),
            "claim_status": c["status"] if c else "no_claim",
            "claim_citation": c["citation"] if c else "",
        })
    if not rows:
        return ""
    fields = ["graph", "app", "l3_size", "policy", "miss_rate",
              "delta_vs_lru_pct", "l3_accesses", "claim_status", "claim_citation"]
    import io as _io
    buf = _io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fields, lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow(r)
    return buf.getvalue()


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
    p.add_argument("--md-out", type=Path, default=None,
                   help="Write a Markdown summary suitable for paste into the audit doc.")
    p.add_argument("--csv-out", type=Path, default=None,
                   help="Write one CSV row per (graph,app,L3,policy) with miss-rate + verdict.")
    p.add_argument("--no-exit-on-disagree", action="store_true")
    p.add_argument("--min-accesses", type=int, default=10_000,
                   help="Minimum L3 accesses for a claim to count; below this it's "
                        "tagged insufficient_data (default: 10000).")
    args = p.parse_args(argv)

    obs = load_observations(args.sweep_root, args.sweep_subdir,
                              policy_filter=set(CANONICAL_POLICY_ROSTER))
    if not obs:
        print(f"[lit] no observations found under {args.sweep_root}/*/{args.sweep_subdir}/", file=sys.stderr)
        return 1
    obs_idx = index(obs)
    result = evaluate(obs_idx, graphs=args.graphs, apps=args.apps, min_accesses=args.min_accesses)
    print(format_table(result))
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"\n[lit] JSON: {args.json_out}", file=sys.stderr)
    if args.md_out:
        args.md_out.parent.mkdir(parents=True, exist_ok=True)
        args.md_out.write_text(format_markdown(result, args.sweep_root))
        print(f"[lit] Markdown: {args.md_out}", file=sys.stderr)
    if args.csv_out:
        args.csv_out.parent.mkdir(parents=True, exist_ok=True)
        args.csv_out.write_text(format_summary_csv(result))
        print(f"[lit] CSV: {args.csv_out}", file=sys.stderr)
    if result["summary"]["disagree"] and not args.no_exit_on_disagree:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
