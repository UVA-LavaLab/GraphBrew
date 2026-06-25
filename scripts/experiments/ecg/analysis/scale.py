#!/usr/bin/env python3
"""Aggregate + interpret the preliminary scale sweep (Sprint 3 analysis).

Reads results/ecg_experiments/prelim_scale/prelim_scale.csv (works on a partial
in-progress file) and emits the honest comparison the rubber-duck mandated:
  * per (graph, L3) DEMAND miss-rate with pressure ratio (prop/L3) + per-cell
    winner + Delta(ECG-GRASP) + Delta(ECG-chargedPOPT) + P-OPT reserved ways/feasibility
  * PROPERTY miss-rate table (the replacement-policy-governed metric)
  * prefetcher ON vs OFF (main vs deg0): demand mr + total traffic, to separate
    a real replacement win from structure-stream hiding
  * total memory traffic (the ECG 8B-record bandwidth cost)
  * pressure-bucket win/loss/tie summary for ECG vs GRASP and vs charged P-OPT

Usage: python3 scripts/experiments/ecg/analyze_prelim_scale.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
CSV_PATH = ROOT / "results" / "ecg_experiments" / "prelim_scale" / "prelim_scale.csv"
MD_PATH = ROOT / "results" / "ecg_experiments" / "prelim_scale" / "prelim_scale_summary.md"

GRAPH_ORDER = ["web-Google", "roadNet-CA", "soc-pokec", "cit-Patents",
               "com-orkut", "soc-LiveJournal1", "kron-s24"]
COL_ORDER = ["LRU", "GRASP", "POPT", "POPT_UNCH", "ECG", "ECG_sc"]


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load():
    if not CSV_PATH.exists():
        raise SystemExit(f"no CSV yet at {CSV_PATH}")
    return list(csv.DictReader(open(CSV_PATH)))


def pivot(rows, block, value_key):
    """(graph,l3) -> {column: value} for one block."""
    t = defaultdict(dict)
    meta = {}
    for r in rows:
        if r["block"] != block or r["status"] != "ok":
            continue
        key = (r["graph"], int(r["l3_mb"]))
        t[key][r["column"]] = f(r.get(value_key))
        meta[key] = (f(r["prop_mb"]), r.get("popt_reserved_ways"),
                     r.get("popt_matrix_fits"))
    return t, meta


def gkey(k):
    g, l3 = k
    return (GRAPH_ORDER.index(g) if g in GRAPH_ORDER else 99, l3)


def fmt(v, w=7):
    return (f"{v:.4f}" if isinstance(v, float) else "  --  ").rjust(w)


def winner(cell):
    cand = {c: cell[c] for c in ("LRU", "GRASP", "POPT", "ECG")
            if isinstance(cell.get(c), float)}
    return min(cand, key=cand.get) if cand else "?"


def section_demand(rows, out):
    t, meta = pivot(rows, "main", "l3_miss_rate")
    out.append("## Demand L3 miss-rate (prefetcher ON, -o5, 8B full epoch)\n")
    out.append("pressure = property(4*|V|) / L3.  rW=charged-POPT reserved ways; "
               "fit=matrix feasible.  dG=ECG-GRASP, dP=ECG-chargedPOPT (negative=ECG better).\n")
    out.append("| graph | prop | L3 | press | LRU | GRASP | POPT(ch) | POPT(un) | ECG | "
               "win | dG | dP | rW/fit |")
    out.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|")
    for k in sorted(t, key=gkey):
        c = t[k]
        prop, rw, fit = meta[k]
        press = prop / k[1] if prop else 0
        dG = (c["ECG"] - c["GRASP"]) if isinstance(c.get("ECG"), float) and isinstance(c.get("GRASP"), float) else None
        dP = (c["ECG"] - c["POPT"]) if isinstance(c.get("ECG"), float) and isinstance(c.get("POPT"), float) else None
        out.append(f"| {k[0]} | {prop:.1f} | {k[1]} | {press:.2f}x | "
                   f"{fmt(c.get('LRU'))} | {fmt(c.get('GRASP'))} | {fmt(c.get('POPT'))} | "
                   f"{fmt(c.get('POPT_UNCH'))} | {fmt(c.get('ECG'))} | {winner(c)} | "
                   f"{(f'{dG:+.4f}' if dG is not None else '--')} | "
                   f"{(f'{dP:+.4f}' if dP is not None else '--')} | "
                   f"{rw}/{fit} |")
    out.append("")


def section_property(rows, out):
    t, meta = pivot(rows, "main", "l3_prop_miss_rate")
    out.append("## Property L3 miss-rate (the replacement-governed metric)\n")
    out.append("| graph | L3 | press | LRU | GRASP | POPT(ch) | POPT(un) | ECG | win | dG | dP |")
    out.append("|---|--:|--:|--:|--:|--:|--:|--:|---|--:|--:|")
    for k in sorted(t, key=gkey):
        c = t[k]
        prop = meta[k][0]
        press = prop / k[1] if prop else 0
        dG = (c["ECG"] - c["GRASP"]) if isinstance(c.get("ECG"), float) and isinstance(c.get("GRASP"), float) else None
        dP = (c["ECG"] - c["POPT"]) if isinstance(c.get("ECG"), float) and isinstance(c.get("POPT"), float) else None
        out.append(f"| {k[0]} | {k[1]} | {press:.2f}x | {fmt(c.get('LRU'))} | "
                   f"{fmt(c.get('GRASP'))} | {fmt(c.get('POPT'))} | {fmt(c.get('POPT_UNCH'))} | "
                   f"{fmt(c.get('ECG'))} | {winner(c)} | "
                   f"{(f'{dG:+.4f}' if dG is not None else '--')} | "
                   f"{(f'{dP:+.4f}' if dP is not None else '--')} |")
    out.append("")


def section_prefetcher(rows, out):
    on, _ = pivot(rows, "main", "l3_miss_rate")
    off, _ = pivot(rows, "deg0", "l3_miss_rate")
    on_tr, _ = pivot(rows, "main", "total_memory_traffic")
    off_tr, _ = pivot(rows, "deg0", "total_memory_traffic")
    out.append("## Prefetcher ON vs OFF (demand mr; traffic in M lines)\n")
    out.append("Separates a real replacement win (OFF) from structure-stream hiding (ON).\n")
    out.append("| graph | L3 | pol | mr_off | mr_on | traffic_off | traffic_on |")
    out.append("|---|--:|---|--:|--:|--:|--:|")
    for k in sorted(off, key=gkey):
        for col in ("LRU", "GRASP", "POPT", "ECG"):
            mo, mn = off[k].get(col), on.get(k, {}).get(col)
            to = off_tr.get(k, {}).get(col)
            tn = on_tr.get(k, {}).get(col)
            out.append(f"| {k[0]} | {k[1]} | {col} | {fmt(mo)} | {fmt(mn)} | "
                       f"{(f'{to/1e6:.2f}' if isinstance(to,float) else '--')} | "
                       f"{(f'{tn/1e6:.2f}' if isinstance(tn,float) else '--')} |")
    out.append("")


def section_traffic(rows, out):
    t, _ = pivot(rows, "main", "total_memory_traffic")
    out.append("## Total memory traffic, main grid (M lines; ECG carries the 8B record)\n")
    out.append("| graph | L3 | LRU | GRASP | POPT(ch) | ECG |")
    out.append("|---|--:|--:|--:|--:|--:|")
    for k in sorted(t, key=gkey):
        c = t[k]
        def m(x):
            return f"{c[x]/1e6:.2f}" if isinstance(c.get(x), float) else "--"
        out.append(f"| {k[0]} | {k[1]} | {m('LRU')} | {m('GRASP')} | {m('POPT')} | {m('ECG')} |")
    out.append("")


def section_summary(rows, out):
    t, meta = pivot(rows, "main", "l3_miss_rate")
    tp, _ = pivot(rows, "main", "l3_prop_miss_rate")
    buckets = {"high(>=2x)": [], "mid(1-2x)": [], "low(<1x)": []}
    for k in t:
        prop = meta[k][0]
        press = prop / k[1] if prop else 0
        b = "high(>=2x)" if press >= 2 else ("mid(1-2x)" if press >= 1 else "low(<1x)")
        buckets[b].append(k)
    out.append("## ECG win/loss vs GRASP and vs charged-POPT, by pressure bucket (demand mr)\n")
    out.append("| bucket | cells | ECG<GRASP | ECG<POPT(ch) | ECG<both |")
    out.append("|---|--:|--:|--:|--:|")
    for b, ks in buckets.items():
        n = wg = wp = wb = 0
        for k in ks:
            c = t[k]
            if not (isinstance(c.get("ECG"), float)):
                continue
            n += 1
            bg = isinstance(c.get("GRASP"), float) and c["ECG"] < c["GRASP"] - 1e-9
            bp = isinstance(c.get("POPT"), float) and c["ECG"] < c["POPT"] - 1e-9
            wg += bg
            wp += bp
            wb += bg and bp
        out.append(f"| {b} | {n} | {wg} | {wp} | {wb} |")
    out.append("")


def main():
    rows = load()
    ok = sum(r["status"] == "ok" for r in rows)
    out = [f"# Preliminary scale sweep summary ({ok}/{len(rows)} cells ok)\n"]
    section_demand(rows, out)
    section_property(rows, out)
    section_summary(rows, out)
    section_prefetcher(rows, out)
    section_traffic(rows, out)
    text = "\n".join(out)
    MD_PATH.write_text(text)
    print(text)
    print(f"\n[written {MD_PATH}]")


if __name__ == "__main__":
    main()
