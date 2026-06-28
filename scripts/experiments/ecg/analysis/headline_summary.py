#!/usr/bin/env python3
"""ECG headline WIN/TIE/LOSE summary over the eviction_matrix tables.

Consumes one or more `eviction_matrix` markdown tables (the SSOT ECG-variant
matrix produced by scripts/experiments/ecg/flows/eviction_matrix.py: columns
LRU | GRASP | POPT | ECG:grasp_only | ECG:epoch_only | ECG:rrip_first |
ECG:epoch_first | ECG:shortcirc; rows = graph/L3/order). It does NOT run any
simulation — it only distills already-measured cells into the paper headline.

For every row it picks the BEST ECG variant (lowest L3 miss rate; lower=better)
and renders the verdict of that best-ECG against each baseline:
  WIN  best_ecg is lower than the baseline by more than --eps
  TIE  within +/- --eps
  LOSE best_ecg is higher than the baseline by more than --eps
The headline claim is "ECG is at least as good as the strongest static baseline"
i.e. best_ecg <= min(GRASP, POPT). We report vs POPT (the P-OPT policy, usually
the strongest), vs GRASP (the degree-only baseline), vs LRU (the do-no-harm
floor), and the combined verdict vs min(GRASP, POPT).

Kernel is taken from the table title `[suite/kernel]` if present, else from the
filename (eviction_matrix_<kernel>.md), else "pr".

Usage:
  python3 scripts/experiments/ecg/analysis/headline_summary.py \
      results/ecg_experiments/m3_local_fit/eviction_matrix_local.md \
      results/ecg_experiments/c3_multikernel/eviction_matrix_*.md \
      --out results/ecg_experiments/c3_multikernel/headline_summary.md
"""
from __future__ import annotations
import argparse
import glob
import re
import sys
from pathlib import Path

ECG_COLS_ORDER = [
    "ECG:grasp_only", "ECG:epoch_only", "ECG:rrip_first",
    "ECG:epoch_first", "ECG:shortcirc",
]
BASELINES = ["LRU", "GRASP", "POPT"]


KNOWN_KERNELS = {"pr", "pr_spmv", "bfs", "bc", "cc", "cc_sv", "sssp", "tc"}


def _kernel_for(path: Path, title_kernel: str | None) -> str:
    if title_kernel in KNOWN_KERNELS:
        return title_kernel
    m = re.search(r"eviction_matrix_([a-z_]+)\.md$", path.name)
    if m and m.group(1) in KNOWN_KERNELS:
        return m.group(1)
    return "pr"


def parse_md_table(path: Path):
    """Parse one eviction_matrix markdown table -> (kernel, [row dicts])."""
    text = path.read_text()
    title_kernel = None
    mt = re.search(r"\[[a-z0-9\-]+/([a-z_]+)\]", text)
    if mt:
        title_kernel = mt.group(1)
    kernel = _kernel_for(path, title_kernel)

    header = None
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            header = cells
            continue
        if set("".join(cells)) <= set("-: "):  # separator row
            continue
        if len(cells) != len(header):
            continue
        rowmap = dict(zip(header, cells))
        tag = rowmap.get(header[0], "")
        # tag like "web-Google/512kB/o5"
        parts = tag.split("/")
        if len(parts) != 3:
            continue
        graph, l3, order = parts
        vals = {}
        for col in header[1:]:
            v = rowmap.get(col, "--")
            try:
                vals[col] = float(v)
            except (TypeError, ValueError):
                vals[col] = None
        rows.append({"kernel": kernel, "graph": graph, "l3": l3,
                     "order": order, "vals": vals})
    return kernel, rows


def best_ecg(vals: dict):
    """Return (label, value) of the lowest ECG:* column present."""
    cand = [(c, vals.get(c)) for c in ECG_COLS_ORDER if vals.get(c) is not None]
    if not cand:
        # fall back to any ECG-prefixed column in the table
        cand = [(c, v) for c, v in vals.items()
                if c.startswith("ECG:") and v is not None]
    if not cand:
        return None, None
    return min(cand, key=lambda kv: kv[1])


def verdict(best: float, base: float, eps: float) -> str:
    if base is None or best is None:
        return "n/a"
    if best < base - eps:
        return "WIN"
    if best > base + eps:
        return "LOSE"
    return "TIE"


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tables", nargs="+",
                    help="eviction_matrix .md files (globs allowed).")
    ap.add_argument("--eps", type=float, default=0.005,
                    help="TIE band on l3_miss_rate (default 0.005 = 0.5pp).")
    ap.add_argument("--orders", default="5",
                    help="Space/comma list of orders to score for the headline "
                         "(default '5'; the DBG reorder GRASP/ECG need). Use '0 5' "
                         "to include both. Counts are reported per order regardless.")
    ap.add_argument("--out", default="", help="Write the markdown summary here.")
    args = ap.parse_args(argv)

    paths = []
    for t in args.tables:
        paths.extend(sorted(glob.glob(t)) or [t])
    paths = [Path(p) for p in paths if Path(p).exists()]
    if not paths:
        print("no input tables found", file=sys.stderr)
        return 2

    rows = []
    for p in paths:
        _, rws = parse_md_table(p)
        rows.extend(rws)
    if not rows:
        print("no rows parsed from input tables", file=sys.stderr)
        return 2

    headline_orders = {o.lstrip("o") for o in re.split(r"[ ,]+", args.orders) if o}

    md = []
    md.append("# ECG headline WIN/TIE/LOSE summary (cache_sim, l3_miss_rate, lower=better)")
    md.append("")
    md.append(f"- best ECG = lowest of {{{', '.join(ECG_COLS_ORDER)}}} per cell; "
              f"TIE band eps={args.eps} ({args.eps*100:.1f}pp).")
    md.append("- verdicts are best-ECG vs each baseline; `vs min(G,P)` is the headline "
              "(ECG >= strongest static baseline).")
    md.append("")
    md.append("| kernel | graph | L3 | o | best ECG (var) | LRU | GRASP | POPT | "
              "vs LRU | vs GRASP | vs POPT | vs min(G,P) |")
    md.append("|" + "---|" * 12)

    # tally only the headline order(s)
    tally = {}  # (kernel, key) -> dict of counts
    for r in rows:
        v = r["vals"]
        be_lbl, be_val = best_ecg(v)
        lru, grasp, popt = v.get("LRU"), v.get("GRASP"), v.get("POPT")
        gp = None
        if grasp is not None and popt is not None:
            gp = min(grasp, popt)
        elif popt is not None:
            gp = popt
        elif grasp is not None:
            gp = grasp
        vd_lru = verdict(be_val, lru, args.eps)
        vd_grasp = verdict(be_val, grasp, args.eps)
        vd_popt = verdict(be_val, popt, args.eps)
        vd_gp = verdict(be_val, gp, args.eps)
        be_short = (be_lbl or "--").replace("ECG:", "")
        def f(x):
            return f"{x:.4f}" if isinstance(x, float) else "--"
        md.append(f"| {r['kernel']} | {r['graph']} | {r['l3']} | {r['order'].lstrip('o')} | "
                  f"{f(be_val)} ({be_short}) | {f(lru)} | {f(grasp)} | {f(popt)} | "
                  f"{vd_lru} | {vd_grasp} | {vd_popt} | **{vd_gp}** |")
        o = r["order"].lstrip("o")
        if o in headline_orders:
            for key, vd in (("vs_LRU", vd_lru), ("vs_GRASP", vd_grasp),
                            ("vs_POPT", vd_popt), ("vs_minGP", vd_gp)):
                d = tally.setdefault((r["kernel"], key),
                                     {"WIN": 0, "TIE": 0, "LOSE": 0, "n/a": 0})
                d[vd] = d.get(vd, 0) + 1

    md.append("")
    md.append(f"## Headline tally (order(s) {sorted(headline_orders)} only)")
    md.append("")
    md.append("| kernel | comparison | WIN | TIE | LOSE |")
    md.append("|---|---|---|---|---|")
    kernels = sorted({k for (k, _) in tally})
    for k in kernels:
        for cmp in ("vs_minGP", "vs_POPT", "vs_GRASP", "vs_LRU"):
            d = tally.get((k, cmp))
            if not d:
                continue
            md.append(f"| {k} | {cmp} | {d['WIN']} | {d['TIE']} | {d['LOSE']} |")
    # grand total vs min(G,P)
    tot = {"WIN": 0, "TIE": 0, "LOSE": 0, "n/a": 0}
    for (k, cmp), d in tally.items():
        if cmp == "vs_minGP":
            for kk in tot:
                tot[kk] += d.get(kk, 0)
    md.append(f"| **ALL** | **vs_minGP** | **{tot['WIN']}** | **{tot['TIE']}** | "
              f"**{tot['LOSE']}** |")

    out_text = "\n".join(md) + "\n"
    print(out_text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(out_text)
        print(f"[written] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
