#!/usr/bin/env python3
"""Translation-pressure proxy: DROPLET (all-K) vs ECG_PFX (best-1) at PAGE granularity.

Raw prefetch count is not a faithful TLB-pressure metric (a 4KB page holds ~1024
4B properties). This measures, per prefetcher, the DISTINCT pages its property
prefetch targets touch and the misses of a finite LRU MTLB (DROPLET's design uses
a dedicated memory-controller MTLB; P-OPT uses a 1GB huge page). It tests the
"conservative for ECG_PFX" claim raised in the TLB analysis rubber-duck:

  - distinct_4k / distinct_2m pages: with an infinite TLB or huge pages, do the two
    prefetchers differ? (Both sweep the property array, so expect ~EQUAL.)
  - mtlb_misses at a small vs large MTLB: when the page working set >> MTLB, misses
    track the request count (ECG_PFX ~K x fewer); when MTLB covers the working set,
    both -> ~0 (translation is a non-issue for either).

cache_sim only (it is the authoritative prefetch model; gem5 SE-mode does not model
the TLB, dtb.accesses=0). Uses CACHE_PFX_MTLB_ENTRIES to size the modelled MTLB.

Usage:
  python3 scripts/experiments/ecg/pfx_tlb_pressure.py \
      --cells "web-Google:512kB soc-pokec:1MB com-orkut:2MB" \
      --mtlb "128 8192" --out /tmp/pfx_tlb.md
"""
import argparse, json, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GRAPHS = ROOT / "results" / "graphs"
PREFETCHERS = ["DROPLET", "ECG_PFX"]


def run(graph, l3, pf, mtlb, lookahead):
    gpath = GRAPHS / graph / f"{graph}.sg"
    outdir = Path("/tmp") / f"pfxtlb_{graph}_{l3}_{pf}_m{mtlb}"
    cmd = [sys.executable, str(ROOT / "scripts/experiments/ecg/roi_matrix.py"),
           "--suite", "cache-sim", "--no-build", "--benchmark", "pr",
           "--policies", "LRU", "--options", f"-f {gpath} -o 0 -n 1 -i 1",
           "--l3-sizes", l3, "--l3-ways", "16", "--l1d-size", "32kB", "--l2-size", "256kB",
           "--prefetcher", pf, "--ecg-pfx-lookahead", str(lookahead), "--out-dir", str(outdir)]
    env = dict(os.environ, CACHE_PFX_MTLB_ENTRIES=str(mtlb))
    try:
        subprocess.run(cmd, env=env, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=3600, check=False)
        r = json.load(open(outdir / "roi_matrix.json"))
        r = r[0] if isinstance(r, list) else r
        return {
            "fills": r.get("prefetch_fills") or 0,
            "p4k": r.get("prefetch_distinct_pages_4k") or 0,
            "p2m": r.get("prefetch_distinct_pages_2m") or 0,
            "mtlb_miss": r.get("prefetch_mtlb_misses") or 0,
        }
    except Exception:
        return None


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="web-Google:512kB soc-pokec:1MB com-orkut:2MB")
    ap.add_argument("--mtlb", default="128 8192", help="space-separated MTLB entry counts")
    ap.add_argument("--lookahead", type=int, default=8)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    cells = [c.split(":") for c in args.cells.split()]
    mtlbs = [int(m) for m in args.mtlb.split()]
    hdr = ["graph/L3", "prefetcher", "fills", "distinct_4k", "distinct_2m"] + [f"MTLB{m}_miss" for m in mtlbs]
    print("\n=== Property-prefetch translation pressure (page granularity) ===")
    print("distinct_4k/2m: pages the prefetch targets touch (infinite-TLB view).")
    print("MTLBn_miss: misses of a finite n-entry LRU MTLB (DROPLET-style MC TLB).\n")
    print("".join(h.ljust(17) if i < 2 else h.rjust(13) for i, h in enumerate(hdr)))
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for graph, l3 in cells:
        for pf in PREFETCHERS:
            base = None
            misses = []
            for m in mtlbs:
                r = run(graph, l3, pf, m, args.lookahead)
                if r is None:
                    misses.append(None); continue
                base = r
                misses.append(r["mtlb_miss"])
            if base is None:
                continue
            def mf(v): return f"{v/1e6:.2f}M" if v and v >= 1e6 else (str(v) if v is not None else "--")
            cols = [mf(base["fills"]), str(base["p4k"]), str(base["p2m"])] + [mf(x) for x in misses]
            print(f"{graph+'/'+l3:17s}{pf:17s}" + "".join(c.rjust(13) for c in cols[:1]) +
                  "".join(c.rjust(13) for c in cols[1:]))
            md.append("| " + f"{graph}/{l3}" + " | " + pf + " | " + " | ".join(cols) + " |")
        md.append("|" + "   |" * len(hdr))
    if args.out:
        Path(args.out).write_text("\n".join(md) + "\n")
        print(f"\n[written] {args.out}")


if __name__ == "__main__":
    main(sys.argv[1:])
