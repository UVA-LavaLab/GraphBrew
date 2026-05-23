#!/usr/bin/env python3
"""Amortisation analysis for Hour 4.
Computes amortised_ratio(n) = (reorder_time + n * kernel_avg) / (n * baseline_kernel_avg)
for each cell across n in {1, 5, 20, 100}.
"""
import os, re, glob, sys
from collections import defaultdict

def parse(path):
    if not os.path.exists(path): return None
    try:
        txt = open(path).read()
    except Exception: return None
    out = {}
    m = re.search(r'Average Time:\s*([\d.]+)', txt); out['kavg'] = float(m.group(1)) if m else None
    # Reorder Time may have several formats; grep all.
    m = re.search(r'Reorder Time:\s*([\d.]+)', txt)
    if not m: m = re.search(r'Reorder.*?:\s*([\d.]+)\s*s', txt)
    out['reorder'] = float(m.group(1)) if m else 0.0
    return out

def amortised(reorder, kavg, base_kavg, n):
    if not kavg or not base_kavg: return None
    return (reorder + n * kavg) / (n * base_kavg)

def main(dirs, labels, baseline_label='A'):
    cells = defaultdict(dict)
    for d in dirs:
        for f in glob.glob(f"{d}/*.log"):
            base = os.path.basename(f)[:-4]
            L = None
            for cand in sorted(labels, key=len, reverse=True):
                if base.endswith('_'+cand):
                    L = cand; rest = base[:-(len(cand)+1)]; break
            if not L: continue
            kernel = None
            for k in ('pr','bfs','cc','sssp','tc','bc'):
                if rest.endswith('_'+k):
                    kernel = k; graph = rest[:-(len(k)+1)]; break
            if not kernel: continue
            cells[(graph,kernel)][L] = parse(f)

    print('| Graph | Kernel | Config | reorder(s) | kavg(s) | A_kavg | amort(n=1) | amort(n=5) | amort(n=20) | amort(n=100) |')
    print('|---|---|---|---:|---:|---:|---:|---:|---:|---:|')
    for (g,k), d in sorted(cells.items()):
        base_d = d.get(baseline_label)
        if not base_d or not base_d.get('kavg'): continue
        bk = base_d['kavg']
        for L, r in sorted(d.items()):
            if L == baseline_label or not r or not r.get('kavg'): continue
            ro = r.get('reorder', 0.0)
            row = [f"{ro:.3f}", f"{r['kavg']:.5f}", f"{bk:.5f}"]
            for n in (1, 5, 20, 100):
                row.append(f"{amortised(ro, r['kavg'], bk, n):.3f}")
            print(f"| {g} | {k} | {L} | " + ' | '.join(row) + " |")

if __name__ == '__main__':
    dirs = sys.argv[1:] or ['/tmp/h3_hub2']
    labels = ['A','LeidH_dgd','SgRabH_dgd','LeidHub2_dgd','SgRabHub2_dgd','LeidBFS_dgd','LeidCutMin_dgd','LeidHilbert_dgd']
    main(dirs, labels)
