#!/usr/bin/env python3
"""Preliminary large-cache_sim ECG scale sweep (Sprint 2).

Headline config: 8B full epoch (--ecg-epoch-pack-bits 64 --ecg-epochs 65535,
honest charged record) + uniform structure-stream prefetcher applied to ALL
policies + size_correct charged P-OPT (reserves LLC ways for its resident
rereference-matrix columns). Compares GRASP vs charged P-OPT vs ECG across the
property-working-set pressure range, per the rubber-duck's de-risking:
  * spans fit -> high-pressure L3 per graph (property = 4*|V|)
  * degree-0 control slice (separate replacement win from structure hiding)
  * small -o0 / -i3 / shortcircuit sanity slices
  * captures DEMAND mr + PROPERTY mr + STRUCTURE misses + total traffic +
    prefetch fills + P-OPT feasibility (popt_matrix_fits) per cell

Single-thread cache_sim authority (OMP_NUM_THREADS=1). Incremental + resumable
CSV; kron-s24 ordered last so faster graphs land first. Outputs (gitignored):
results/ecg_experiments/prelim_scale/prelim_scale.csv
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
GRAPHS = ROOT / "results" / "graphs"
ROI = ROOT / "scripts" / "experiments" / "ecg" / "roi_matrix.py"
OUTDIR = ROOT / "results" / "ecg_experiments" / "prelim_scale"
CSV_PATH = OUTDIR / "prelim_scale.csv"
LOG_PATH = OUTDIR / "prelim_scale.log"

# Validated |V| from .sg headers (property array = 4*|V| bytes).
NUM_V = {
    "web-Google": 916_428,
    "roadNet-CA": 1_971_281,
    "soc-pokec": 1_632_804,
    "cit-Patents": 3_774_768,
    "com-orkut": 3_072_627,
    "soc-LiveJournal1": 4_847_571,
    "kron-s24": 16_777_212,
}
GRAPH_RANK = {  # process small/fast first, kron-s24 last
    "web-Google": 0, "roadNet-CA": 1, "soc-pokec": 2, "cit-Patents": 3,
    "com-orkut": 4, "soc-LiveJournal1": 5, "kron-s24": 6,
}
TIMEOUT = {  # per-cell seconds, scaled by graph cost
    "web-Google": 1800, "roadNet-CA": 1800, "soc-pokec": 1800,
    "cit-Patents": 2400, "com-orkut": 5400, "soc-LiveJournal1": 5400,
    "kron-s24": 10800,
}

# (column label, policy spec, ECG_VARIANT or None)
COLS = [
    ("LRU", "LRU", None),
    ("GRASP", "GRASP", None),
    ("POPT", "POPT", None),                 # charged size_correct
    ("POPT_UNCH", "POPT:UNCHARGED", None),  # full-capacity reference
    ("ECG", "ECG:ECG_GRASP_POPT", "epoch_only"),
]
COL_SC = ("ECG_sc", "ECG:ECG_GRASP_POPT", "shortcircuit")

# block -> {graph: [l3_MB,...]}  (order, prefetch_degree, iters, columns)
BLOCKS = [
    # main headline grid: prefetcher on, -o5, -i1
    ("main", 5, 4, 1, COLS, {
        "web-Google": [2, 4],
        "roadNet-CA": [4, 8],
        "com-orkut": [4, 8, 16],
        "cit-Patents": [4, 8, 16],
        "soc-LiveJournal1": [8, 16, 32],
        "kron-s24": [8, 16, 32],
    }),
    # degree-0 control: prefetcher OFF, one representative size/graph
    ("deg0", 5, 0, 1, COLS, {
        "web-Google": [4],
        "roadNet-CA": [8],
        "com-orkut": [8],
        "cit-Patents": [8],
        "soc-LiveJournal1": [16],
        "kron-s24": [16],
    }),
    # -o0 robustness (un-reordered) slice
    ("o0", 0, 4, 1, COLS, {
        "com-orkut": [8],
        "cit-Patents": [8],
        "kron-s24": [16],
    }),
    # -i3 steady-state sanity slice
    ("i3", 5, 4, 3, COLS, {
        "com-orkut": [8],
        "kron-s24": [16],
    }),
    # shortcircuit == epoch_only sanity (post valid-bit fix they should match)
    ("sc", 5, 4, 1, [COL_SC], {
        "com-orkut": [8],
        "kron-s24": [16],
    }),
]

FIELDS = [
    "block", "graph", "num_vertices", "prop_mb", "l3_mb", "order",
    "prefetch_degree", "iters", "column", "policy", "variant", "status",
    "l3_miss_rate", "l3_prop_miss_rate", "l3_misses", "l3_accesses",
    "l3_prop_misses", "l3_struct_misses", "total_memory_traffic",
    "prefetch_fills", "prefetch_requests", "prefetch_useful",
    "memory_accesses", "total_accesses", "popt_reserve_model",
    "popt_reserved_ways", "popt_matrix_fits", "ecg_charged",
    "ecg_epoch_pack_bits", "ecg_epochs", "seconds",
]


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as fh:
        fh.write(line + "\n")


def cell_key(block, graph, l3, order, degree, iters, col_label):
    return f"{block}|{graph}|{l3}|o{order}|d{degree}|i{iters}|{col_label}"


def load_done() -> set[str]:
    if not CSV_PATH.exists():
        return set()
    done = set()
    with open(CSV_PATH) as fh:
        for r in csv.DictReader(fh):
            if r.get("status") != "ok":
                continue  # re-run errored/timed-out cells on resume
            done.add(cell_key(r["block"], r["graph"], int(r["l3_mb"]),
                              int(r["order"]), int(r["prefetch_degree"]),
                              int(r["iters"]), r["column"]))
    return done


def run_one(block, graph, l3, order, degree, iters, col):
    label, policy, variant = col
    gpath = GRAPHS / graph / f"{graph}.sg"
    odir = Path("/tmp") / f"prelim_{block}_{graph}_{l3}MB_o{order}_d{degree}_i{iters}_{label}"
    cmd = [
        sys.executable, str(ROI), "--suite", "cache-sim", "--no-build",
        "--benchmark", "pr", "--policies", policy,
        "--options", f"-f {gpath} -o {order} -n 1 -i {iters}",
        "--l3-sizes", f"{l3}MB", "--l3-ways", "16",
        "--l1d-size", "32kB", "--l2-size", "256kB",
        "--popt-reserve-model", "size_correct",
        "--popt-active-columns", "2",
        "--ecg-epoch-pack-bits", "64",
        "--ecg-epochs", "65535",
        "--ecg-charged", "1",
        "--cache-stream-prefetch-degree", str(degree),
        "--timeout-cache", "7200",
        "--out-dir", str(odir),
    ]
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    if variant:
        env["ECG_VARIANT"] = variant
    row = {
        "block": block, "graph": graph, "num_vertices": NUM_V[graph],
        "prop_mb": round(4 * NUM_V[graph] / 2**20, 2), "l3_mb": l3,
        "order": order, "prefetch_degree": degree, "iters": iters,
        "column": label, "policy": policy, "variant": variant or "",
        "status": "error",
    }
    t0 = time.time()
    try:
        subprocess.run(cmd, env=env, cwd=str(ROOT), check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       timeout=TIMEOUT[graph])
        data = json.load(open(odir / "roi_matrix.json"))
        x = (data if isinstance(data, list) else [data])[0]
        for k in ("l3_miss_rate", "l3_prop_miss_rate", "l3_misses",
                  "l3_accesses", "l3_prop_misses", "l3_struct_misses",
                  "total_memory_traffic", "prefetch_fills", "prefetch_requests",
                  "prefetch_useful", "memory_accesses", "total_accesses",
                  "popt_reserve_model", "popt_reserved_ways", "popt_matrix_fits",
                  "ecg_charged", "ecg_epoch_pack_bits", "ecg_epochs"):
            row[k] = x.get(k)
        row["status"] = x.get("status", "ok")
    except subprocess.TimeoutExpired:
        row["status"] = "timeout"
    except Exception as e:  # noqa: BLE001
        row["status"] = f"error:{type(e).__name__}"
    row["seconds"] = round(time.time() - t0, 1)
    return row


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    done = load_done()
    new_file = not CSV_PATH.exists()

    # flatten all (block,graph,l3,...,col) cells, order kron-s24 last
    cells = []
    for block, order, degree, iters, cols, grid in BLOCKS:
        for graph, sizes in grid.items():
            for l3 in sizes:
                for col in cols:
                    cells.append((block, graph, l3, order, degree, iters, col))
    cells.sort(key=lambda c: (GRAPH_RANK[c[1]], c[2], c[0], c[6][0]))

    todo = [c for c in cells
            if cell_key(c[0], c[1], c[2], c[3], c[4], c[5], c[6][0]) not in done]
    log(f"prelim scale sweep: {len(cells)} cells total, {len(todo)} to run "
        f"({len(done)} already done)")

    with open(CSV_PATH, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
            fh.flush()
        for i, (block, graph, l3, order, degree, iters, col) in enumerate(todo, 1):
            log(f"[{i}/{len(todo)}] {block} {graph} {l3}MB o{order} d{degree} "
                f"i{iters} {col[0]} ...")
            row = run_one(block, graph, l3, order, degree, iters, col)
            w.writerow(row)
            fh.flush()
            mr = row.get("l3_miss_rate")
            pmr = row.get("l3_prop_miss_rate")
            log(f"    -> {row['status']} demand_mr={mr} prop_mr={pmr} "
                f"fits={row.get('popt_matrix_fits')} {row['seconds']}s")
    log("DONE prelim scale sweep")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
