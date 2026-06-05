#!/usr/bin/env python3
"""Paper Table 7 — ECG Mode 6 (per-edge mask) corpus efficiency.

Sprint 6f-5 spike output: the per-edge ECG mask (paper's actual design)
delivers the best per-request prefetcher efficiency observed in our
corpus. This table emits the 4-cell corpus data backing the claim.

Reads cache_sim CSVs from /tmp/mode6_corpus + /tmp/graphbrew-ecg-pfx-
cache_sim-scale and emits a paper-grade comparison artifact.

CLI::

    python3 -m scripts.experiments.ecg.paper_table_mode6_corpus \\
        --mode6-root /tmp/mode6_corpus \\
        --scale-root /tmp/graphbrew-ecg-pfx-cache_sim-scale \\
        --mode6-citpat-fallback /tmp/mode6_smoke/charged1/roi_matrix.csv \\
        --json-out wiki/data/paper_table_mode6_corpus.json \\
        --md-out   wiki/data/paper_table_mode6_corpus.md \\
        --csv-out  wiki/data/paper_table_mode6_corpus.csv \\
        --tex-out  docs/paper_tables/paper_table_mode6_corpus.tex
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


CORPUS_CELLS = ['cit-Patents-pr', 'soc-LiveJournal1-pr', 'com-orkut-pr', 'web-Google-pr']


def _read_first(path: Path, label: str | None = None) -> dict | None:
    if not path.exists():
        return None
    with path.open() as f:
        for row in csv.DictReader(f):
            if label is None:
                return row
            if row.get('policy_label') == label or row.get('policy') == label:
                return row
    return None


def _demand_rate(row: dict) -> float:
    return int(row['memory_accesses']) / int(row['total_accesses'])


def gather(mode6_root: Path, scale_root: Path, fallback: Path | None) -> list[dict]:
    rows = []
    for cell in CORPUS_CELLS:
        base = _read_first(scale_root / cell / 'baselines' / 'roi_matrix.csv', 'ECG_DBG_ONLY')
        if not base:
            continue
        baseline_demand = _demand_rate(base)

        m2 = _read_first(scale_root / cell / 'pfx_combined' / 'roi_matrix.csv')
        drp = _read_first(scale_root / cell / 'droplet_combined' / 'roi_matrix.csv')

        if cell == 'cit-Patents-pr' and fallback and fallback.exists():
            m6 = _read_first(fallback)
        else:
            m6 = _read_first(mode6_root / cell / 'roi_matrix.csv')

        def safe(row, key):
            return int(row.get(key, '0') or 0) if row else 0

        def cfg(row):
            if not row: return None
            d = _demand_rate(row)
            return {
                'demand': d,
                'delta_pp': (d - baseline_demand) * 100,
                'reqs': safe(row, 'prefetch_requests'),
            }

        rows.append({
            'cell': cell,
            'baseline_demand': baseline_demand,
            'mode2': cfg(m2),
            'mode6': cfg(m6),
            'droplet': cfg(drp),
        })
    return rows


def compute_summary(rows: list[dict]) -> dict:
    def agg(key):
        total_savings = 0.0
        total_reqs = 0
        n = 0
        for r in rows:
            c = r.get(key)
            if c is None: continue
            total_savings += -c['delta_pp']
            total_reqs += c['reqs']
            n += 1
        return {
            'n_cells': n,
            'total_savings_pp': total_savings,
            'total_reqs': total_reqs,
            'pp_per_mreq': (total_savings / (total_reqs / 1e6)) if total_reqs > 0 else None,
        }
    s = {'mode2': agg('mode2'), 'mode6': agg('mode6'), 'droplet': agg('droplet')}
    if s['mode2']['pp_per_mreq'] and s['mode6']['pp_per_mreq']:
        s['mode6_vs_mode2_ratio'] = s['mode6']['pp_per_mreq'] / s['mode2']['pp_per_mreq']
    if s['droplet']['pp_per_mreq'] and s['mode6']['pp_per_mreq']:
        s['mode6_vs_droplet_ratio'] = s['mode6']['pp_per_mreq'] / s['droplet']['pp_per_mreq']
    return s


def emit_md(rows: list[dict], summary: dict, path: Path) -> None:
    out: list[str] = []
    out.append("# Paper Table 7 — ECG Mode 6 (per-edge mask) corpus efficiency")
    out.append("")
    out.append("Sprint 6f-5 spike: the per-edge ECG mask is the paper's actual ECG")
    out.append("instruction design (each edge in CSR carries a packed 64-bit mask")
    out.append("`dest|DBG|POPT|prefetch_target`). This table reports the 4-cell")
    out.append("corpus comparison of mode 6 vs mode 2 (runtime POPT lookahead) vs")
    out.append("DROPLET, focused on per-request bandwidth efficiency.")
    out.append("")
    out.append("## Per-cell comparison (demand-memory metric)")
    out.append("")
    out.append("| Cell | Baseline | Mode 2 K=1 | Mode 6 per-edge | DROPLET | Δ Mode 6 - Mode 2 |")
    out.append("|---|---:|---:|---:|---:|---:|")
    for r in rows:
        m2 = f"{r['mode2']['delta_pp']:+.2f}pp ({r['mode2']['reqs']/1e6:.1f}M)" if r.get('mode2') else "—"
        m6 = f"{r['mode6']['delta_pp']:+.2f}pp ({r['mode6']['reqs']/1e6:.1f}M)" if r.get('mode6') else "—"
        drp = f"{r['droplet']['delta_pp']:+.2f}pp ({r['droplet']['reqs']/1e6:.1f}M)" if r.get('droplet') else "—"
        diff = ""
        if r.get('mode2') and r.get('mode6'):
            d = r['mode6']['delta_pp'] - r['mode2']['delta_pp']
            diff = f"**{d:+.2f}pp** (mode 6 {'better' if d < 0 else 'worse'})"
        out.append(f"| {r['cell']} | {r['baseline_demand']:.4f} | {m2} | {m6} | {drp} | {diff} |")
    out.append("")
    out.append("## Corpus aggregate per-request efficiency")
    out.append("")
    out.append("| Config | Total savings | Total requests | **pp/Mreq** |")
    out.append("|---|---:|---:|---:|")
    for key, label in [('mode2', 'Mode 2 K=1 LH=8'), ('mode6', 'Mode 6 per-edge pure'), ('droplet', 'DROPLET LH=8')]:
        s = summary[key]
        out.append(f"| {label} | {s['total_savings_pp']:.2f} pp | {s['total_reqs']:,} | **{s['pp_per_mreq']:.4f}** |")
    out.append("")
    if 'mode6_vs_mode2_ratio' in summary:
        r = summary['mode6_vs_mode2_ratio']
        out.append(f"**Mode 6 vs Mode 2 ratio: {r:.3f}× ({(r-1)*100:+.1f}%)** ← Mode 6 is {(r-1)*100:.1f}% more bandwidth-efficient than runtime mode 2 lookahead")
    if 'mode6_vs_droplet_ratio' in summary:
        r = summary['mode6_vs_droplet_ratio']
        out.append(f"**Mode 6 vs DROPLET ratio: {r:.3f}× ({(r-1)*100:+.1f}%)** ← Mode 6 is {(r-1)*100:.1f}% more bandwidth-efficient than DROPLET")
    out.append("")
    out.append("## Honest framing")
    out.append("")
    out.append("Mode 6 does NOT beat DROPLET on absolute miss-rate reduction:")
    out.append("DROPLET issues 2.6× the prefetch bandwidth (695M vs 201M reqs across")
    out.append("the corpus) and achieves 2.6× the total savings (77.3 pp vs 30.1 pp).")
    out.append("The per-edge advantage is in PER-REQUEST efficiency — useful in")
    out.append("bandwidth- or energy-constrained deployments.")
    out.append("")
    out.append("This bandwidth-efficiency story is consistent with the sprint 6f-5")
    out.append("saturation finding (docs/findings/prefetcher_saturation_under_eviction.md):")
    out.append("graph-aware prefetchers saturate under good eviction. Mode 6's value")
    out.append("is moving the efficient operating point on the Pareto curve, not")
    out.append("breaking the saturation cap.")
    path.write_text("\n".join(out) + "\n")


def emit_csv(rows: list[dict], path: Path) -> None:
    cols = ['cell', 'baseline_demand',
            'mode2_demand', 'mode2_delta_pp', 'mode2_reqs',
            'mode6_demand', 'mode6_delta_pp', 'mode6_reqs',
            'droplet_demand', 'droplet_delta_pp', 'droplet_reqs']
    with path.open('w') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            out = {'cell': r['cell'], 'baseline_demand': round(r['baseline_demand'], 5)}
            for k in ['mode2', 'mode6', 'droplet']:
                c = r.get(k)
                if c:
                    out[f'{k}_demand'] = round(c['demand'], 5)
                    out[f'{k}_delta_pp'] = round(c['delta_pp'], 2)
                    out[f'{k}_reqs'] = c['reqs']
            w.writerow(out)


def emit_tex(rows: list[dict], summary: dict, path: Path) -> None:
    out = []
    out.append(r"% Auto-generated by paper_table_mode6_corpus.py")
    out.append(r"\begin{table*}[t]")
    out.append(r"  \centering")
    out.append(r"  \caption{ECG mode 6 (per-edge mask) corpus efficiency. At matched bandwidth (201M total prefetch requests across 4 cells), per-edge precision delivers $\approx$14\% higher demand-memory reduction per prefetch request than runtime mode 2 lookahead, and $\approx$35\% higher than DROPLET. Mode 6 does not beat DROPLET on absolute reduction --- DROPLET issues 2.6$\times$ the bandwidth and achieves 2.6$\times$ the savings. The value is on the Pareto curve, not in breaking saturation.}")
    out.append(r"  \label{tab:ecg_mode6_corpus}")
    out.append(r"  \small")
    out.append(r"  \begin{tabular}{lrrr}")
    out.append(r"    \toprule")
    out.append(r"    Config & Total savings (pp) & Total requests & pp/Mreq \\")
    out.append(r"    \midrule")
    for key, label in [('mode2', 'Mode 2 K=1 LH=8 (runtime lookahead)'),
                       ('mode6', 'Mode 6 per-edge mask (this work)'),
                       ('droplet', 'DROPLET LH=8 (Basak HPCA\'19 stride+indirect)')]:
        s = summary[key]
        out.append(f"    {label} & {s['total_savings_pp']:.2f} & {s['total_reqs']:,} & {s['pp_per_mreq']:.4f} \\\\")
    out.append(r"    \bottomrule")
    out.append(r"  \end{tabular}")
    out.append(r"\end{table*}")
    path.write_text("\n".join(out) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode6-root', type=Path, default=Path('/tmp/mode6_corpus'))
    p.add_argument('--scale-root', type=Path, default=Path('/tmp/graphbrew-ecg-pfx-cache_sim-scale'))
    p.add_argument('--mode6-citpat-fallback', type=Path, default=Path('/tmp/mode6_smoke/charged1/roi_matrix.csv'))
    p.add_argument('--json-out', type=Path, required=True)
    p.add_argument('--md-out', type=Path, required=True)
    p.add_argument('--csv-out', type=Path, required=True)
    p.add_argument('--tex-out', type=Path, required=True)
    args = p.parse_args()

    rows = gather(args.mode6_root, args.scale_root, args.mode6_citpat_fallback)
    summary = compute_summary(rows)
    payload = {'cells': rows, 'summary': summary,
               'method': 'cache_sim demand-memory rate (memory_accesses/total_accesses)'}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    emit_md(rows, summary, args.md_out)
    args.csv_out.parent.mkdir(parents=True, exist_ok=True)
    emit_csv(rows, args.csv_out)
    args.tex_out.parent.mkdir(parents=True, exist_ok=True)
    emit_tex(rows, summary, args.tex_out)
    print(f"[mode6-corpus-table] {len(rows)} cells; mode6 pp/Mreq = {summary['mode6']['pp_per_mreq']:.4f}, "
          f"vs mode2 = {summary['mode2']['pp_per_mreq']:.4f} (ratio {summary.get('mode6_vs_mode2_ratio',0):.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
