#!/usr/bin/env python3
"""Multiple-testing correction across the entire confidence-gate p-value family.

Across gates 34 (per-family-app paired bootstrap), 36/37 are CI-based
(no p-values), 38 (Cliff's delta + Mann-Whitney) we now have ~30-50 p-values.
With α=0.05 per test we'd expect 1.5-2.5 false positives by chance.

This gate applies both:

  * Holm-Bonferroni step-down  — strong FWER control at α=0.05.
  * Benjamini-Hochberg step-up  — FDR control at q=0.05.

and pins how many of the "significant" findings survive each correction.

Conservative paper claim: only HB-survivors should be stated as
"significant".  BH-survivors are "discoveries with FDR ≤ 0.05".

Sources of p-values aggregated:

  * wiki/data/oracle_gap_effect_size.json
      per_app[app][pair_key].mannwhitney_p  (two-sided, dedup unordered pair)
  * wiki/data/oracle_gap_by_app_bootstrap.json
      per_app_pairs[app][pair_key].p_a_lt_b  (one-sided → two-sided)
  * wiki/data/popt_vs_grasp_by_family_app.json
      per_family_app[fam/app].p_popt_lt_grasp  (one-sided → two-sided)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

ALPHA = 0.05


def _resolve_label(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _two_sided_from_one_sided(p_one: float) -> float:
    """Convert a one-sided p in [0,1] to two-sided via 2 * min(p, 1-p)."""
    return min(1.0, 2.0 * min(p_one, 1.0 - p_one))


def collect_p_values(
    effect_size_path: Path,
    bootstrap_path: Path,
    family_app_path: Path,
) -> list[dict]:
    """Return a deduplicated list of {source, scope, label, p_two_sided}.

    Each unordered pair contributes exactly one p-value.
    """
    rows: list[dict] = []

    # 1) Mann-Whitney from oracle_gap_effect_size (already two-sided).
    if effect_size_path.exists():
        es = json.loads(effect_size_path.read_text())
        for app, app_block in es.get("per_app", {}).items():
            seen: set[tuple[str, str]] = set()
            for payload in app_block.get("comparisons", []):
                a = payload.get("a")
                b = payload.get("b")
                if a is None or b is None:
                    continue
                ord_pair = tuple(sorted([a, b]))
                if ord_pair in seen:
                    continue
                seen.add(ord_pair)
                p = float(payload["mannwhitney_p"])
                rows.append({
                    "source": "mannwhitney_gap",
                    "scope":  f"app={app}",
                    "label":  f"{ord_pair[0]} vs {ord_pair[1]}",
                    "p_two_sided": p,
                })

    # 2) Bootstrap one-sided from oracle_gap_by_app_bootstrap.
    if bootstrap_path.exists():
        bs = json.loads(bootstrap_path.read_text())
        for app, pairs in bs.get("per_app_pairs", {}).items():
            seen = set()
            for pair_key, payload in pairs.items():
                if "_vs_" not in pair_key:
                    continue
                a, b = pair_key.split("_vs_", 1)
                ord_pair = tuple(sorted([a, b]))
                if ord_pair in seen:
                    continue
                seen.add(ord_pair)
                p_one = float(payload["p_a_lt_b"])
                p_two = _two_sided_from_one_sided(p_one)
                rows.append({
                    "source": "bootstrap_paired_gap",
                    "scope":  f"app={app}",
                    "label":  f"{ord_pair[0]} vs {ord_pair[1]}",
                    "p_two_sided": p_two,
                })

    # 3) Per-(family,app) POPT-vs-GRASP one-sided.
    if family_app_path.exists():
        fa = json.loads(family_app_path.read_text())
        for fam_app, payload in fa.get("per_family_app", {}).items():
            p_raw = payload.get("p_popt_lt_grasp")
            if p_raw is None:
                continue
            p_one = float(p_raw)
            p_two = _two_sided_from_one_sided(p_one)
            rows.append({
                "source": "popt_vs_grasp_family_app",
                "scope":  fam_app,
                "label":  "POPT vs GRASP",
                "p_two_sided": p_two,
            })

    return rows


def holm_bonferroni(p_values: list[float], alpha: float) -> list[dict]:
    """Holm-Bonferroni step-down. Returns list of dicts with original index,
    sorted p, adjusted threshold, and survives flag.
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    out: list[dict] = []
    rejected_so_far = True
    for rank, (orig_i, p) in enumerate(indexed, start=1):
        threshold = alpha / (n - rank + 1)
        survives = rejected_so_far and (p <= threshold)
        if not survives:
            rejected_so_far = False
        out.append({
            "rank": rank,
            "orig_index": orig_i,
            "p": p,
            "threshold": threshold,
            "survives": survives,
        })
    return out


def benjamini_hochberg(p_values: list[float], q: float) -> list[dict]:
    """Benjamini-Hochberg step-up FDR control at level q.

    Finds largest k with p_(k) <= k/n * q; rejects all H_(1)..H_(k).
    """
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    largest_k = 0
    for rank, (_, p) in enumerate(indexed, start=1):
        if p <= rank / n * q:
            largest_k = rank
    out: list[dict] = []
    for rank, (orig_i, p) in enumerate(indexed, start=1):
        threshold = rank / n * q
        out.append({
            "rank": rank,
            "orig_index": orig_i,
            "p": p,
            "threshold": threshold,
            "survives": rank <= largest_k,
        })
    return out


def build_payload(rows: list[dict], alpha: float) -> dict:
    p_values = [r["p_two_sided"] for r in rows]
    hb = holm_bonferroni(p_values, alpha)
    bh = benjamini_hochberg(p_values, alpha)

    naive_significant = sum(1 for p in p_values if p <= alpha)
    hb_survivors = sum(1 for r in hb if r["survives"])
    bh_survivors = sum(1 for r in bh if r["survives"])

    hb_lookup = {r["orig_index"]: r for r in hb}
    bh_lookup = {r["orig_index"]: r for r in bh}
    annotated = []
    for i, r in enumerate(rows):
        annotated.append({
            **r,
            "naive_significant_at_alpha": r["p_two_sided"] <= alpha,
            "holm_bonferroni_survives": hb_lookup[i]["survives"],
            "benjamini_hochberg_survives": bh_lookup[i]["survives"],
        })

    # Per-source breakdown.
    by_source: dict[str, dict] = {}
    for r in annotated:
        src = r["source"]
        s = by_source.setdefault(src, {
            "n_tests": 0,
            "naive_significant": 0,
            "hb_survivors": 0,
            "bh_survivors": 0,
        })
        s["n_tests"] += 1
        s["naive_significant"] += int(r["naive_significant_at_alpha"])
        s["hb_survivors"] += int(r["holm_bonferroni_survives"])
        s["bh_survivors"] += int(r["benjamini_hochberg_survives"])

    return {
        "meta": {
            "alpha": alpha,
            "n_tests": len(p_values),
            "naive_significant_count": naive_significant,
            "holm_bonferroni_survivor_count": hb_survivors,
            "benjamini_hochberg_survivor_count": bh_survivors,
            "expected_false_positives_at_alpha": round(alpha * len(p_values), 3),
        },
        "by_source": by_source,
        "all_tests": annotated,
        "holm_bonferroni_ladder": hb,
        "benjamini_hochberg_ladder": bh,
    }


def write_md(payload: dict, md_path: Path) -> None:
    meta = payload["meta"]
    lines = [
        "# Multiple-testing correction across confidence gates",
        "",
        "Holm-Bonferroni (FWER) and Benjamini-Hochberg (FDR) corrections",
        "applied jointly to every p-value emitted by the confidence-gate suite.",
        "",
        f"- α (per-test): **{meta['alpha']}**",
        f"- Number of tests in the family: **{meta['n_tests']}**",
        f"- Naive significant (p ≤ α, uncorrected): **{meta['naive_significant_count']}**",
        f"- Holm-Bonferroni survivors (FWER ≤ α): "
        f"**{meta['holm_bonferroni_survivor_count']}**",
        f"- Benjamini-Hochberg survivors (FDR ≤ α): "
        f"**{meta['benjamini_hochberg_survivor_count']}**",
        f"- Expected false positives if all nulls true: "
        f"**{meta['expected_false_positives_at_alpha']}**",
        "",
        "## Per-source breakdown",
        "",
        "| Source | n tests | Naive sig | HB survives | BH survives |",
        "| :----- | ------: | --------: | ----------: | ----------: |",
    ]
    for src, s in sorted(payload["by_source"].items()):
        lines.append(
            f"| `{src}` | {s['n_tests']} | {s['naive_significant']} | "
            f"{s['hb_survivors']} | {s['bh_survivors']} |"
        )

    survivors = [r for r in payload["all_tests"] if r["holm_bonferroni_survives"]]
    survivors.sort(key=lambda r: r["p_two_sided"])
    lines += [
        "",
        f"## Holm-Bonferroni survivors (n={len(survivors)})",
        "",
        "These are the claims that survive **strong family-wise** correction;",
        "they are the ones safe to state as 'statistically significant' in",
        "the paper at α=0.05.",
        "",
        "| Source | Scope | Comparison | p (two-sided) |",
        "| :----- | :---- | :--------- | -----------: |",
    ]
    for r in survivors:
        p = r["p_two_sided"]
        p_str = f"{p:.2e}" if p > 0 else "0"
        lines.append(
            f"| `{r['source']}` | {r['scope']} | {r['label']} | {p_str} |"
        )
    if not survivors:
        lines.append("| (none) | — | — | — |")

    bh_only = [
        r for r in payload["all_tests"]
        if r["benjamini_hochberg_survives"] and not r["holm_bonferroni_survives"]
    ]
    bh_only.sort(key=lambda r: r["p_two_sided"])
    lines += [
        "",
        f"## BH-only survivors (n={len(bh_only)})",
        "",
        "These survive FDR control but NOT FWER — paper-honest framing is",
        "'discoveries with FDR ≤ 5%', not 'significant'.",
        "",
        "| Source | Scope | Comparison | p (two-sided) |",
        "| :----- | :---- | :--------- | -----------: |",
    ]
    for r in bh_only:
        p = r["p_two_sided"]
        p_str = f"{p:.2e}" if p > 0 else "0"
        lines.append(
            f"| `{r['source']}` | {r['scope']} | {r['label']} | {p_str} |"
        )
    if not bh_only:
        lines.append("| (none) | — | — | — |")

    md_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--effect-size-json",
        default=str(WIKI_DATA / "oracle_gap_effect_size.json"),
    )
    parser.add_argument(
        "--bootstrap-json",
        default=str(WIKI_DATA / "oracle_gap_by_app_bootstrap.json"),
    )
    parser.add_argument(
        "--family-app-json",
        default=str(WIKI_DATA / "popt_vs_grasp_by_family_app.json"),
    )
    parser.add_argument(
        "--json-out",
        default=str(WIKI_DATA / "multiple_testing_correction.json"),
    )
    parser.add_argument(
        "--md-out",
        default=str(WIKI_DATA / "multiple_testing_correction.md"),
    )
    parser.add_argument("--alpha", type=float, default=ALPHA)
    args = parser.parse_args()

    rows = collect_p_values(
        Path(args.effect_size_json),
        Path(args.bootstrap_json),
        Path(args.family_app_json),
    )
    payload = build_payload(rows, args.alpha)

    json_path = Path(args.json_out)
    md_path = Path(args.md_out)
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_md(payload, md_path)

    meta = payload["meta"]
    print(
        f"[mt-correction] n_tests={meta['n_tests']} "
        f"naive_sig={meta['naive_significant_count']} "
        f"HB_survives={meta['holm_bonferroni_survivor_count']} "
        f"BH_survives={meta['benjamini_hochberg_survivor_count']} "
        f"→ {_resolve_label(md_path)}"
    )


if __name__ == "__main__":
    main()
