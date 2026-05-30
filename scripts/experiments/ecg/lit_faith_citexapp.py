"""Literature-faithfulness cross-app rationale coherence audit (LIT-CXApp,
gate 228).

Each lit-faith row carries a (`citation`, `expected_sign`, `rationale`)
triple. The citation+sign tuple identifies a *literature claim* (the
specific paper-section / figure being asserted); the rationale is the
per-cell justification for *this* (graph, app, policy, l3) measurement
under that claim.

When the same (citation, sign) group spans many cells, the rationales
must remain mutually coherent — otherwise the corpus is silently
quoting the same paper to support contradictory predictions.

This gate audits, per (citation, sign) group:

  * **Contradiction detection** — no pair of rationales in the same
    group may mix opposing-direction vocabulary (e.g., one says
    "dominates" while another says "underperforms").
  * **Sign-vocabulary alignment** — every rationale must mention at
    least one term consistent with its `expected_sign` band:
      - `sign='-'`  (policy better than GRASP): wins/dominates/improves...
      - `sign='+'`  (policy worse than GRASP):  underperforms/regresses...
      - `sign='~'`  (policy ≈ GRASP):           near-LRU/comparable/modulo...
  * **Common-kernel check** — every group of size ≥ 2 must share at
    least one non-stopword anchor token across all member rationales,
    so the cell-level rationales clearly stay within the same claim.
  * **Length-span ratio** — within a group, `max_len / min_len ≤ 3.0`
    so we don't have a 30-char rationale next to a 250-char one.

The gate is read-only and pure: it processes
`literature_faithfulness_postfix.json` and emits a per-group audit
table; the test suite pins the floors and the zero-contradiction
invariant.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parents[3]

NEGATIVE_VOCAB: Set[str] = {
    # used in sign='-' rationales (policy dominates GRASP)
    "dominates", "dominate", "wins", "win", "improves", "improve",
    "improvement", "improvements", "better", "outperform", "outperforms",
    "outperformed", "oracle", "reduces", "reduction", "retains", "retain",
    "advantage", "lower", "gain", "gains", "lead", "leads", "largest",
    "exceeds", "modest", "beats", "beat", "below", "benefits", "benefit",
    "positive", "strongest", "headline", "smaller-than-pr", "close",
    "strong", "stronger", "spills", "spill",
}
POSITIVE_VOCAB: Set[str] = {
    # used in sign='+' rationales (policy worse than GRASP)
    "worse", "underperforms", "underperform", "regresses", "regress",
    "regression", "higher", "exceeds-grasp",
}
NEUTRAL_VOCAB: Set[str] = {
    # used in sign='~' rationales (policy ≈ GRASP)
    "near", "near-lru", "near-grasp", "behaves", "comparable", "tie",
    "modulo", "weak", "scan-resistant", "scan", "marginally", "parity",
    "small", "modest", "fits", "fit", "shrink", "shrinks", "within",
    "agree", "agrees", "agreement", "disagreement", "cross-check",
    "phase-transition", "cross-checks",
}

SIGN_VOCAB: Dict[str, Set[str]] = {
    "-": NEGATIVE_VOCAB,
    "+": POSITIVE_VOCAB,
    "~": NEUTRAL_VOCAB,
}

# Vocabulary pairs that, if both present in a single group, indicate
# a coherence breakdown. Kept narrow on purpose — many "positive" GRASP
# rationales legitimately use words like "spills" (the property array
# spills the LLC and GRASP still retains the gain) so we only flag
# direct directional opposites.
OPPOSING_PAIRS: List[Tuple[Set[str], Set[str]]] = [
    ({"dominates", "dominate", "outperform", "outperforms", "wins", "win",
      "beats", "beat"},
     {"underperforms", "underperform", "regresses", "regress"}),
    ({"better", "improves", "improve", "improvement"},
     {"worse", "regresses", "regress", "regression"}),
]

STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "and", "or",
    "but", "of", "in", "to", "from", "for", "on", "at", "by", "with", "as",
    "that", "this", "these", "those", "it", "its", "than", "then", "so",
    "not", "no", "if", "we", "any", "some", "all", "each", "every", "such",
    "can", "may", "must", "should", "would", "could", "do", "does", "did",
    "has", "have", "had", "their", "there", "where", "when", "which",
    "while", "because", "although", "even", "still", "also", "more", "less",
    "most", "least", "very", "only", "just", "still", "see",
}

TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9\-]+")


def _tokens(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_RE.findall(text)]


# Detect negation-context: words like "regress", "worse", "regression"
# preceded within 2 tokens by a negator (no, not, never, hardly, must NOT,
# should not, does not). These do not count as a directional signal.
_NEGATORS: Set[str] = {"no", "not", "never", "hardly", "without", "n't"}


def _directional_tokens(text: str) -> Set[str]:
    """Return tokens that contribute to direction, excluding any that
    appear in negation context (e.g., 'must NOT regress')."""
    toks = _tokens(text)
    keep: Set[str] = set()
    for i, t in enumerate(toks):
        ctx = set(toks[max(0, i - 3): i])
        if ctx & _NEGATORS:
            continue
        keep.add(t)
    return keep


def _content_tokens(text: str) -> Set[str]:
    return {t for t in _tokens(text)
            if t not in STOPWORDS and len(t) >= 3}


def _sign_vocab_hits(rationale: str, sign: str) -> List[str]:
    vocab = SIGN_VOCAB.get(sign, set())
    if not vocab:
        return []
    toks = set(_tokens(rationale))
    return sorted(toks & vocab)


def _detect_contradictions(rationales: List[str]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    per_rationale_tokens = [(r, _directional_tokens(r)) for r in rationales]
    for i, (ra, toksa) in enumerate(per_rationale_tokens):
        for rb, toksb in per_rationale_tokens[i + 1:]:
            for pos, neg in OPPOSING_PAIRS:
                if (toksa & pos and toksb & neg) or (toksa & neg and toksb & pos):
                    out.append({
                        "rationale_a": ra,
                        "rationale_b": rb,
                        "pos_terms": sorted(pos),
                        "neg_terms": sorted(neg),
                    })
                    break
    return out


def _group_kernel(rationales: List[str]) -> List[str]:
    if not rationales:
        return []
    per = [_content_tokens(r) for r in rationales]
    kernel = set(per[0])
    for s in per[1:]:
        kernel &= s
    return sorted(kernel)


def _length_span(rationales: List[str]) -> Tuple[int, int, float]:
    lens = [len(r) for r in rationales]
    mn = min(lens)
    mx = max(lens)
    ratio = (mx / mn) if mn else float("inf")
    return mn, mx, round(ratio, 3)


def build_audit(lit_faith: Dict[str, Any]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = list(lit_faith.get("per_claim") or [])
    if not rows:
        raise SystemExit("[lit-faith-citexapp] empty per_claim table — run `make lit-faith` first")

    by_group: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        cite = r.get("citation") or ""
        sign = r.get("expected_sign") or ""
        by_group[(cite, sign)].append(r)

    groups: List[Dict[str, Any]] = []
    total_contradictions = 0
    total_sign_misses = 0
    rows_with_misses = 0
    groups_failing_kernel = 0
    groups_failing_span = 0

    for (cite, sign), members in sorted(by_group.items()):
        rationales = sorted({m["rationale"] for m in members if m.get("rationale")})
        per_app: Dict[str, Set[str]] = defaultdict(set)
        for m in members:
            per_app[m.get("app", "?")].add(m.get("rationale", ""))

        contradictions = _detect_contradictions(rationales)
        kernel = _group_kernel(rationales)
        mn, mx, ratio = _length_span(rationales) if rationales else (0, 0, 0.0)

        per_rationale: List[Dict[str, Any]] = []
        for r in rationales:
            hits = _sign_vocab_hits(r, sign)
            if not hits:
                total_sign_misses += 1
                rows_with_misses += sum(1 for m in members if m.get("rationale") == r)
            per_rationale.append({
                "rationale":     r,
                "length":        len(r),
                "sign_hits":     hits,
                "sign_aligned":  bool(hits),
                "member_count":  sum(1 for m in members if m.get("rationale") == r),
            })

        kernel_ok = (len(rationales) <= 1) or bool(kernel)
        span_ok = ratio <= 3.0 or len(rationales) <= 1
        if not kernel_ok:
            groups_failing_kernel += 1
        if not span_ok:
            groups_failing_span += 1
        total_contradictions += len(contradictions)

        groups.append({
            "citation":            cite,
            "expected_sign":       sign,
            "member_count":        len(members),
            "rationale_count":     len(rationales),
            "app_set":             sorted({m.get("app", "?") for m in members}),
            "graph_set":           sorted({m.get("graph", "?") for m in members}),
            "policy_set":          sorted({m.get("policy", "?") for m in members}),
            "kernel_terms":        kernel,
            "kernel_size":         len(kernel),
            "kernel_ok":           kernel_ok,
            "length_min":          mn,
            "length_max":          mx,
            "length_span_ratio":   ratio,
            "length_span_ok":      span_ok,
            "contradiction_count": len(contradictions),
            "contradictions":      contradictions,
            "rationales":          per_rationale,
        })

    audit = {
        "schema_version": 1,
        "summary": {
            "total_rows":              len(rows),
            "group_count":             len(groups),
            "multi_member_group_count": sum(1 for g in groups if g["member_count"] > 1),
            "unique_rationale_total":  sum(g["rationale_count"] for g in groups),
            "total_contradictions":    total_contradictions,
            "sign_alignment_misses":   total_sign_misses,
            "rows_with_sign_miss":     rows_with_misses,
            "groups_failing_kernel":   groups_failing_kernel,
            "groups_failing_span":     groups_failing_span,
        },
        "sign_vocab": {
            "-": sorted(NEGATIVE_VOCAB),
            "+": sorted(POSITIVE_VOCAB),
            "~": sorted(NEUTRAL_VOCAB),
        },
        "opposing_pairs": [
            {"pos": sorted(p), "neg": sorted(n)} for p, n in OPPOSING_PAIRS
        ],
        "length_span_ceiling": 3.0,
        "groups": groups,
    }
    return audit


def write_markdown(audit: Dict[str, Any], path: Path) -> None:
    s = audit["summary"]
    lines: List[str] = []
    lines.append("# Literature-faithfulness cross-app rationale coherence audit\n")
    lines.append("Generated by `make lit-citexapp`. Audits the per-cell rationales "
                 "for every (citation, expected_sign) group in the lit-faith corpus.\n")
    lines.append("## Summary\n")
    lines.append(f"- Total rows: **{s['total_rows']}**")
    lines.append(f"- (citation, sign) groups: **{s['group_count']}** "
                 f"(multi-member: {s['multi_member_group_count']})")
    lines.append(f"- Unique rationales: **{s['unique_rationale_total']}**")
    lines.append(f"- Contradictions: **{s['total_contradictions']}**")
    lines.append(f"- Sign-alignment misses: **{s['sign_alignment_misses']}** "
                 f"(rows affected: {s['rows_with_sign_miss']})")
    lines.append(f"- Groups failing common-kernel: **{s['groups_failing_kernel']}**")
    lines.append(f"- Groups failing length-span (≤ {audit['length_span_ceiling']}): "
                 f"**{s['groups_failing_span']}**\n")

    lines.append("## Group coherence table\n")
    lines.append("| sign | members | rationales | kernel | span | contradictions | citation |")
    lines.append("|-----:|--------:|-----------:|-------:|-----:|--------------:|----------|")
    for g in audit["groups"]:
        cite = g["citation"]
        if len(cite) > 70:
            cite = cite[:67] + "..."
        lines.append(
            f"| `{g['expected_sign']}` | {g['member_count']} | "
            f"{g['rationale_count']} | {g['kernel_size']} | "
            f"{g['length_span_ratio']:.2f} | {g['contradiction_count']} | "
            f"{cite} |"
        )
    lines.append("")

    if any(not r["sign_aligned"] for g in audit["groups"] for r in g["rationales"]):
        lines.append("## Rationales missing sign-vocabulary\n")
        for g in audit["groups"]:
            misses = [r for r in g["rationales"] if not r["sign_aligned"]]
            if not misses:
                continue
            lines.append(f"- **sign=`{g['expected_sign']}` / cite={g['citation'][:60]}...**")
            for r in misses:
                lines.append(f"  - ({r['member_count']} cells) {r['rationale'][:120]}")
        lines.append("")

    if s["total_contradictions"]:
        lines.append("## Contradictions\n")
        for g in audit["groups"]:
            if not g["contradictions"]:
                continue
            lines.append(f"- **sign=`{g['expected_sign']}` / cite={g['citation'][:60]}...**")
            for c in g["contradictions"]:
                lines.append(f"  - `{c['rationale_a'][:80]}...` vs `{c['rationale_b'][:80]}...`")
        lines.append("")
    else:
        lines.append("_No contradictions detected — every (citation, sign) group is mutually coherent._")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_csv(audit: Dict[str, Any], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "citation", "expected_sign", "member_count", "rationale_count",
            "kernel_size", "kernel_ok", "length_min", "length_max",
            "length_span_ratio", "length_span_ok", "contradiction_count",
        ])
        w.writeheader()
        for g in audit["groups"]:
            w.writerow({
                "citation":            g["citation"],
                "expected_sign":       g["expected_sign"],
                "member_count":        g["member_count"],
                "rationale_count":     g["rationale_count"],
                "kernel_size":         g["kernel_size"],
                "kernel_ok":           g["kernel_ok"],
                "length_min":          g["length_min"],
                "length_max":          g["length_max"],
                "length_span_ratio":   g["length_span_ratio"],
                "length_span_ok":      g["length_span_ok"],
                "contradiction_count": g["contradiction_count"],
            })


def main(argv: Iterable[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="LIT-CXApp cross-app rationale coherence audit (gate 228)")
    p.add_argument("--lit-faith-json", type=Path,
                   default=REPO_ROOT / "wiki" / "data" / "literature_faithfulness_postfix.json")
    p.add_argument("--json-out", type=Path,
                   default=REPO_ROOT / "wiki" / "data" / "lit_faith_citexapp.json")
    p.add_argument("--md-out", type=Path,
                   default=REPO_ROOT / "wiki" / "data" / "lit_faith_citexapp.md")
    p.add_argument("--csv-out", type=Path,
                   default=REPO_ROOT / "wiki" / "data" / "lit_faith_citexapp.csv")
    args = p.parse_args(list(argv) if argv is not None else None)

    if not args.lit_faith_json.exists():
        print(f"[lit-faith-citexapp] missing {args.lit_faith_json}; run `make lit-faith` first",
              file=sys.stderr)
        return 1

    lit_faith = json.loads(args.lit_faith_json.read_text(encoding="utf-8"))
    audit = build_audit(lit_faith)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                              encoding="utf-8")
    write_markdown(audit, args.md_out)
    write_csv(audit, args.csv_out)

    s = audit["summary"]
    print(f"[lit-faith-citexapp] {s['group_count']} groups; "
          f"contradictions {s['total_contradictions']}, "
          f"sign-misses {s['sign_alignment_misses']}, "
          f"kernel-fail {s['groups_failing_kernel']}, "
          f"span-fail {s['groups_failing_span']}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
