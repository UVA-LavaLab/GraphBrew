#!/usr/bin/env python3
"""Gate 251 — L3 cache-size registry.

The codebase ships >40 modules that each declare their own ``L3_*``
constant (``PAPER_L3``, ``PAPER_L3_SIZES``, ``L3_SIZES``, ``L3_MB``,
``L3_BYTES``).  Today every one of those constants happens to agree
on the same anchor triplet ``("1MB", "4MB", "8MB")``, but no test
fails if a future contributor adds a 36th module that re-declares
``L3_SIZES = ("2MB", "4MB", "16MB")`` and silently moves the
anchor.  Once that happens, the cross-tool reports, the saturation-
onset tallies, and the gem5/Sniper anchors stop comparing apples
to apples — and the paper text drifts from the data without any
gate firing.

This gate codifies the L3 size universe with a hand-curated
``CANONICAL_L3_TIERS`` (token → bytes + MB + role) plus the
anchor triplet ``ANCHOR_TRIPLET = ("1MB", "4MB", "8MB")``.  It
then AST-harvests every module-level ``L3_*``-shaped tuple/dict
literal in ``scripts/experiments/ecg/`` and ``scripts/test/`` and
asserts:

  L1 — every token used as a key/element in any harvested constant
       appears in ``CANONICAL_L3_TIERS`` (no rogue size names);
  L2 — every harvested PAPER_L3 / PAPER_L3_SIZES tuple equals the
       canonical ANCHOR_TRIPLET in element-order (no permutations
       or substitutions);
  L3 — every harvested ``L3_MB``-shaped dict pairs each anchor
       token with its canonical MB value (1MB→1.0, 4MB→4.0, 8MB→8.0);
  L4 — every harvested ``L3_BYTES``-shaped dict pairs each token
       with the correct byte count, exactly as declared in
       CANONICAL_L3_TIERS (caught via constant-folding the AST
       BinOp expressions like ``4 * 1024``);
  L5 — every CANONICAL_L3_TIERS entry has a valid role from
       {anchor, probe, sweep_low, sweep_high} and a valid sub-tier
       from {paper_anchor, small_l3_probe, knee, l_curve_lowend,
       l_curve_highend, reserved_bytes};
  L6 — every anchor-tier token (role=="anchor") appears in at
       least one harvested PAPER_L3 / PAPER_L3_SIZES tuple
       (defensive — anchor tokens must actually anchor something);
  L7 — no harvested constant uses a deprecated token (any token
       absent from CANONICAL_L3_TIERS is flagged); no two harvested
       PAPER_L3-shaped tuples disagree with each other.

Source-of-truth: every ``.py`` file under ``scripts/experiments/ecg/``
and ``scripts/test/`` (AST-harvested), plus ``CANONICAL_L3_TIERS``
in this file.
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2].parent
SOURCE_DIRS = [
    ROOT / "scripts" / "experiments" / "ecg",
    ROOT / "scripts" / "test",
]

# ----------------------------------------------------------- canonical --

# Each entry locks one L3 cache-size token.
#   bytes : exact byte count
#   mb    : MB scaling factor (only meaningful for >= 1MB tokens)
#   role  : 'anchor' (the 1/4/8 MB triplet used by the paper) |
#           'probe' (small-L3 thrash + onset probes) |
#           'sweep_low' (L-curve low-end <= 256kB) |
#           'sweep_high' (L-curve high-end >= 16MB)
#   sub_tier : finer label documenting the size's intended use
CANONICAL_L3_TIERS: dict[str, dict] = {
    "4kB":   {"bytes":     4 * 1024,         "mb": 4 / 1024,
              "role": "probe",
              "sub_tier": "small_l3_probe"},
    "16kB":  {"bytes":    16 * 1024,         "mb": 16 / 1024,
              "role": "sweep_low",
              "sub_tier": "l_curve_lowend"},
    "32kB":  {"bytes":    32 * 1024,         "mb": 32 / 1024,
              "role": "probe",
              "sub_tier": "small_l3_probe"},
    "64kB":  {"bytes":    64 * 1024,         "mb": 64 / 1024,
              "role": "sweep_low",
              "sub_tier": "l_curve_lowend"},
    "256kB": {"bytes":   256 * 1024,         "mb": 256 / 1024,
              "role": "sweep_low",
              "sub_tier": "knee"},
    "1MB":   {"bytes":  1024 * 1024,         "mb": 1.0,
              "role": "anchor",
              "sub_tier": "paper_anchor"},
    "2MB":   {"bytes":     2 * 1024 * 1024,  "mb": 2.0,
              "role": "sweep_high",
              "sub_tier": "l_curve_highend"},
    "4MB":   {"bytes":     4 * 1024 * 1024,  "mb": 4.0,
              "role": "anchor",
              "sub_tier": "paper_anchor"},
    "8MB":   {"bytes":     8 * 1024 * 1024,  "mb": 8.0,
              "role": "anchor",
              "sub_tier": "paper_anchor"},
    "16MB":  {"bytes":    16 * 1024 * 1024,  "mb": 16.0,
              "role": "sweep_high",
              "sub_tier": "l_curve_highend"},
    "32MB":  {"bytes":    32 * 1024 * 1024,  "mb": 32.0,
              "role": "sweep_high",
              "sub_tier": "l_curve_highend"},
}

ANCHOR_TRIPLET: tuple[str, str, str] = ("1MB", "4MB", "8MB")

VALID_ROLES = {"anchor", "probe", "sweep_low", "sweep_high"}
VALID_SUB_TIERS = {
    "paper_anchor", "small_l3_probe", "knee",
    "l_curve_lowend", "l_curve_highend", "reserved_bytes",
}

# Names of module-level constants the harvester treats as L3-shaped.
ANCHOR_TUPLE_NAMES = {"PAPER_L3", "PAPER_L3_SIZES", "L3_SIZES"}
MB_DICT_NAMES = {"L3_MB"}
BYTES_DICT_NAMES = {"L3_BYTES"}
# Other tuple names that should be a subset of canonical tokens
# (small-L3 probe tuples like MANDATORY_L3_SIZES).
SUBSET_TUPLE_NAMES = {"MANDATORY_L3_SIZES"}

# Self-skip — this file declares CANONICAL_L3_TIERS so naïve harvesting
# would pick it up as a "third party L3_BYTES".
SELF_SKIP = {
    Path(__file__).resolve().relative_to(ROOT).as_posix(),
}


# ----------------------------------------------------------- ast helpers --

def _fold_int(node: ast.AST) -> int | None:
    """Constant-fold a small int expression (used for ``4 * 1024``)."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int):
        return node.value
    if isinstance(node, ast.BinOp):
        left = _fold_int(node.left)
        right = _fold_int(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Add):
            return left + right
    return None


def _fold_float(node: ast.AST) -> float | None:
    if isinstance(node, ast.Constant) and isinstance(node.value,
                                                      (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        left = _fold_float(node.left)
        right = _fold_float(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right if right != 0 else None
        if isinstance(node.op, ast.Add):
            return left + right
    return None


def _tuple_str_elements(node: ast.AST) -> tuple[str, ...] | None:
    if not isinstance(node, (ast.Tuple, ast.List)):
        return None
    out: list[str] = []
    for el in node.elts:
        if isinstance(el, ast.Constant) and isinstance(el.value, str):
            out.append(el.value)
        else:
            return None
    return tuple(out)


def _dict_str_key_pairs(node: ast.AST):
    if not isinstance(node, ast.Dict):
        return None
    pairs: list[tuple[str, ast.AST]] = []
    for k, v in zip(node.keys, node.values):
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            pairs.append((k.value, v))
        else:
            return None
    return pairs


def _harvest_file(path: Path) -> dict:
    """Return per-file harvested constants."""
    rel = path.relative_to(ROOT).as_posix()
    out = {
        "rel":             rel,
        "anchor_tuples":   [],   # list[dict(name, value)]
        "mb_dicts":        [],   # list[dict(name, mapping)]
        "bytes_dicts":     [],   # list[dict(name, mapping)]
        "subset_tuples":   [],
    }
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return out
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        name = tgt.id
        if name in ANCHOR_TUPLE_NAMES:
            tup = _tuple_str_elements(node.value)
            if tup is not None:
                out["anchor_tuples"].append({"name": name, "value": tup})
        elif name in SUBSET_TUPLE_NAMES:
            tup = _tuple_str_elements(node.value)
            if tup is not None:
                out["subset_tuples"].append({"name": name, "value": tup})
        elif name in MB_DICT_NAMES:
            pairs = _dict_str_key_pairs(node.value)
            if pairs is not None:
                mapping: dict[str, float] = {}
                ok = True
                for k, v in pairs:
                    fv = _fold_float(v)
                    if fv is None:
                        ok = False
                        break
                    mapping[k] = fv
                if ok:
                    out["mb_dicts"].append({"name": name,
                                            "value": mapping})
        elif name in BYTES_DICT_NAMES:
            pairs = _dict_str_key_pairs(node.value)
            if pairs is not None:
                mapping_b: dict[str, int] = {}
                ok = True
                for k, v in pairs:
                    bv = _fold_int(v)
                    if bv is None:
                        ok = False
                        break
                    mapping_b[k] = bv
                if ok:
                    out["bytes_dicts"].append({"name": name,
                                               "value": mapping_b})
    return out


def _harvest_all() -> list[dict]:
    out: list[dict] = []
    for root in SOURCE_DIRS:
        if not root.exists():
            continue
        for p in sorted(root.glob("*.py")):
            rel = p.relative_to(ROOT).as_posix()
            if rel in SELF_SKIP:
                continue
            h = _harvest_file(p)
            if (h["anchor_tuples"] or h["mb_dicts"] or
                    h["bytes_dicts"] or h["subset_tuples"]):
                out.append(h)
    return out


# ----------------------------------------------------------- rules --

def _all_tokens(harvested: list[dict]) -> set[str]:
    tokens: set[str] = set()
    for h in harvested:
        for at in h["anchor_tuples"]:
            tokens.update(at["value"])
        for st in h["subset_tuples"]:
            tokens.update(st["value"])
        for md in h["mb_dicts"]:
            tokens.update(md["value"].keys())
        for bd in h["bytes_dicts"]:
            tokens.update(bd["value"].keys())
    return tokens


def _rule_l1(harvested: list[dict]) -> list[dict]:
    out: list[dict] = []
    canon = set(CANONICAL_L3_TIERS.keys())
    for h in harvested:
        for at in h["anchor_tuples"]:
            for tok in at["value"]:
                if tok not in canon:
                    out.append({"rule": "L1", "file": h["rel"],
                                "constant": at["name"], "token": tok,
                                "issue": "L3 token not in canonical registry"})
        for st in h["subset_tuples"]:
            for tok in st["value"]:
                if tok not in canon:
                    out.append({"rule": "L1", "file": h["rel"],
                                "constant": st["name"], "token": tok,
                                "issue": "L3 token not in canonical registry"})
        for md in h["mb_dicts"]:
            for tok in md["value"]:
                if tok not in canon:
                    out.append({"rule": "L1", "file": h["rel"],
                                "constant": md["name"], "token": tok,
                                "issue": "L3 token not in canonical registry"})
        for bd in h["bytes_dicts"]:
            for tok in bd["value"]:
                if tok not in canon:
                    out.append({"rule": "L1", "file": h["rel"],
                                "constant": bd["name"], "token": tok,
                                "issue": "L3 token not in canonical registry"})
    return out


def _rule_l2(harvested: list[dict]) -> list[dict]:
    out: list[dict] = []
    for h in harvested:
        for at in h["anchor_tuples"]:
            if tuple(at["value"]) != ANCHOR_TRIPLET:
                out.append({"rule": "L2", "file": h["rel"],
                            "constant": at["name"],
                            "expected": list(ANCHOR_TRIPLET),
                            "got":      list(at["value"]),
                            "issue": "anchor tuple does not match "
                                     "ANCHOR_TRIPLET"})
    return out


def _rule_l3(harvested: list[dict]) -> list[dict]:
    out: list[dict] = []
    for h in harvested:
        for md in h["mb_dicts"]:
            for tok, val in md["value"].items():
                if tok not in CANONICAL_L3_TIERS:
                    continue
                expected = CANONICAL_L3_TIERS[tok]["mb"]
                if abs(val - expected) > 1e-9:
                    out.append({"rule": "L3", "file": h["rel"],
                                "constant": md["name"], "token": tok,
                                "expected_mb": expected,
                                "got_mb":      val,
                                "issue": "MB scaling does not match "
                                         "canonical bytes/(1024*1024)"})
    return out


def _rule_l4(harvested: list[dict]) -> list[dict]:
    out: list[dict] = []
    for h in harvested:
        for bd in h["bytes_dicts"]:
            for tok, val in bd["value"].items():
                if tok not in CANONICAL_L3_TIERS:
                    continue
                expected = CANONICAL_L3_TIERS[tok]["bytes"]
                if val != expected:
                    out.append({"rule": "L4", "file": h["rel"],
                                "constant": bd["name"], "token": tok,
                                "expected_bytes": expected,
                                "got_bytes":      val,
                                "issue": "byte count does not match "
                                         "canonical"})
    return out


def _rule_l5(_harvested: list[dict]) -> list[dict]:
    out: list[dict] = []
    for tok, info in CANONICAL_L3_TIERS.items():
        if info.get("role") not in VALID_ROLES:
            out.append({"rule": "L5", "token": tok,
                        "role":  info.get("role"),
                        "issue": "canonical entry has invalid role"})
        if info.get("sub_tier") not in VALID_SUB_TIERS:
            out.append({"rule": "L5", "token": tok,
                        "sub_tier": info.get("sub_tier"),
                        "issue": "canonical entry has invalid sub_tier"})
    return out


def _rule_l6(harvested: list[dict]) -> list[dict]:
    out: list[dict] = []
    # Collect every token that appears in ANY anchor tuple harvested.
    seen_in_anchor: set[str] = set()
    for h in harvested:
        for at in h["anchor_tuples"]:
            seen_in_anchor.update(at["value"])
    for tok, info in CANONICAL_L3_TIERS.items():
        if info["role"] == "anchor" and tok not in seen_in_anchor:
            out.append({"rule": "L6", "token": tok,
                        "issue": "canonical anchor token never appears "
                                 "in any harvested PAPER_L3 tuple"})
    return out


def _rule_l7(harvested: list[dict]) -> list[dict]:
    out: list[dict] = []
    # Pairwise anchor-tuple disagreement
    by_name: dict[str, list[tuple[str, tuple[str, ...]]]] = {}
    for h in harvested:
        for at in h["anchor_tuples"]:
            by_name.setdefault(at["name"], []).append(
                (h["rel"], tuple(at["value"])))
    for name, instances in by_name.items():
        values = {v for _, v in instances}
        if len(values) > 1:
            out.append({"rule": "L7", "constant": name,
                        "distinct_values": [list(v) for v in values],
                        "instances": [
                            {"file": r, "value": list(v)}
                            for r, v in instances
                        ],
                        "issue": "harvested PAPER_L3-shaped constants "
                                 "disagree across files"})
    return out


# ----------------------------------------------------------- audit --

def audit() -> dict:
    harvested = _harvest_all()
    violations: list[dict] = []
    for fn in (_rule_l1, _rule_l2, _rule_l3, _rule_l4,
               _rule_l5, _rule_l6, _rule_l7):
        violations.extend(fn(harvested))

    # Tallies
    files = len(harvested)
    anchor_count = sum(len(h["anchor_tuples"]) for h in harvested)
    mb_count = sum(len(h["mb_dicts"]) for h in harvested)
    bytes_count = sum(len(h["bytes_dicts"]) for h in harvested)
    subset_count = sum(len(h["subset_tuples"]) for h in harvested)
    tokens_seen = sorted(_all_tokens(harvested))

    return {
        "status": "active",
        "rules": {
            "L1": "every harvested L3 token is in canonical registry",
            "L2": "every PAPER_L3 / PAPER_L3_SIZES tuple == ANCHOR_TRIPLET",
            "L3": "every L3_MB dict pairs token with canonical MB",
            "L4": "every L3_BYTES dict pairs token with canonical bytes",
            "L5": "every canonical entry has valid role + sub_tier",
            "L6": "every canonical anchor token appears in some "
                  "harvested PAPER_L3 tuple",
            "L7": "harvested PAPER_L3-shaped constants agree across "
                  "files (no two files disagree)",
        },
        "anchor_triplet": list(ANCHOR_TRIPLET),
        "canonical":      CANONICAL_L3_TIERS,
        "harvested":      harvested,
        "totals": {
            "canonical_size":   len(CANONICAL_L3_TIERS),
            "files_with_l3":    files,
            "anchor_tuples":    anchor_count,
            "mb_dicts":         mb_count,
            "bytes_dicts":      bytes_count,
            "subset_tuples":    subset_count,
            "tokens_seen":      len(tokens_seen),
            "violations":       len(violations),
        },
        "tokens_seen": tokens_seen,
        "violations":  violations,
    }


# ----------------------------------------------------------- writers --

def _render_md(audit: dict) -> str:
    L: list[str] = []
    L.append("# L3 cache-size registry (gate 251)")
    L.append("")
    t = audit["totals"]
    L.append(f"**Status:** {audit['status']}  •  "
             f"canonical: {t['canonical_size']}  •  "
             f"files: {t['files_with_l3']}  •  "
             f"PAPER_L3-shaped tuples: {t['anchor_tuples']}  •  "
             f"L3_MB dicts: {t['mb_dicts']}  •  "
             f"L3_BYTES dicts: {t['bytes_dicts']}  •  "
             f"tokens seen: {t['tokens_seen']}  •  "
             f"violations: {t['violations']}")
    L.append("")
    L.append(f"**Anchor triplet:** "
             f"`{ '`, `'.join(audit['anchor_triplet']) }`")
    L.append("")
    L.append("## Rules")
    for k, v in audit.get("rules", {}).items():
        L.append(f"- **{k}** — {v}")
    L.append("")
    L.append("## Canonical registry")
    L.append("")
    L.append("| token | bytes | mb | role | sub_tier |")
    L.append("|---|---:|---:|---|---|")
    for tok, info in audit["canonical"].items():
        L.append(f"| `{tok}` | {info['bytes']:,} | "
                 f"{info['mb']:.4g} | `{info['role']}` "
                 f"| `{info['sub_tier']}` |")
    L.append("")
    L.append("## Harvested constants (per file)")
    L.append("")
    L.append("| file | PAPER_L3 tuples | L3_MB dicts | L3_BYTES dicts | subset tuples |")
    L.append("|---|---:|---:|---:|---:|")
    for h in audit["harvested"]:
        L.append(f"| `{h['rel']}` | {len(h['anchor_tuples'])} "
                 f"| {len(h['mb_dicts'])} | {len(h['bytes_dicts'])} "
                 f"| {len(h['subset_tuples'])} |")
    L.append("")
    if audit.get("violations"):
        L.append("## Violations")
        for v in audit["violations"]:
            L.append(f"- {v}")
    else:
        L.append("**0 violations** — every harvested L3 cache-size "
                 "constant agrees with the canonical registry on "
                 "tokens, byte counts, MB scaling, and anchor "
                 "ordering.")
    return "\n".join(L) + "\n"


def _render_csv(audit: dict) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["field", "value"])
    t = audit["totals"]
    for k in ("canonical_size", "files_with_l3", "anchor_tuples",
              "mb_dicts", "bytes_dicts", "subset_tuples",
              "tokens_seen", "violations"):
        w.writerow([k, t[k]])
    w.writerow(["anchor_triplet",
                "|".join(audit["anchor_triplet"])])
    return buf.getvalue()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--json-out", required=True)
    p.add_argument("--md-out",   required=True)
    p.add_argument("--csv-out",  required=True)
    args = p.parse_args()
    a = audit()
    Path(args.json_out).write_text(
        json.dumps(a, indent=2, sort_keys=True) + "\n")
    Path(args.md_out).write_text(_render_md(a))
    Path(args.csv_out).write_text(_render_csv(a))
    print(f"[lit-faith-l3-registry] status={a['status']} "
          f"canonical={a['totals']['canonical_size']} "
          f"files={a['totals']['files_with_l3']} "
          f"tuples={a['totals']['anchor_tuples']} "
          f"violations={a['totals']['violations']}")


if __name__ == "__main__":
    main()
