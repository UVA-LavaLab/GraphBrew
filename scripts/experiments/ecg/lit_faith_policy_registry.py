#!/usr/bin/env python3
"""Gate 255 — cache-policy vocabulary registry.

The codebase ships >20 modules that each declare their own
``POLICIES`` tuple (`("LRU", "SRRIP", "GRASP", "POPT")`, etc.).
Today every one of those tuples is a subset of the
``CANONICAL_POLICY_NAMES`` set, but no test fails if a future
contributor adds a 25th module that re-declares
``POLICIES = ("Lru", "SRIP", "Grasp")`` and silently misspells a
policy name. Once that happens, the cross-tool reports, the
per-policy aggregations, and the per-app slope tables stop
comparing apples to apples — and the paper text drifts from the
data without any gate firing.

This gate codifies the cache-replacement-policy name universe
with a hand-curated ``CANONICAL_POLICY_NAMES`` map (token →
{family, aliases, paper_label, role}) plus the four-policy
default tuple ``CANONICAL_FOUR_TUPLE = ("LRU", "SRRIP", "GRASP",
"POPT")`` and the three-policy gem5/Sniper anchor tuple
``ANCHOR_TRIPLET = ("GRASP", "LRU", "SRRIP")``. It then
AST-harvests every module-level ``POLICIES``-shaped tuple/list
literal in ``scripts/experiments/ecg/`` and ``scripts/test/`` and
asserts:

* P1 every token used as an element in any harvested constant
  appears in ``CANONICAL_POLICY_NAMES`` (no rogue policy names);
* P2 every harvested ``POLICIES`` tuple has elements drawn from
  the canonical set AND has no duplicates;
* P3 every ``ALL_POLICIES``-shaped constant is exactly
  ``BASELINE_POLICIES + GRAPH_AWARE_POLICIES`` per the
  authoritative declaration in
  ``scripts/experiments/ecg/config.py``;
* P4 every canonical token has a valid ``family`` from
  {baseline, graph_aware} and a non-empty ``paper_label``
  (every policy must be presentable in paper figures);
* P5 no two canonical tokens share the same ``paper_label``
  (paper figures cannot have label collisions);
* P6 every harvested four-tuple is a permutation of
  ``CANONICAL_FOUR_TUPLE`` (LRU+SRRIP+GRASP+POPT); aside from
  ordering, no module ships a four-tuple that swaps in a
  different policy;
* P7 every harvested three-tuple matches ``ANCHOR_TRIPLET``
  set-wise (i.e. {GRASP,LRU,SRRIP}) for gem5/Sniper anchor
  contexts, or is a subset of CANONICAL when the surrounding
  module is non-anchor;
* P8 the canonical set is closed: every alias listed in any
  ``aliases`` field is rejected (aliases document forbidden
  spellings; if a harvested token equals an alias it's a
  P1 violation).
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
SOURCE_DIRS = [
    ROOT / "scripts" / "experiments" / "ecg",
    ROOT / "scripts" / "test",
]

# Each entry locks one cache-policy token.
#   family       : 'baseline' (LRU/FIFO/RANDOM/LFU/SRRIP) |
#                  'graph_aware' (GRASP/POPT/ECG)
#   paper_label  : human-readable label used in paper figures
#   aliases      : forbidden misspellings / case-variants that
#                  would silently break per-policy aggregations
CANONICAL_POLICY_NAMES: dict[str, dict[str, Any]] = {
    "LRU":    {"family": "baseline",
               "paper_label": "LRU",
               "aliases": ["Lru", "lru", "L.R.U", "LRU_cache"]},
    "FIFO":   {"family": "baseline",
               "paper_label": "FIFO",
               "aliases": ["Fifo", "fifo"]},
    "RANDOM": {"family": "baseline",
               "paper_label": "Random",
               "aliases": ["Random", "RAND", "RND"]},
    "LFU":    {"family": "baseline",
               "paper_label": "LFU",
               "aliases": ["Lfu", "lfu"]},
    "SRRIP":  {"family": "baseline",
               "paper_label": "SRRIP",
               "aliases": ["Srrip", "srrip", "SRIP", "S-RRIP"]},
    "GRASP":  {"family": "graph_aware",
               "paper_label": "GRASP",
               "aliases": ["Grasp", "grasp", "G.R.A.S.P"]},
    "POPT":   {"family": "graph_aware",
               "paper_label": "P-OPT",
               "aliases": ["Popt", "popt", "P_OPT"]},
    "ECG":    {"family": "graph_aware",
               "paper_label": "ECG",
               "aliases": ["Ecg", "ecg"]},
}

# ECG ablation arms shipped by roi_matrix.py. These are NOT
# stand-alone policies — they are operational variants of the ECG
# graph-aware policy used for the ECG-substrate ablation matrix
# (gate 238 + descendants). Each arm has a documented purpose so
# new arms can't silently appear without bumping this map.
CANONICAL_ECG_ARMS: dict[str, dict[str, Any]] = {
    "POPT_CHARGED": {
        "parent": "POPT",
        "purpose": "POPT with the charged-overhead accounting "
                   "model active (ROI substrate ablation arm)."},
    "ECG:DBG_ONLY": {
        "parent": "ECG",
        "purpose": "Debug-only ECG arm — emits diagnostic "
                   "counters; not a production policy."},
    "ECG:DBG_PRIMARY_CHARGED": {
        "parent": "ECG",
        "purpose": "Debug primary with charged-overhead accounting."},
    "ECG:DBG_PRIMARY": {
        "parent": "ECG",
        "purpose": "Debug primary ECG arm (uncharged baseline)."},
    "ECG:POPT_TIE": {
        "parent": "ECG",
        "purpose": "ECG with POPT tie-breaking on equal scores."},
    "ECG:POPT_PRIMARY": {
        "parent": "ECG",
        "purpose": "ECG with POPT as the primary scorer "
                   "(gate-239 parity arm)."},
    "ECG:ECG_EMBEDDED": {
        "parent": "ECG",
        "purpose": "ECG embedded inside the L3 substrate "
                   "directly."},
    "ECG:ECG_EPOCH_EMBEDDED": {
        "parent": "ECG",
        "purpose": "ECG embedded with epoch-bounded retraining."},
    "ECG:ECG_COMBINED": {
        "parent": "ECG",
        "purpose": "ECG combining multiple sub-scorers into one "
                   "decision."},
}

CANONICAL_FOUR_TUPLE: tuple[str, ...] = ("LRU", "SRRIP", "GRASP", "POPT")
ANCHOR_TRIPLET: tuple[str, ...] = ("GRASP", "LRU", "SRRIP")

VALID_FAMILIES = {"baseline", "graph_aware"}

# Names of module-level constants the harvester treats as
# policy-shaped.
POLICIES_TUPLE_NAMES = {"POLICIES"}
ALL_POLICIES_NAMES = {"ALL_POLICIES"}
BASELINE_NAMES = {"BASELINE_POLICIES"}
GRAPH_AWARE_NAMES = {"GRAPH_AWARE_POLICIES"}

# config.py is the authoritative declaration; harvest values from
# it as ground truth for P3.
CONFIG_PATH = ROOT / "scripts" / "experiments" / "ecg" / "config.py"

# Self-skip — this file declares the canonical map; naïve
# harvesting would pick up CANONICAL_POLICY_NAMES keys as a
# "third-party POLICIES tuple".
SELF_SKIP = {
    Path(__file__).resolve().relative_to(ROOT).as_posix(),
}


# --- harvester -------------------------------------------------------

def _is_string_literal_collection(node: ast.AST) -> bool:
    if not isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return False
    return all(
        isinstance(e, ast.Constant) and isinstance(e.value, str)
        for e in node.elts
    )


def _string_collection_values(node: ast.AST) -> list[str]:
    return [e.value for e in node.elts]  # type: ignore[attr-defined]


def _harvest_file(path: Path) -> dict[str, list[Any]]:
    """Return {constant_name: [(rel_path, values)...]} for every
    module-level POLICIES-shaped assignment in `path`."""
    out: dict[str, list[Any]] = {
        "POLICIES": [],
        "ALL_POLICIES": [],
        "BASELINE_POLICIES": [],
        "GRAPH_AWARE_POLICIES": [],
    }
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return out
    rel = path.resolve().relative_to(ROOT).as_posix()
    if rel in SELF_SKIP:
        return out
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Name):
                continue
            name = target.id
            if name in out and _is_string_literal_collection(node.value):
                out[name].append((rel, tuple(_string_collection_values(node.value))))
    return out


def _harvest_all() -> dict[str, list[Any]]:
    aggregate: dict[str, list[Any]] = {
        "POLICIES": [],
        "ALL_POLICIES": [],
        "BASELINE_POLICIES": [],
        "GRAPH_AWARE_POLICIES": [],
    }
    for d in SOURCE_DIRS:
        for p in sorted(d.glob("*.py")):
            chunk = _harvest_file(p)
            for k, v in chunk.items():
                aggregate[k].extend(v)
    return aggregate


# --- rules -----------------------------------------------------------

def _rule_p1(harvest: dict[str, list[Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    canonical = set(CANONICAL_POLICY_NAMES) | set(CANONICAL_ECG_ARMS)
    for kind, hits in harvest.items():
        for rel, values in hits:
            for v in values:
                if v not in canonical:
                    out.append({
                        "rule": "P1",
                        "file": rel,
                        "constant": kind,
                        "token": v,
                        "msg": f"token {v!r} not in CANONICAL_POLICY_NAMES "
                               "or CANONICAL_ECG_ARMS",
                    })
    return out


def _rule_p2(harvest: dict[str, list[Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for rel, values in harvest["POLICIES"]:
        if len(set(values)) != len(values):
            out.append({
                "rule": "P2",
                "file": rel,
                "constant": "POLICIES",
                "values": list(values),
                "msg": "POLICIES tuple contains duplicates",
            })
    return out


def _rule_p3(harvest: dict[str, list[Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    config_baseline = None
    config_graph_aware = None
    for rel, values in harvest["BASELINE_POLICIES"]:
        if rel.endswith("config.py"):
            config_baseline = tuple(values)
    for rel, values in harvest["GRAPH_AWARE_POLICIES"]:
        if rel.endswith("config.py"):
            config_graph_aware = tuple(values)
    if config_baseline is None or config_graph_aware is None:
        out.append({
            "rule": "P3",
            "msg": "config.py missing BASELINE_POLICIES or "
                   "GRAPH_AWARE_POLICIES",
        })
        return out
    canonical_simple = config_baseline + config_graph_aware
    canonical_set = set(CANONICAL_POLICY_NAMES) | set(CANONICAL_ECG_ARMS)
    for rel, values in harvest["ALL_POLICIES"]:
        actual = tuple(values)
        if rel.endswith("config.py"):
            # config.py is the simple form — locked exactly.
            if actual != canonical_simple:
                out.append({
                    "rule": "P3",
                    "file": rel,
                    "actual": list(actual),
                    "expected": list(canonical_simple),
                    "msg": "config.py ALL_POLICIES != "
                           "BASELINE_POLICIES + GRAPH_AWARE_POLICIES",
                })
        else:
            # Extended ALL_POLICIES (e.g. roi_matrix.py) — every token
            # must be a known canonical name OR ECG arm.
            stray = [v for v in actual if v not in canonical_set]
            if stray:
                out.append({
                    "rule": "P3",
                    "file": rel,
                    "stray": stray,
                    "msg": "extended ALL_POLICIES contains tokens that "
                           "are neither canonical nor documented ECG arms",
                })
    return out


def _rule_p4() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for token, meta in CANONICAL_POLICY_NAMES.items():
        if meta.get("family") not in VALID_FAMILIES:
            out.append({
                "rule": "P4",
                "token": token,
                "family": meta.get("family"),
                "msg": f"invalid family (must be in {sorted(VALID_FAMILIES)})",
            })
        label = meta.get("paper_label", "")
        if not isinstance(label, str) or not label.strip():
            out.append({
                "rule": "P4",
                "token": token,
                "paper_label": label,
                "msg": "missing/empty paper_label",
            })
    return out


def _rule_p5() -> list[dict[str, Any]]:
    seen: dict[str, list[str]] = {}
    for token, meta in CANONICAL_POLICY_NAMES.items():
        seen.setdefault(meta.get("paper_label", ""), []).append(token)
    return [
        {
            "rule": "P5",
            "paper_label": k,
            "tokens": v,
            "msg": f"paper_label {k!r} claimed by {len(v)} canonical tokens",
        }
        for k, v in seen.items() if len(v) > 1
    ]


def _rule_p6(harvest: dict[str, list[Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    expected = set(CANONICAL_FOUR_TUPLE)
    for rel, values in harvest["POLICIES"]:
        if len(values) != 4:
            continue
        if set(values) != expected:
            out.append({
                "rule": "P6",
                "file": rel,
                "actual": list(values),
                "expected_set": sorted(expected),
                "msg": "4-tuple POLICIES is not a permutation of CANONICAL_FOUR_TUPLE",
            })
    return out


def _rule_p7(harvest: dict[str, list[Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    canonical = set(CANONICAL_POLICY_NAMES)
    for rel, values in harvest["POLICIES"]:
        if len(values) != 3:
            continue
        if not set(values).issubset(canonical):
            out.append({
                "rule": "P7",
                "file": rel,
                "actual": list(values),
                "msg": "3-tuple POLICIES contains non-canonical token "
                       "(should be a subset of CANONICAL)",
            })
    return out


def _rule_p8(harvest: dict[str, list[Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    aliases: set[str] = set()
    for meta in CANONICAL_POLICY_NAMES.values():
        aliases.update(meta.get("aliases", []))
    for kind, hits in harvest.items():
        for rel, values in hits:
            for v in values:
                if v in aliases:
                    out.append({
                        "rule": "P8",
                        "file": rel,
                        "constant": kind,
                        "token": v,
                        "msg": f"token {v!r} is a documented forbidden "
                               "alias of a canonical policy",
                    })
    return out


def _rule_p9() -> list[dict[str, Any]]:
    """Every ECG-arm parent must be a real canonical policy."""
    out: list[dict[str, Any]] = []
    canonical = set(CANONICAL_POLICY_NAMES)
    for arm, meta in CANONICAL_ECG_ARMS.items():
        parent = meta.get("parent")
        if parent not in canonical:
            out.append({
                "rule": "P9",
                "arm": arm,
                "parent": parent,
                "msg": "ECG arm references unknown parent policy",
            })
        if not meta.get("purpose", "").strip():
            out.append({
                "rule": "P9",
                "arm": arm,
                "msg": "ECG arm missing/empty purpose",
            })
    return out


# --- driver ----------------------------------------------------------

def audit() -> dict[str, Any]:
    harvest = _harvest_all()
    violations: list[dict[str, Any]] = []
    violations.extend(_rule_p1(harvest))
    violations.extend(_rule_p2(harvest))
    violations.extend(_rule_p3(harvest))
    violations.extend(_rule_p4())
    violations.extend(_rule_p5())
    violations.extend(_rule_p6(harvest))
    violations.extend(_rule_p7(harvest))
    violations.extend(_rule_p8(harvest))
    violations.extend(_rule_p9())

    return {
        "status": "active",
        "rules": {
            "P1": "every harvested policy token is in CANONICAL_POLICY_NAMES or CANONICAL_ECG_ARMS",
            "P2": "POLICIES tuples have no duplicates",
            "P3": "config.py ALL_POLICIES == BASELINE_POLICIES + GRAPH_AWARE_POLICIES; extended ALL_POLICIES only adds canonical/arm tokens",
            "P4": "every canonical token has a valid family + non-empty paper_label",
            "P5": "no two canonical tokens share the same paper_label",
            "P6": "every harvested 4-tuple POLICIES is a permutation of CANONICAL_FOUR_TUPLE",
            "P7": "every harvested 3-tuple POLICIES is a subset of CANONICAL",
            "P8": "no harvested token is a documented forbidden alias",
            "P9": "every CANONICAL_ECG_ARMS entry has a real parent + non-empty purpose",
        },
        "canonical_four_tuple": list(CANONICAL_FOUR_TUPLE),
        "anchor_triplet":       list(ANCHOR_TRIPLET),
        "totals": {
            "canonical_tokens":    len(CANONICAL_POLICY_NAMES),
            "ecg_arms":            len(CANONICAL_ECG_ARMS),
            "harvested_POLICIES":  len(harvest["POLICIES"]),
            "harvested_ALL":       len(harvest["ALL_POLICIES"]),
            "harvested_BASELINE":  len(harvest["BASELINE_POLICIES"]),
            "harvested_GRAPH":     len(harvest["GRAPH_AWARE_POLICIES"]),
            "violations":          len(violations),
        },
        "canonical_tokens": sorted(CANONICAL_POLICY_NAMES.keys()),
        "ecg_arms":         sorted(CANONICAL_ECG_ARMS.keys()),
        "violations":       violations,
    }


def write_outputs(data: dict[str, Any], json_out: Path | None,
                  md_out: Path | None, csv_out: Path | None) -> None:
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if md_out:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append("# cache-policy vocabulary registry — gate 255")
        lines.append("")
        lines.append(f"Status: `{data['status']}`")
        lines.append("")
        t = data["totals"]
        lines.append(
            f"Totals: canonical={t['canonical_tokens']}  "
            f"harvested_POLICIES={t['harvested_POLICIES']}  "
            f"harvested_ALL={t['harvested_ALL']}  "
            f"violations={t['violations']}")
        lines.append("")
        lines.append(
            f"Canonical four-tuple: `{tuple(data['canonical_four_tuple'])}`")
        lines.append(f"Anchor triplet: `{tuple(data['anchor_triplet'])}`")
        lines.append("")
        lines.append("## Canonical tokens")
        lines.append("")
        for t_ in data["canonical_tokens"]:
            meta = CANONICAL_POLICY_NAMES[t_]
            lines.append(f"- `{t_}` — family=`{meta['family']}` "
                         f"paper_label=`{meta['paper_label']}` "
                         f"aliases={meta['aliases']}")
        lines.append("")
        lines.append("## ECG arms")
        lines.append("")
        for arm in sorted(CANONICAL_ECG_ARMS):
            meta = CANONICAL_ECG_ARMS[arm]
            lines.append(f"- `{arm}` — parent=`{meta['parent']}` — "
                         f"{meta['purpose']}")
        lines.append("")
        if data["violations"]:
            lines.append("## Violations")
            lines.append("")
            for v in data["violations"][:50]:
                lines.append(f"- {v}")
            lines.append("")
        md_out.write_text("\n".join(lines))
    if csv_out:
        csv_out.parent.mkdir(parents=True, exist_ok=True)
        with csv_out.open("w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["metric", "value"])
            for k, v in data["totals"].items():
                w.writerow([k, v])
            w.writerow(["canonical_four_tuple",
                        "|".join(data["canonical_four_tuple"])])
            w.writerow(["anchor_triplet",
                        "|".join(data["anchor_triplet"])])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--md-out", type=Path, default=None)
    ap.add_argument("--csv-out", type=Path, default=None)
    args = ap.parse_args()
    a = audit()
    write_outputs(a, args.json_out, args.md_out, args.csv_out)
    print(
        f"[lit-faith-policy-registry] status={a['status']} "
        f"canonical={a['totals']['canonical_tokens']} "
        f"POLICIES={a['totals']['harvested_POLICIES']} "
        f"ALL={a['totals']['harvested_ALL']} "
        f"violations={a['totals']['violations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
