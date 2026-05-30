#!/usr/bin/env python3
"""Gate 249 — graph-family map full-coverage audit.

Many GraphBrew analysis modules each carry a private copy of the
``GRAPH_FAMILY`` (or analogous) dict mapping a graph name to its
family label (``social`` / ``web`` / ``citation`` / ``road`` /
``mesh``). Gate 107 (test_graph_family_map_duplication) already locks
the topology of 7 known copies (2 full + 5 short). This gate hardens
that by:

  * harvesting *every* module-level dict literal in
    ``scripts/experiments/ecg/`` and ``scripts/test/`` whose keys are
    known graph names and whose values are known family labels, and
  * asserting that every such copy agrees with the canonical map
    declared in this generator on every shared key.

That catches new modules added by a future contributor that ship
their own GRAPH_FAMILY copy and silently diverge.

Source-of-truth (two halves on disk):

  1. Every ``.py`` file under ``scripts/experiments/ecg/`` and
     ``scripts/test/`` (excluding this generator and its pytest).
  2. ``CANONICAL_GRAPH_FAMILY`` declared in this generator — the
     single authoritative map.

Rules:

  F1 — every module-level dict literal whose keys are known graph
       names and whose values are known family labels is harvested;
  F2 — every harvested copy is a subset of the canonical map
       (no unknown graph→family pairs);
  F3 — every harvested copy agrees with the canonical map on every
       shared key (no value drift);
  F4 — the canonical map is non-empty AND every value comes from
       the documented family allow-list;
  F5 — the existing gate-107 dup test covers ALL "FULL" copies
       (no unguarded full copies that include reserved-for-future
       graph tags);
  F6 — every harvested copy that is NOT in the gate-107 universe
       is either (a) a strict subset of the canonical map or
       (b) flagged here for tracking.
"""
from __future__ import annotations

import argparse
import ast
import csv
import io
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2].parent


# ----------------------------------------------------------- registry --

CANONICAL_GRAPH_FAMILY: dict[str, str] = {
    # currently-shipped corpus (8 graphs across 5 families)
    "email-Eu-core":    "social",
    "soc-pokec":        "social",
    "soc-LiveJournal1": "social",
    "com-orkut":        "social",
    "web-Google":       "web",
    "cit-Patents":      "citation",
    "roadNet-CA":       "road",
    "delaunay_n19":     "mesh",
}

# Reserved-for-future graph tags that legitimately appear ONLY in
# the gate-107 "FULL" copies. These are not violations.
RESERVED_FUTURE_KEYS: dict[str, str] = {
    "soc-LJ":           "social",
    "kron21":           "mesh",
    "kron22":           "mesh",
    "kron23":           "mesh",
    "twitter7":         "social",
    "web-uk":           "web",
}

ALLOWED_FAMILIES: set[str] = {
    "social", "web", "citation", "road", "mesh",
}


KNOWN_GRAPH_NAMES: set[str] = (
    set(CANONICAL_GRAPH_FAMILY) | set(RESERVED_FUTURE_KEYS)
)


# Scan roots
SCAN_DIRS: list[str] = [
    "scripts/experiments/ecg",
    "scripts/test",
]

# Don't audit ourselves
SELF_SKIP: set[str] = {
    "scripts/experiments/ecg/lit_faith_graph_family.py",
    "scripts/test/test_lit_faith_graph_family.py",
}

# Gate-107 universe (already locked by another test)
GATE_107_UNIVERSE: set[str] = {
    "scripts/experiments/ecg/policy_winner_table.py",
    "scripts/test/test_corpus_diversity_floor.py",
    "scripts/experiments/ecg/literature_deviations_report.py",
    "scripts/experiments/ecg/oracle_gap_report.py",
    "scripts/experiments/ecg/winning_regime_taxonomy.py",
    "scripts/experiments/ecg/popt_vs_grasp_report.py",
    "scripts/experiments/ecg/family_saturation_distance.py",
}


# ----------------------------------------------------------- harvester --

def _is_graph_family_dict(node: ast.Dict) -> dict[str, str] | None:
    """If node is a {str: str} dict whose keys look like known graphs
    and values look like known families, return it as a plain dict."""
    if not node.keys or len(node.keys) < 2:
        return None
    out: dict[str, str] = {}
    for k, v in zip(node.keys, node.values):
        if not isinstance(k, ast.Constant) or not isinstance(k.value, str):
            return None
        if not isinstance(v, ast.Constant) or not isinstance(v.value, str):
            return None
        out[k.value] = v.value
    # signature: every key must be in known graph names AND every value
    # must be in the allowed families allow-list. This is intentionally
    # strict — keeps the harvester from grabbing unrelated str→str dicts.
    if not all(key in KNOWN_GRAPH_NAMES for key in out):
        return None
    if not all(val in ALLOWED_FAMILIES for val in out.values()):
        return None
    # at least one currently-shipped graph (don't grab pure-future stubs)
    if not any(key in CANONICAL_GRAPH_FAMILY for key in out):
        return None
    return out


def _harvest_file(path: Path) -> list[dict[str, str]]:
    """Walk a .py file and return all module-level GRAPH_FAMILY-shaped dicts."""
    try:
        tree = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return []
    copies: list[dict[str, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Dict):
            harvested = _is_graph_family_dict(node)
            if harvested is not None:
                copies.append(harvested)
    return copies


def _scan_for_copies() -> dict[str, list[dict[str, str]]]:
    """Return {relpath: [copies...]} for every .py with at least one copy."""
    result: dict[str, list[dict[str, str]]] = {}
    for dirname in SCAN_DIRS:
        d = ROOT / dirname
        if not d.is_dir():
            continue
        for path in sorted(d.rglob("*.py")):
            rel = str(path.relative_to(ROOT))
            if rel in SELF_SKIP:
                continue
            copies = _harvest_file(path)
            if copies:
                result[rel] = copies
    return result


# ----------------------------------------------------------- audit --

def audit() -> dict:
    violations: list[dict] = []

    # F4 — canonical map sanity
    if not CANONICAL_GRAPH_FAMILY:
        violations.append({"rule": "F4", "site": "(canonical)",
                           "detail": "CANONICAL_GRAPH_FAMILY is empty"})
    for graph, family in CANONICAL_GRAPH_FAMILY.items():
        if family not in ALLOWED_FAMILIES:
            violations.append({
                "rule": "F4", "site": "(canonical)",
                "detail": f"{graph}->{family} not in allow-list "
                          f"{sorted(ALLOWED_FAMILIES)}",
            })

    copies_by_file = _scan_for_copies()

    site_rows: list[dict] = []
    for rel in sorted(copies_by_file):
        copies = copies_by_file[rel]
        # tag each copy
        for idx, c in enumerate(copies):
            # F3 — agree with canonical on every shared key
            for graph, family in c.items():
                if (graph in CANONICAL_GRAPH_FAMILY
                        and CANONICAL_GRAPH_FAMILY[graph] != family):
                    violations.append({
                        "rule": "F3", "site": rel,
                        "detail": f"copy#{idx} {graph}->{family} "
                                  f"!= canonical {CANONICAL_GRAPH_FAMILY[graph]}",
                    })
            # F2 — subset check (no key outside KNOWN_GRAPH_NAMES, by
            # construction of the harvester; but check anyway)
            unknown = [k for k in c
                       if k not in CANONICAL_GRAPH_FAMILY
                       and k not in RESERVED_FUTURE_KEYS]
            if unknown:
                violations.append({
                    "rule": "F2", "site": rel,
                    "detail": f"copy#{idx} unknown graphs: {sorted(unknown)}",
                })
            site_rows.append({
                "site": rel,
                "copy_index": idx,
                "size": len(c),
                "has_reserved_future_keys": any(
                    k in RESERVED_FUTURE_KEYS for k in c),
                "is_in_gate_107_universe": rel in GATE_107_UNIVERSE,
            })

    # F5 — every FULL copy (has reserved-future keys) is in gate-107 universe
    for row in site_rows:
        if row["has_reserved_future_keys"] and not row["is_in_gate_107_universe"]:
            violations.append({
                "rule": "F5", "site": row["site"],
                "detail": f"copy#{row['copy_index']} is a FULL copy "
                          f"(has reserved-future keys) but is NOT in "
                          f"gate-107 universe — add to FULL_SOURCES",
            })

    # F6 — out-of-universe copies are subsets of canonical (tracked here)
    extras = [r for r in site_rows
              if not r["is_in_gate_107_universe"]]
    # nothing to assert beyond F2/F3 above — we surface the list in JSON

    return {
        "status": "active",
        "canonical_size": len(CANONICAL_GRAPH_FAMILY),
        "reserved_future_size": len(RESERVED_FUTURE_KEYS),
        "allowed_families": sorted(ALLOWED_FAMILIES),
        "files_scanned_dirs": SCAN_DIRS,
        "copy_count": len(site_rows),
        "files_with_copies": len(copies_by_file),
        "gate_107_universe_size": len(GATE_107_UNIVERSE),
        "out_of_universe_copies": [
            r["site"] for r in site_rows
            if not r["is_in_gate_107_universe"]
        ],
        "site_rows": site_rows,
        "violations": violations,
    }


# ----------------------------------------------------------- writers --

def _write_json(out: Path, data: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n")


def _write_md(out: Path, data: dict) -> None:
    buf = io.StringIO()
    buf.write("# Gate 249 — graph-family map full-coverage audit\n\n")
    buf.write(f"- status: **{data['status']}**\n")
    buf.write(f"- canonical map size: {data['canonical_size']}\n")
    buf.write(f"- reserved-future keys: {data['reserved_future_size']}\n")
    buf.write(f"- allowed families: {data['allowed_families']}\n")
    buf.write(f"- copies harvested: {data['copy_count']} "
              f"across {data['files_with_copies']} files\n")
    buf.write(f"- gate-107 universe size: {data['gate_107_universe_size']}\n")
    buf.write(f"- out-of-universe copies: "
              f"{len(data['out_of_universe_copies'])}\n")
    buf.write(f"- violations: {len(data['violations'])}\n\n")
    if data["out_of_universe_copies"]:
        buf.write("## Out-of-universe copies (currently subset-clean)\n\n")
        for s in data["out_of_universe_copies"]:
            buf.write(f"- `{s}`\n")
    buf.write("\n## Per-site\n\n")
    buf.write("| site | copy# | size | full? | in gate-107? |\n")
    buf.write("|---|---|---|---|---|\n")
    for r in data["site_rows"]:
        buf.write(f"| `{r['site']}` | {r['copy_index']} | {r['size']} | "
                  f"{r['has_reserved_future_keys']} | "
                  f"{r['is_in_gate_107_universe']} |\n")
    if data["violations"]:
        buf.write("\n## Violations\n\n")
        for v in data["violations"]:
            buf.write(f"- {v['rule']} {v['site']} — {v['detail']}\n")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(buf.getvalue())


def _write_csv(out: Path, data: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["site", "copy_index", "size",
                    "has_reserved_future_keys", "is_in_gate_107_universe"])
        for r in data["site_rows"]:
            w.writerow([
                r["site"], r["copy_index"], r["size"],
                int(r["has_reserved_future_keys"]),
                int(r["is_in_gate_107_universe"]),
            ])


# ----------------------------------------------------------- cli --

def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--json-out", type=Path, required=True)
    p.add_argument("--md-out",   type=Path, required=True)
    p.add_argument("--csv-out",  type=Path, required=True)
    args = p.parse_args()
    data = audit()
    _write_json(args.json_out, data)
    _write_md(args.md_out, data)
    _write_csv(args.csv_out, data)
    print(f"[lit-faith-graph-family] status={data['status']} "
          f"copies={data['copy_count']} files={data['files_with_copies']} "
          f"out_of_universe={len(data['out_of_universe_copies'])} "
          f"violations={len(data['violations'])}")
    return 0 if not data["violations"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
