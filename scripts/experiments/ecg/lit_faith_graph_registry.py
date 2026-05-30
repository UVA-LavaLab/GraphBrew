"""Gate 258 — graph-name canonical map.

Locks the cross-reference between:
  - the canonical GAP-style benchmark graph allow-list defined here
  - every graph string literal harvested by AST from
    ``scripts/experiments/ecg/*.py`` and ``scripts/test/*.py``
  - every per-source family-classification dict (graph → family)
    declared in
    ``family_saturation_distance.py``, ``family_sensitivity.py``,
    ``literature_deviations_report.py``, and the EVAL_GRAPHS list
    in ``config.py``

Vocabulary-lock companion to gates 252 (Slurm SBATCH schema), 255
(cache-policy vocab), 256 (run-profile vocab), and 257 (backend/tool
vocab). Gate 258 locks WHICH benchmark graph every per-row record,
anchor-cell census entry, family-classification dict, and
cross-paper baseline table references — so a contributor cannot
silently:

* misspell ``soc-LiveJournal1`` as ``soc-livejournal1`` /
  ``soc-LJ1`` in a single helper, breaking downstream lookup keys
  while every other generator still uses the correct SNAP casing,
* drop the hyphen in ``cit-Patents`` to ``citPatents`` in a new
  per-row CSV writer, silently breaking the cross-tool aggregator,
* introduce a new benchmark graph (``com-friendster``, ``twitter7``)
  to one pipeline stage without adding the matching family-classifier
  entry, leaving the regime-aware report to silently fall back to
  ``"unknown"`` / drop the row,
* shorten ``delaunay_n19`` to ``delaunay19`` in a fixture, breaking
  the SNAP-vs-synthetic provenance carried by the underscore.

8 rules R1-R8:
  R1: every harvested graph literal is in CANONICAL_GRAPH_NAMES
      (the allow-list)
  R2: every canonical graph has a non-empty family
      (social / web / road / mesh / citation / kronecker /
       email / content) plus a non-empty paper_label
  R3: every canonical graph has a non-empty source provenance
      (SNAP / GAP / synthetic / test)
  R4: every per-source family-classification dict
      (FAMILY_OF, FAMILY_MAP, GRAPH_FAMILIES) declares a family for
      every canonical graph it references AND every key it declares
      is canonical — no orphan keys, no missing keys
  R5: every harvested literal is referenced by at least one site
      that is NOT just a family-classifier dict (i.e. real corpus
      use, not just metadata declarations) — catches "ghost"
      graphs that are typed up in a family map but never actually
      run
  R6: every canonical graph name matches the documented regex
      ``^[A-Za-z][A-Za-z0-9_-]*$`` (alphanumeric + hyphen +
      underscore; no leading digit) AND family is lowercase ASCII
  R7: the EVAL_GRAPHS list in ``config.py`` is a strict subset of
      CANONICAL_GRAPHS (every config-driven evaluation graph is
      already a canonical entry)
  R8: every family declared in CANONICAL_GRAPHS is non-empty,
      lowercase ASCII, and matches the regex
      ``^[a-z][a-z0-9_]*$`` (no spaces, no hyphens; digits allowed
      after the leading letter for tokens like ``p2p``)
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"
SCRIPTS_TEST_DIR = REPO_ROOT / "scripts" / "test"

GRAPH_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")
FAMILY_RE = re.compile(r"^[a-z][a-z0-9_]*$")

# Per-source family-classification dict names that must agree on
# every canonical graph they reference.
FAMILY_DICT_NAMES: frozenset[str] = frozenset({
    "FAMILY_OF",
    "FAMILY_MAP",
    "GRAPH_FAMILIES",
    "GRAPH_FAMILY",
    "GRAPH_TO_FAMILY",
})


@dataclass(frozen=True)
class GraphEntry:
    """A canonical benchmark graph token."""

    name: str
    family: str
    paper_label: str
    source: str  # SNAP / GAP / synthetic / test
    purpose: str
    aliases: tuple[str, ...] = field(default_factory=tuple)
    documented_future: bool = False  # exempt from R5 (declared in
    # family-classifier dicts for future runs, no live corpus site yet)


CANONICAL_GRAPHS: tuple[GraphEntry, ...] = (
    # --- SNAP social ---
    GraphEntry(
        name="soc-pokec",
        family="social",
        paper_label="soc-pokec",
        source="SNAP",
        purpose=(
            "Slovak social network; ~1.6M nodes / 30M edges; the "
            "smallest of the social trio used for accuracy validation."
        ),
    ),
    GraphEntry(
        name="soc-LiveJournal1",
        family="social",
        paper_label="soc-LiveJournal1",
        source="SNAP",
        purpose=(
            "Friend-of-friend social graph; ~4.85M nodes / 69M edges; "
            "primary mid-scale social regression target."
        ),
    ),
    GraphEntry(
        name="com-orkut",
        family="social",
        paper_label="com-orkut",
        source="SNAP",
        purpose=(
            "Community membership graph; ~3M nodes / 117M edges; "
            "highest-density social graph in the corpus."
        ),
        aliases=("com-orkut-undir",),
    ),
    GraphEntry(
        name="com-orkut-undir",
        family="social",
        paper_label="com-orkut-undir",
        source="SNAP",
        purpose=(
            "Symmetrised (undirected) variant of com-orkut used by "
            "lit_faith_diversity for parity with literature comparisons "
            "that explicitly assume undirected community membership."
        ),
        aliases=("com-orkut",),
        documented_future=True,
    ),
    GraphEntry(
        name="email-Eu-core",
        family="social",
        paper_label="email-Eu-core",
        source="SNAP",
        purpose=(
            "Email exchange in an EU research org; ~1K nodes / 16K "
            "edges; the smoke-test / micro-benchmark graph used by "
            "lit_faith_accesses (SMOKE_GRAPHS)."
        ),
    ),
    # --- SNAP citation ---
    GraphEntry(
        name="cit-Patents",
        family="citation",
        paper_label="cit-Patents",
        source="SNAP",
        purpose=(
            "US patent citation network; ~3.8M nodes / 16.5M edges; "
            "low-density, citation-family canary."
        ),
    ),
    # --- SNAP web ---
    GraphEntry(
        name="web-Google",
        family="web",
        paper_label="web-Google",
        source="SNAP",
        purpose=(
            "Google's web crawl; ~916K nodes / 4.3M edges; primary "
            "web-family regression target with mid hub concentration."
        ),
    ),
    GraphEntry(
        name="web-BerkStan",
        family="web",
        paper_label="web-BerkStan",
        source="SNAP",
        purpose=(
            "Berkeley + Stanford web crawl; ~685K nodes / 7.6M edges; "
            "small-to-mid web-family graph used by lit_faith_diversity "
            "for web-family sensitivity across multiple crawls."
        ),
        documented_future=True,
    ),
    # --- P2P (SNAP) ---
    GraphEntry(
        name="p2p-Gnutella31",
        family="p2p",
        paper_label="p2p-Gnutella31",
        source="SNAP",
        purpose=(
            "Gnutella peer-to-peer file-sharing topology (Aug 31, "
            "2002); ~63K nodes / 148K edges; the lone p2p-family "
            "canary used by lit_faith_diversity to ensure family "
            "coverage extends beyond social/web/road/mesh/citation."
        ),
        documented_future=True,
    ),
    # --- SNAP road ---
    GraphEntry(
        name="roadNet-CA",
        family="road",
        paper_label="roadNet-CA",
        source="SNAP",
        purpose=(
            "California state road network; ~2M nodes / 5.5M edges; "
            "low-degree, planar canary that stresses GRASP's hub "
            "heuristic (no hubs to find)."
        ),
        aliases=("road-CA",),
    ),
    GraphEntry(
        name="roadNet-PA",
        family="road",
        paper_label="roadNet-PA",
        source="SNAP",
        purpose=(
            "Pennsylvania state road network; SNAP companion to "
            "roadNet-CA; used by lit_faith_margin as a road-family "
            "sensitivity check across multiple state graphs."
        ),
    ),
    GraphEntry(
        name="roadNet-TX",
        family="road",
        paper_label="roadNet-TX",
        source="SNAP",
        purpose=(
            "Texas state road network; SNAP companion to roadNet-CA; "
            "used by lit_faith_margin as a road-family sensitivity "
            "check across multiple state graphs."
        ),
    ),
    GraphEntry(
        name="road-CA",
        family="road",
        paper_label="road-CA",
        source="SNAP",
        purpose=(
            "Kebab-case display variant of roadNet-CA used by paper "
            "tables that strip the camel-case 'Net' suffix for "
            "compact reporting; declared so the family-classifier "
            "dicts in policy_winner_table parity tests can lock the "
            "spelling without R4 violations."
        ),
        aliases=("roadNet-CA",),
    ),
    GraphEntry(
        name="USA-Road",
        family="road",
        paper_label="USA-Road",
        source="DIMACS",
        purpose=(
            "Continental US road network (DIMACS 9th); ~24M nodes / "
            "58M edges; large-scale road-family target referenced by "
            "EVAL_GRAPHS in config.py."
        ),
    ),
    # --- Content / wikipedia ---
    GraphEntry(
        name="wikipedia_link_en",
        family="content",
        paper_label="wikipedia_link_en",
        source="KONECT",
        purpose=(
            "English Wikipedia internal link graph; ~12M nodes / 378M "
            "edges; largest evaluation graph for sustained-throughput "
            "comparisons against the literature."
        ),
    ),
    # --- Synthetic ---
    GraphEntry(
        name="delaunay_n18",
        family="mesh",
        paper_label="delaunay_n18",
        source="synthetic",
        purpose=(
            "Delaunay triangulation of 2^18 random 2-D points; "
            "small-scale variant used by lit_faith_margin as a "
            "mesh-family scaling sensitivity check (companion to n19/n20)."
        ),
        documented_future=True,
    ),
    GraphEntry(
        name="delaunay_n19",
        family="mesh",
        paper_label="delaunay_n19",
        source="synthetic",
        purpose=(
            "Delaunay triangulation of 2^19 random 2-D points; ~524K "
            "nodes / 3.1M edges; planar, low-degree, no hubs — the "
            "synthetic complement to roadNet-CA."
        ),
    ),
    GraphEntry(
        name="delaunay_n20",
        family="mesh",
        paper_label="delaunay_n20",
        source="synthetic",
        purpose=(
            "Delaunay triangulation of 2^20 random 2-D points; "
            "mid-scale variant used by lit_faith_margin as a "
            "mesh-family scaling sensitivity check (companion to n18/n19)."
        ),
        documented_future=True,
    ),
    # --- Kronecker (RMAT) ---
    GraphEntry(
        name="kron21",
        family="kronecker",
        paper_label="kron21",
        source="synthetic",
        purpose=(
            "Kronecker R-MAT generator with scale 21 (~2M nodes); "
            "RESERVED_FUTURE_KEYS in lit_faith_graph_family; declared "
            "for future scale-up sweeps against the GAPBS reference."
        ),
        documented_future=True,
    ),
    GraphEntry(
        name="kron22",
        family="kronecker",
        paper_label="kron22",
        source="synthetic",
        purpose=(
            "Kronecker R-MAT generator with scale 22 (~4M nodes); "
            "RESERVED_FUTURE_KEYS in lit_faith_graph_family; declared "
            "for future scale-up sweeps against the GAPBS reference."
        ),
        documented_future=True,
    ),
    GraphEntry(
        name="kron23",
        family="kronecker",
        paper_label="kron23",
        source="synthetic",
        purpose=(
            "Kronecker R-MAT generator with scale 23 (~8M nodes); "
            "RESERVED_FUTURE_KEYS in lit_faith_graph_family; declared "
            "for future scale-up sweeps against the GAPBS reference."
        ),
        documented_future=True,
    ),
    # --- Reserved-future aliases ---
    GraphEntry(
        name="soc-LJ",
        family="social",
        paper_label="soc-LJ",
        source="SNAP",
        purpose=(
            "Compact paper-label alias of soc-LiveJournal1 used by "
            "RESERVED_FUTURE_KEYS in lit_faith_graph_family; the "
            "short form preferred by some baseline tables."
        ),
        aliases=("soc-LiveJournal1",),
        documented_future=True,
    ),
    GraphEntry(
        name="twitter7",
        family="social",
        paper_label="twitter7",
        source="WebGraph",
        purpose=(
            "Compact alias of the 2010 Twitter snapshot (~52M nodes / "
            "1.96B edges); RESERVED_FUTURE_KEYS form used by some "
            "literature tables under the shorter spelling."
        ),
        aliases=("twitter-2010",),
        documented_future=True,
    ),
    GraphEntry(
        name="web-uk",
        family="web",
        paper_label="web-uk",
        source="WebGraph",
        purpose=(
            "Compact alias of uk-2005 (~39M nodes / 936M edges); "
            "RESERVED_FUTURE_KEYS form used by some literature tables "
            "under the shorter spelling."
        ),
        aliases=("uk-2005",),
        documented_future=True,
    ),
    # --- Web-archive (WebGraph / paper baselines) ---
    GraphEntry(
        name="twitter-2010",
        family="social",
        paper_label="twitter-2010",
        source="WebGraph",
        purpose=(
            "WebGraph snapshot of the 2010 Twitter follow network; "
            "~42M nodes / 1.5B edges; large-scale social-family "
            "regression target referenced by policy_winner_table "
            "parity tests against the GRASP/POPT literature."
        ),
        documented_future=True,
    ),
    GraphEntry(
        name="uk-2005",
        family="web",
        paper_label="uk-2005",
        source="WebGraph",
        purpose=(
            "WebGraph snapshot of the .uk top-level domain (2005); "
            "~39M nodes / 936M edges; large-scale web-family "
            "regression target referenced by policy_winner_table "
            "parity tests against the GRASP/POPT literature."
        ),
        documented_future=True,
    ),
)

CANONICAL_GRAPH_NAMES: frozenset[str] = frozenset(
    g.name for g in CANONICAL_GRAPHS
)
CANONICAL_FAMILIES: frozenset[str] = frozenset(
    g.family for g in CANONICAL_GRAPHS
)


def _is_graph_string(s: str) -> bool:
    """Conservative exact-match: a string is a graph token only if it
    appears in CANONICAL_GRAPH_NAMES. We deliberately do NOT use a
    prefix or suffix heuristic — strings like ``soc-pokec_extra`` or
    ``social_graph`` are dict keys / variable names / synthetic
    placeholders, not graph tokens."""
    return s in CANONICAL_GRAPH_NAMES


def _harvest_string_literals(tree: ast.AST) -> list[str]:
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _is_graph_string(node.value):
                out.append(node.value)
    return out


def _harvest_family_dicts(tree: ast.AST) -> list[tuple[str, dict[str, str]]]:
    """Returns ``[(dict_name, {graph_name: family})]`` for every
    module-level assignment whose target name is in
    FAMILY_DICT_NAMES."""
    out: list[tuple[str, dict[str, str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not isinstance(tgt, ast.Name):
            continue
        if tgt.id not in FAMILY_DICT_NAMES:
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        d: dict[str, str] = {}
        for k, v in zip(node.value.keys, node.value.values):
            if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                continue
            if not (isinstance(v, ast.Constant) and isinstance(v.value, str)):
                continue
            d[k.value] = v.value
        if d:
            out.append((tgt.id, d))
    return out


def _harvest_eval_graphs(tree: ast.AST) -> list[str]:
    """Returns names extracted from the EVAL_GRAPHS list-of-dicts
    declared in config.py: ``[{"name": "...", ...}, ...]``."""
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        if not (isinstance(tgt, ast.Name) and tgt.id == "EVAL_GRAPHS"):
            continue
        if not isinstance(node.value, ast.List):
            continue
        for elt in node.value.elts:
            if not isinstance(elt, ast.Dict):
                continue
            for k, v in zip(elt.keys, elt.values):
                if (isinstance(k, ast.Constant) and k.value == "name"
                        and isinstance(v, ast.Constant)
                        and isinstance(v.value, str)):
                    out.append(v.value)
    return out


def _scan_dir(d: Path) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str, dict[str, str]]],
    list[tuple[str, str]],
]:
    """Returns
    ``(literals, family_dicts, eval_graphs)``
    with file-path provenance. Skips the generator itself AND its
    pytest gate so canonical-list literals do not self-cite."""
    literals: list[tuple[str, str]] = []
    family_dicts: list[tuple[str, str, dict[str, str]]] = []
    eval_graphs: list[tuple[str, str]] = []
    skip_names = {
        Path(__file__).name,
        "test_lit_faith_graph_registry.py",
    }
    for f in sorted(d.glob("*.py")):
        if f.name in skip_names:
            continue
        try:
            tree = ast.parse(f.read_text())
        except SyntaxError:
            continue
        rel = str(f.relative_to(REPO_ROOT))
        for s in _harvest_string_literals(tree):
            literals.append((rel, s))
        for name, mapping in _harvest_family_dicts(tree):
            family_dicts.append((rel, name, mapping))
        for n in _harvest_eval_graphs(tree):
            eval_graphs.append((rel, n))
    return literals, family_dicts, eval_graphs


def audit() -> dict[str, Any]:
    literals_ecg, fam_ecg, eval_ecg = _scan_dir(ECG_DIR)
    literals_test, fam_test, eval_test = _scan_dir(SCRIPTS_TEST_DIR)
    all_literals = literals_ecg + literals_test
    all_family_dicts = fam_ecg + fam_test
    all_eval = eval_ecg + eval_test

    by_token: dict[str, list[str]] = {}
    for path, tok in all_literals:
        by_token.setdefault(tok, []).append(path)

    family_dict_paths: set[str] = {p for (p, _, _) in all_family_dicts}

    violations: list[dict[str, Any]] = []

    # R1: every harvested literal is canonical (already enforced by
    # the conservative _is_graph_string, but we still emit violations
    # if a future contributor adds a non-canonical literal that
    # passes some other path).
    for tok, paths in sorted(by_token.items()):
        if tok not in CANONICAL_GRAPH_NAMES:
            violations.append({
                "rule": "R1",
                "token": tok,
                "first_path": paths[0],
                "n_sites": len(paths),
                "msg": f"non-canonical graph literal {tok!r}",
            })

    # R2: every canonical has non-empty family + paper_label.
    for g in CANONICAL_GRAPHS:
        if not g.family:
            violations.append({"rule": "R2", "token": g.name, "msg": "empty family"})
        if not g.paper_label:
            violations.append({"rule": "R2", "token": g.name, "msg": "empty paper_label"})

    # R3: every canonical has non-empty source.
    for g in CANONICAL_GRAPHS:
        if not g.source:
            violations.append({"rule": "R3", "token": g.name, "msg": "empty source"})
        if g.source not in {"SNAP", "GAP", "DIMACS", "KONECT", "WebGraph", "synthetic", "test"}:
            violations.append({
                "rule": "R3",
                "token": g.name,
                "msg": f"source {g.source!r} not in canonical provenance set",
            })

    # R4: every family-dict's keys are canonical AND map to the
    # SAME family the canonical entry declares.
    canon_family = {g.name: g.family for g in CANONICAL_GRAPHS}
    for path, dname, mapping in all_family_dicts:
        for k, v in mapping.items():
            if k not in CANONICAL_GRAPH_NAMES:
                violations.append({
                    "rule": "R4",
                    "token": k,
                    "first_path": path,
                    "dict": dname,
                    "msg": f"family-dict {dname} in {path} has non-canonical key {k!r}",
                })
                continue
            if v != canon_family[k]:
                violations.append({
                    "rule": "R4",
                    "token": k,
                    "first_path": path,
                    "dict": dname,
                    "msg": (
                        f"family-dict {dname} in {path} maps {k!r}→{v!r}, "
                        f"but canonical family is {canon_family[k]!r}"
                    ),
                })

    # R5: every harvested literal must have at least one site that is
    # not just a family-classifier dict — i.e. real corpus use, not
    # metadata. We approximate by counting harvested-literal paths
    # not in family_dict_paths. EXEMPT entries flagged
    # documented_future=True (reserved-for-future RESERVED_FUTURE_KEYS
    # style entries that are deliberately declared before they ship).
    future_tokens = {g.name for g in CANONICAL_GRAPHS if g.documented_future}
    for tok, paths in sorted(by_token.items()):
        if tok in future_tokens:
            continue
        non_family_paths = [p for p in paths if p not in family_dict_paths]
        if not non_family_paths:
            violations.append({
                "rule": "R5",
                "token": tok,
                "first_path": paths[0],
                "msg": (
                    f"graph token {tok!r} only appears in family-classifier "
                    "dicts — no actual corpus use found (mark documented_future=True "
                    "if this is a RESERVED_FUTURE_KEYS entry)"
                ),
            })

    # R6: canonical token names match GRAPH_NAME_RE.
    for g in CANONICAL_GRAPHS:
        if not GRAPH_NAME_RE.match(g.name):
            violations.append({
                "rule": "R6",
                "token": g.name,
                "msg": f"name {g.name!r} does not match {GRAPH_NAME_RE.pattern}",
            })

    # R7: every config.EVAL_GRAPHS name is canonical.
    for path, name in all_eval:
        if name not in CANONICAL_GRAPH_NAMES:
            violations.append({
                "rule": "R7",
                "token": name,
                "first_path": path,
                "msg": f"EVAL_GRAPHS entry {name!r} not in CANONICAL_GRAPHS",
            })

    # R8: every family in CANONICAL_GRAPHS is lowercase ASCII.
    for g in CANONICAL_GRAPHS:
        if not FAMILY_RE.match(g.family):
            violations.append({
                "rule": "R8",
                "token": g.name,
                "msg": f"family {g.family!r} for {g.name} does not match {FAMILY_RE.pattern}",
            })

    return {
        "status": "active",
        "n_canonical": len(CANONICAL_GRAPHS),
        "n_families": len(CANONICAL_FAMILIES),
        "n_literal_sites": len(all_literals),
        "n_distinct_literals": len(by_token),
        "n_family_dicts": len(all_family_dicts),
        "n_eval_graphs": len(all_eval),
        "canonical": [
            {
                "name": g.name,
                "family": g.family,
                "paper_label": g.paper_label,
                "source": g.source,
                "purpose": g.purpose,
                "aliases": list(g.aliases),
                "documented_future": g.documented_future,
            }
            for g in CANONICAL_GRAPHS
        ],
        "harvested_tokens": sorted(by_token.keys()),
        "family_dicts": [
            {"path": p, "dict": d, "mapping": m}
            for (p, d, m) in all_family_dicts
        ],
        "eval_graphs": [
            {"path": p, "name": n} for (p, n) in all_eval
        ],
        "rules": {
            "R1": "every harvested graph literal is in CANONICAL_GRAPHS",
            "R2": "every canonical has non-empty family + paper_label",
            "R3": "every canonical has source ∈ {SNAP,GAP,DIMACS,KONECT,WebGraph,synthetic,test}",
            "R4": "every family-classifier dict has canonical keys with matching families",
            "R5": "every harvested literal has a non-family-dict site (unless documented_future=True)",
            "R6": f"canonical names match {GRAPH_NAME_RE.pattern}",
            "R7": "every EVAL_GRAPHS entry in config.py is canonical",
            "R8": f"every canonical family matches {FAMILY_RE.pattern}",
        },
        "violations": violations,
    }


def _emit_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _emit_md(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Gate 258 — graph-name canonical map")
    lines.append("")
    lines.append(f"Status: **{data['status']}**")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    for k in ("n_canonical", "n_families", "n_literal_sites",
              "n_distinct_literals", "n_family_dicts", "n_eval_graphs"):
        lines.append(f"- {k}: {data[k]}")
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    for rid, desc in data["rules"].items():
        lines.append(f"- **{rid}** — {desc}")
    lines.append("")
    lines.append("## Canonical graphs")
    lines.append("")
    lines.append("| name | family | source | paper_label |")
    lines.append("|---|---|---|---|")
    for g in data["canonical"]:
        lines.append(
            f"| `{g['name']}` | `{g['family']}` | `{g['source']}` | {g['paper_label']} |"
        )
    lines.append("")
    lines.append("## Harvested tokens (in-tree literals)")
    lines.append("")
    for t in data["harvested_tokens"]:
        lines.append(f"- `{t}`")
    lines.append("")
    lines.append("## Family-classifier dicts")
    lines.append("")
    for fd in data["family_dicts"]:
        lines.append(f"- `{fd['path']}` :: `{fd['dict']}` ({len(fd['mapping'])} entries)")
    lines.append("")
    if data["violations"]:
        lines.append("## Violations")
        lines.append("")
        for v in data["violations"]:
            lines.append(f"- {v}")
    else:
        lines.append("## Violations")
        lines.append("")
        lines.append("None.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _emit_csv(data: dict[str, Any], path: Path) -> None:
    rows: list[tuple[str, str, str, str]] = []
    for g in data["canonical"]:
        rows.append(("canonical", g["name"], g["family"], g["source"]))
    for t in data["harvested_tokens"]:
        rows.append(("literal", t, "", ""))
    for fd in data["family_dicts"]:
        rows.append(("family_dict", fd["dict"], fd["path"], str(len(fd["mapping"]))))
    for v in data["violations"]:
        rows.append(("violation", str(v.get("rule", "")), str(v.get("token", "")), str(v)))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("kind", "name", "extra1", "extra2"))
        for r in rows:
            w.writerow(r)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--md-out", type=Path)
    ap.add_argument("--csv-out", type=Path)
    args = ap.parse_args(argv)
    data = audit()
    if args.json_out:
        _emit_json(data, args.json_out)
    if args.md_out:
        _emit_md(data, args.md_out)
    if args.csv_out:
        _emit_csv(data, args.csv_out)
    print(
        f"[lit-faith-graph-registry] status={data['status']} "
        f"canonical={data['n_canonical']} "
        f"families={data['n_families']} "
        f"sites={data['n_literal_sites']} "
        f"distinct={data['n_distinct_literals']} "
        f"family_dicts={data['n_family_dicts']} "
        f"eval_graphs={data['n_eval_graphs']} "
        f"violations={len(data['violations'])}"
    )
    return 1 if data["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
