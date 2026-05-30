"""Gate 264 — wiki/data artifact filename grammar registry.

Eleventh in the vocabulary-lock series (252 SBATCH, 255 policy,
256 profile, 257 backend, 258 graph, 259 build, 260 CLI, 261
arm-catalog, 262 cross-tool aggregator schema, 263 config
matrix, 264 wiki/data filename grammar). Locks the file-system
shape of ``wiki/data/`` — the single shipping surface for every
generator artifact in the paper-confidence pipeline — against
silent drift in filename casing, extension, trio-pairing,
catalog-presence, and subdirectory layout.

Today wiki/data/ ships 120 stems (119 .json + 116 .md + 58 .csv
= 293 files plus 1 subdirectory ``paper_pipeline_20260528/``).
115 of those stems are declared in
``scripts/experiments/ecg/artifact_catalog.py`` as the headline
artifact for their gate; the remaining 5 are documented here as
``IMPLICIT_PAPER_PIPELINE_STEMS`` (md-only summary), trio-sibling
stems that share their gate's catalog entry, or paired-table
postfix companions.

Catches the silent-drift cases:

* a contributor adds ``DiscoveryResults.json`` (CamelCase) to
  wiki/data/ — every consumer that does
  ``json.load(open(stem.lower() + '.json'))`` ``FileNotFoundError``\
  s and the rollup script silently treats the report as missing;
* the trio convention is broken by emitting only ``foo.json``
  without ``foo.md`` — the per-artifact preview-link doc
  template renders a broken markdown link in HANDOFF;
* a generator drops an artifact under
  ``wiki/data/results/foo.json`` instead of
  ``wiki/data/foo.json`` — the catalog presence-check passes
  (artifact path is literal), but the dashboard's
  ``wiki/data/*.json`` glob misses it and the gate goes
  un-scored;
* a contributor adds a ``.txt`` or ``.tex`` file directly to
  wiki/data/ — the trio pattern is ambiguous, the artifact
  catalog grows by zero entries, and the dashboard's
  ``mtime``-based regen check picks an arbitrary cutoff;
* a renamed catalog entry's ``artifact`` field points at a
  path that doesn't exist on disk — the catalog has
  ``missing art=[]`` but a single typo can flip it to
  ``missing art=[wiki/data/foo.json]`` and the entire reproduce
  smoke gate trips.

7 rules F1-F7:

  F1: every regular file in wiki/data/ matches
      ``^[a-z][a-z0-9_]*\\.(json|md|csv)$`` — lower_snake_case
      stem, no leading digit, no whitespace, no dash, exactly
      one of three approved extensions.
  F2: every .json under wiki/data/ has a matching .md sibling,
      UNLESS the stem is in the documented
      ``MD_OPTIONAL_STEMS`` allow-list (paired-table postfix
      companions whose narrative lives in the parent table's
      .md).
  F3: every .md under wiki/data/ has a matching .json sibling
      OR is in the documented ``JSON_OPTIONAL_STEMS`` allow-list
      (today: ``literature_reproduction_summary`` — a
      free-form text report).
  F4: every artifact-catalog entry's ``artifact`` field points
      at a file that exists on disk.
  F5: every artifact-catalog entry's ``artifact`` field has
      extension in {.json, .csv} (the catalog tracks data
      artifacts, not narrative artifacts; narrative .md files
      are implicit trio-siblings of their .json parent).
  F6: every wiki/data/ stem is accounted for: either declared
      in artifact_catalog as the ``artifact`` (modulo .md/.csv
      sibling), or in the documented
      ``IMPLICIT_PAPER_PIPELINE_STEMS`` allow-list (free-form
      summaries, paired-table postfix companions, archive
      stems).
  F7: every wiki/data/ subdirectory name is in the documented
      ``DOCUMENTED_SUBDIRS`` allow-list (today: only
      ``paper_pipeline_*`` snapshot dirs are allowed; ad-hoc
      ``results/`` or ``archive/`` subdirs are blocked).

Today: 120 stems, 293 files, 1 subdir; 115/115 catalog
artifacts on disk; 0 violations.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"
CATALOG_PY = REPO_ROOT / "scripts" / "experiments" / "ecg" / "artifact_catalog.py"

# F1 enforced pattern
STEM_RE = re.compile(r"^[a-z][a-z0-9_]*$")
EXT_ALLOWED = {".json", ".md", ".csv"}

# F2 documented exceptions — JSONs that intentionally have no .md sibling
# because their narrative lives in a parent table's .md.
MD_OPTIONAL_STEMS = {
    "ecg_gem5_parity_postfix",
    "ecg_pfx_vs_droplet_postfix",
    "ecg_sniper_parity_postfix",
    "ecg_substrate_parity_postfix",
}

# F3 documented exceptions — .mds that intentionally have no .json sibling
# because they are free-form text reports.
JSON_OPTIONAL_STEMS = {
    "literature_reproduction_summary",
}

# F6 — meta-artifact stems: present on disk but not catalog-declared
# because they are *meta* artifacts that aggregate the catalog itself
# (and thus cannot self-reference without bootstrapping concerns).
META_ARTIFACT_STEMS = {
    "artifact_catalog",
}

# F6 documented stems present on disk but not catalog-declared.
# These are either implicit trio-siblings (md/csv whose .json is catalog-declared)
# or documented free-form summaries.
IMPLICIT_PAPER_PIPELINE_STEMS = (
    set(MD_OPTIONAL_STEMS)
    | set(JSON_OPTIONAL_STEMS)
    | set(META_ARTIFACT_STEMS)
)

# F7 — allowed subdirectory name patterns
DOCUMENTED_SUBDIR_RE = re.compile(r"^paper_pipeline_\d{8}(_[a-z0-9_]+)?$")


def _load_catalog():
    spec = importlib.util.spec_from_file_location("ecg_artifact_catalog_gate264", CATALOG_PY)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ecg_artifact_catalog_gate264"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def audit() -> dict[str, Any]:
    """Run all 7 rules and return the registry shape."""
    violations: list[dict[str, Any]] = []

    catalog_mod = _load_catalog()
    catalog = list(catalog_mod.CATALOG)
    catalog_artifacts: dict[str, dict[str, Any]] = {}
    for entry in catalog:
        a = entry.get("artifact", "")
        catalog_artifacts[a] = entry

    # --- F4 + F5: catalog → disk presence + extension allow-list -----------
    n_catalog = len(catalog)
    for entry in catalog:
        art = entry.get("artifact", "")
        if not art:
            violations.append(
                {"rule": "F4", "subject": entry.get("id", "?"),
                 "reason": "catalog entry has empty artifact field"}
            )
            continue
        art_path = REPO_ROOT / art
        if not art_path.is_file():
            violations.append(
                {"rule": "F4", "subject": art,
                 "reason": f"catalog artifact {art} does not exist on disk"}
            )
        ext = art_path.suffix
        if ext not in {".json", ".csv"}:
            violations.append(
                {"rule": "F5", "subject": art,
                 "reason": f"catalog artifact extension {ext!r} not in {{'.json','.csv'}}"}
            )

    # --- F1, F2, F3: walk wiki/data/ -------------------------------------
    files: list[Path] = []
    subdirs: list[Path] = []
    for p in sorted(WIKI_DATA.iterdir()):
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            subdirs.append(p)

    n_files = len(files)
    stems_seen: dict[str, set[str]] = {}
    for f in files:
        stem = f.stem
        ext = f.suffix
        if ext not in EXT_ALLOWED:
            violations.append(
                {"rule": "F1", "subject": f.name,
                 "reason": f"extension {ext!r} not in {{'.json','.md','.csv'}}"}
            )
        if not STEM_RE.match(stem):
            violations.append(
                {"rule": "F1", "subject": f.name,
                 "reason": f"stem {stem!r} not lower_snake_case "
                           f"(must match {STEM_RE.pattern!r})"}
            )
        stems_seen.setdefault(stem, set()).add(ext)
    n_stems = len(stems_seen)

    # F2: every .json has .md sibling (modulo allow-list)
    for stem, exts in stems_seen.items():
        if ".json" in exts and ".md" not in exts:
            if stem not in MD_OPTIONAL_STEMS:
                violations.append(
                    {"rule": "F2", "subject": stem,
                     "reason": f"{stem}.json has no {stem}.md sibling and "
                               f"is not in MD_OPTIONAL_STEMS allow-list"}
                )

    # F3: every .md has .json sibling (modulo allow-list)
    for stem, exts in stems_seen.items():
        if ".md" in exts and ".json" not in exts:
            if stem not in JSON_OPTIONAL_STEMS:
                violations.append(
                    {"rule": "F3", "subject": stem,
                     "reason": f"{stem}.md has no {stem}.json sibling and "
                               f"is not in JSON_OPTIONAL_STEMS allow-list"}
                )

    # --- F6: every wiki/data stem is catalog-declared or allow-listed ----
    catalog_stems = set()
    for art_path in catalog_artifacts:
        catalog_stems.add(Path(art_path).stem)
    n_catalog_stems = len(catalog_stems)
    for stem in stems_seen:
        if stem in catalog_stems:
            continue
        if stem in IMPLICIT_PAPER_PIPELINE_STEMS:
            continue
        violations.append(
            {"rule": "F6", "subject": stem,
             "reason": f"wiki/data stem {stem!r} is neither in artifact_catalog "
                       f"nor in IMPLICIT_PAPER_PIPELINE_STEMS"}
        )

    # --- F7: every subdirectory is documented ----------------------------
    n_subdirs = len(subdirs)
    for d in subdirs:
        if not DOCUMENTED_SUBDIR_RE.match(d.name):
            violations.append(
                {"rule": "F7", "subject": d.name,
                 "reason": f"wiki/data/ subdir {d.name!r} does not match "
                           f"DOCUMENTED_SUBDIR_RE pattern {DOCUMENTED_SUBDIR_RE.pattern!r}"}
            )

    # --- Catalog internal consistency (id uniqueness) ---------------------
    seen_ids: set[str] = set()
    for entry in catalog:
        eid = entry.get("id", "")
        if eid in seen_ids:
            violations.append(
                {"rule": "F4", "subject": eid,
                 "reason": f"duplicate catalog id {eid!r}"}
            )
        seen_ids.add(eid)
    seen_paths: set[str] = set()
    for entry in catalog:
        art = entry.get("artifact", "")
        if art in seen_paths and art:
            violations.append(
                {"rule": "F4", "subject": art,
                 "reason": f"duplicate catalog artifact path {art!r}"}
            )
        seen_paths.add(art)

    n_json = sum(1 for s, e in stems_seen.items() if ".json" in e)
    n_md = sum(1 for s, e in stems_seen.items() if ".md" in e)
    n_csv = sum(1 for s, e in stems_seen.items() if ".csv" in e)

    return {
        "status": "active",
        "n_files": n_files,
        "n_stems": n_stems,
        "n_json": n_json,
        "n_md": n_md,
        "n_csv": n_csv,
        "n_subdirs": n_subdirs,
        "n_catalog": n_catalog,
        "n_catalog_stems": n_catalog_stems,
        "md_optional_stems": sorted(MD_OPTIONAL_STEMS),
        "json_optional_stems": sorted(JSON_OPTIONAL_STEMS),
        "meta_artifact_stems": sorted(META_ARTIFACT_STEMS),
        "implicit_paper_pipeline_stems": sorted(IMPLICIT_PAPER_PIPELINE_STEMS),
        "subdirs": [d.name for d in subdirs],
        "rules": {
            "F1": ("every file in wiki/data/ matches "
                   r"`^[a-z][a-z0-9_]*\.(json|md|csv)$`"),
            "F2": ("every .json has a matching .md sibling OR is in "
                   "MD_OPTIONAL_STEMS"),
            "F3": ("every .md has a matching .json sibling OR is in "
                   "JSON_OPTIONAL_STEMS"),
            "F4": "every catalog.artifact path exists on disk; catalog id+path unique",
            "F5": "every catalog.artifact extension is in {.json, .csv}",
            "F6": ("every wiki/data stem is in artifact_catalog OR in "
                   "IMPLICIT_PAPER_PIPELINE_STEMS"),
            "F7": ("every wiki/data subdirectory matches DOCUMENTED_SUBDIR_RE "
                   "(paper_pipeline_<YYYYMMDD>...)"),
        },
        "violations": violations,
    }


def _emit_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _emit_md(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Gate 264 — wiki/data artifact filename grammar registry")
    lines.append("")
    lines.append(f"Status: **{data['status']}**")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    for k in ("n_files", "n_stems", "n_json", "n_md", "n_csv", "n_subdirs",
              "n_catalog", "n_catalog_stems"):
        lines.append(f"- {k}: {data[k]}")
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    for rid, desc in data["rules"].items():
        lines.append(f"- **{rid}** — {desc}")
    lines.append("")
    lines.append("## Allow-lists")
    lines.append("")
    lines.append(f"- MD_OPTIONAL_STEMS: `{data['md_optional_stems']}`")
    lines.append(f"- JSON_OPTIONAL_STEMS: `{data['json_optional_stems']}`")
    lines.append(f"- META_ARTIFACT_STEMS: `{data['meta_artifact_stems']}`")
    lines.append("")
    lines.append("## Subdirectories")
    lines.append("")
    for d in data["subdirs"]:
        lines.append(f"- `{d}`")
    if not data["subdirs"]:
        lines.append("(none)")
    lines.append("")
    if data["violations"]:
        lines.append("## Violations")
        lines.append("")
        for v in data["violations"]:
            lines.append(f"- **{v['rule']}** `{v['subject']}` — {v['reason']}")
    else:
        lines.append("## Violations")
        lines.append("")
        lines.append("None.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _emit_csv(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("kind", "name", "extra"))
        for k in ("n_files", "n_stems", "n_json", "n_md", "n_csv", "n_subdirs",
                  "n_catalog", "n_catalog_stems"):
            w.writerow(("total", k, str(data[k])))
        for s in data["md_optional_stems"]:
            w.writerow(("md_optional_stem", s, ""))
        for s in data["json_optional_stems"]:
            w.writerow(("json_optional_stem", s, ""))
        for d in data["subdirs"]:
            w.writerow(("subdir", d, ""))
        for v in data["violations"]:
            w.writerow(("violation", v["subject"], f"{v['rule']}: {v['reason']}"))


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
        f"[lit-faith-filename-grammar] status={data['status']} "
        f"files={data['n_files']} stems={data['n_stems']} "
        f"catalog={data['n_catalog']} violations={len(data['violations'])}"
    )
    return 1 if data["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
