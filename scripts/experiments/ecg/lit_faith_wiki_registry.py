#!/usr/bin/env python3
"""Gate 254 — wiki/data bidirectional registry.

Locks the contract that every JSON artifact under ``wiki/data/`` is
fully accounted for:

* listed in ``scripts/experiments/ecg/artifact_catalog.py`` as a
  catalog entry with a generator and a pytest gate, OR
* explicitly enumerated in the ``ALLOWED_AUXILIARY`` allow-list with
  a parent catalog entry and a clear documented purpose, OR
* explicitly enumerated in ``SELF_REFERENTIAL`` (the catalog
  artifact itself, which can't catalog itself reflectively).

Symmetric to gate 253 (HANDOFF↔PYTEST_SUITES): gate 253 binds
narrative ↔ live suite count; gate 254 binds raw artifact filesystem
↔ catalog entries ↔ pytest coverage.

8 rules:

  * W1 every ``wiki/data/*.json`` file is accounted for (catalog
    entry OR auxiliary allow-list OR self-referential);
  * W2 every catalog entry that names ``wiki/data/<x>.json`` has the
    underlying file present on disk (no ghost entries);
  * W3 every catalog entry has ``generator``, ``gate`` (pytest), and
    ``artifact`` populated as non-empty strings;
  * W4 every catalog entry's generator/gate/artifact file path
    actually exists in the working tree;
  * W5 if a catalog entry names a ``.json`` artifact, the sibling
    ``.md`` (same stem) exists too (every JSON artifact has a human
    summary);
  * W6 catalog entry ``id`` values are unique;
  * W7 catalog entry ``artifact`` paths are unique (no two entries
    point at the same file);
  * W8 every auxiliary entry in ``ALLOWED_AUXILIARY`` lists a
    ``parent_id`` that is a real catalog entry id (no dangling
    parents).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = ROOT / "wiki" / "data"
CATALOG_JSON = WIKI_DATA / "artifact_catalog.json"

SELF_REFERENTIAL: set[str] = {
    "wiki/data/artifact_catalog.json",
}

ALLOWED_AUXILIARY: dict[str, dict[str, str]] = {
    "wiki/data/ecg_gem5_parity_postfix.json": {
        "parent_id": "lit_faith_ecg_gem5_parity",
        "purpose": "Gate 239 ECG-Gem5-Parity postfix summary "
                   "(epsilon parity + gate metadata) emitted alongside "
                   "the lit-faith generator's main JSON.",
    },
    "wiki/data/ecg_sniper_parity_postfix.json": {
        "parent_id": "lit_faith_ecg_sniper_parity",
        "purpose": "Gate 240 ECG-Sniper-Parity postfix scaffold "
                   "(deferred — no matched-proof Sniper ECG run yet).",
    },
    "wiki/data/ecg_pfx_vs_droplet_postfix.json": {
        "parent_id": "lit_faith_ecg_pfx_vs_droplet",
        "purpose": "Gate 241 ECG-Pfx-vs-DROPLET postfix scaffold "
                   "(deferred — neither prefetcher arm has runs yet).",
    },
    "wiki/data/ecg_substrate_parity_postfix.json": {
        "parent_id": "lit_faith_ecg_parity",
        "purpose": "Gate 238 ECG substrate-parity raw per-observation "
                   "rows; companion artifact to the main parity JSON.",
    },
}


def _load_catalog() -> list[dict[str, Any]]:
    data = json.loads(CATALOG_JSON.read_text())
    return data["entries"]


def _scan_wiki_json() -> list[str]:
    out: list[str] = []
    for p in sorted(WIKI_DATA.glob("*.json")):
        out.append(f"wiki/data/{p.name}")
    return out


# --- rules -----------------------------------------------------------

def _rule_w1(json_files: list[str], catalog_artifacts: set[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for f in json_files:
        if f in SELF_REFERENTIAL:
            continue
        if f in catalog_artifacts:
            continue
        if f in ALLOWED_AUXILIARY:
            continue
        out.append({
            "rule": "W1",
            "artifact": f,
            "msg": "json artifact is not in catalog, auxiliary "
                   "allow-list, or self-referential set",
        })
    return out


def _rule_w2(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in entries:
        art = e.get("artifact", "")
        if not art.startswith("wiki/data/"):
            continue
        if not (ROOT / art).is_file():
            out.append({
                "rule": "W2",
                "id": e.get("id"),
                "artifact": art,
                "msg": "catalog entry's artifact file does not exist",
            })
    return out


def _rule_w3(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in entries:
        for k in ("generator", "gate", "artifact"):
            v = e.get(k)
            if not isinstance(v, str) or not v.strip():
                out.append({
                    "rule": "W3",
                    "id": e.get("id"),
                    "field": k,
                    "value": v,
                    "msg": f"catalog entry has empty/missing {k}",
                })
    return out


def _rule_w4(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in entries:
        for k in ("generator", "gate", "artifact"):
            v = e.get(k, "")
            if not isinstance(v, str) or not v.strip():
                continue
            if not (ROOT / v).is_file():
                out.append({
                    "rule": "W4",
                    "id": e.get("id"),
                    "field": k,
                    "path": v,
                    "msg": f"catalog entry's {k} path does not exist",
                })
    return out


def _rule_w5(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for e in entries:
        art = e.get("artifact", "")
        if not art.endswith(".json"):
            continue
        md_sibling = art[:-len(".json")] + ".md"
        if not (ROOT / md_sibling).is_file():
            out.append({
                "rule": "W5",
                "id": e.get("id"),
                "artifact": art,
                "missing": md_sibling,
                "msg": "json artifact has no sibling .md summary",
            })
    return out


def _rule_w6(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, list[int]] = {}
    for i, e in enumerate(entries):
        seen.setdefault(e.get("id", ""), []).append(i)
    return [
        {
            "rule": "W6",
            "id": k,
            "indices": v,
            "msg": f"duplicate catalog id (appears {len(v)} times)",
        }
        for k, v in seen.items()
        if len(v) > 1
    ]


def _rule_w7(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: dict[str, list[str]] = {}
    for e in entries:
        a = e.get("artifact", "")
        if not a:
            continue
        seen.setdefault(a, []).append(e.get("id", ""))
    return [
        {
            "rule": "W7",
            "artifact": k,
            "ids": v,
            "msg": f"artifact path claimed by {len(v)} catalog entries",
        }
        for k, v in seen.items()
        if len(v) > 1
    ]


def _rule_w8(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    catalog_ids = {e.get("id") for e in entries}
    out: list[dict[str, Any]] = []
    for aux, meta in ALLOWED_AUXILIARY.items():
        parent = meta.get("parent_id")
        if parent not in catalog_ids:
            out.append({
                "rule": "W8",
                "auxiliary": aux,
                "parent_id": parent,
                "msg": "auxiliary entry references unknown parent_id",
            })
    return out


# --- driver ----------------------------------------------------------

def audit() -> dict[str, Any]:
    entries = _load_catalog()
    json_files = _scan_wiki_json()
    catalog_artifacts = {e.get("artifact", "") for e in entries}

    violations: list[dict[str, Any]] = []
    violations.extend(_rule_w1(json_files, catalog_artifacts))
    violations.extend(_rule_w2(entries))
    violations.extend(_rule_w3(entries))
    violations.extend(_rule_w4(entries))
    violations.extend(_rule_w5(entries))
    violations.extend(_rule_w6(entries))
    violations.extend(_rule_w7(entries))
    violations.extend(_rule_w8(entries))

    return {
        "status": "active",
        "rules": {
            "W1": "every wiki/data/*.json accounted for (catalog | aux | self-ref)",
            "W2": "no ghost catalog entries (artifact files exist on disk)",
            "W3": "catalog entries have non-empty generator/gate/artifact",
            "W4": "catalog entries' generator/gate/artifact paths exist",
            "W5": "every .json artifact has a sibling .md summary",
            "W6": "catalog ids are unique",
            "W7": "catalog artifact paths are unique",
            "W8": "auxiliary parent_id values reference real catalog ids",
        },
        "totals": {
            "wiki_json_files": len(json_files),
            "catalog_entries": len(entries),
            "auxiliary_entries": len(ALLOWED_AUXILIARY),
            "self_referential": len(SELF_REFERENTIAL),
            "violations": len(violations),
        },
        "wiki_json_files": json_files,
        "auxiliary": [
            {"artifact": k, **v} for k, v in sorted(ALLOWED_AUXILIARY.items())
        ],
        "violations": violations,
    }


def write_outputs(data: dict[str, Any], json_out: Path | None,
                  md_out: Path | None, csv_out: Path | None) -> None:
    if json_out:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if md_out:
        md_out.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        lines.append("# wiki/data bidirectional registry — gate 254")
        lines.append("")
        lines.append(f"Status: `{data['status']}`")
        lines.append("")
        t = data["totals"]
        lines.append(
            f"Totals: wiki_json={t['wiki_json_files']}  "
            f"catalog={t['catalog_entries']}  "
            f"auxiliary={t['auxiliary_entries']}  "
            f"self_referential={t['self_referential']}  "
            f"violations={t['violations']}")
        lines.append("")
        lines.append("## Auxiliary allow-list")
        lines.append("")
        for aux in data["auxiliary"]:
            lines.append(f"- `{aux['artifact']}` (parent: `{aux['parent_id']}`) — "
                         f"{aux['purpose']}")
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--md-out", type=Path, default=None)
    ap.add_argument("--csv-out", type=Path, default=None)
    args = ap.parse_args()
    a = audit()
    write_outputs(a, args.json_out, args.md_out, args.csv_out)
    print(
        f"[lit-faith-wiki-registry] status={a['status']} "
        f"wiki_json={a['totals']['wiki_json_files']} "
        f"catalog={a['totals']['catalog_entries']} "
        f"aux={a['totals']['auxiliary_entries']} "
        f"violations={a['totals']['violations']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
