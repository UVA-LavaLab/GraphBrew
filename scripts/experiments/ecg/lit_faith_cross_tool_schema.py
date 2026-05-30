"""Gate 262 — ECG cross-tool aggregator schema registry.

Ninth in the vocabulary-lock series (252 SBATCH, 255 policy, 256
profile, 257 backend, 258 graph, 259 build, 260 CLI, 261
arm-catalog, 262 cross-tool aggregator schema). Locks the
on-disk JSON shape of every cross-tool aggregator artifact
(every report that joins cache_sim ↔ gem5 ↔ sniper at the per-cell
or per-policy level) so that a contributor cannot silently:

* rename a top-level key in one aggregator (downstream
  rollup scripts ``KeyError`` at the next confidence-fast run,
  but only after a 6-minute pytest start-up);
* delete the per-cell ``cells`` list and replace it with a
  scalar summary (every cross-artifact parity test passes
  trivially with 0 rows, hiding the regression);
* introduce a new aggregator that picks a third name for
  the same concept (``meta`` vs ``summary`` vs ``schema`` —
  this gate enforces both shape AND naming);
* shadow a canonical tool name (``sniper-v8`` instead of
  ``sniper``) in the ``tools`` field, so the dashboard parser
  registers ``sniper`` as missing and silently ignores half
  the data.

The gate registers the SHAPE of each aggregator's JSON (top-level
keys, required nested key paths, evidence-bearing list paths,
tool-set field) and asserts the on-disk shape matches. The
canonical aggregator set is enumerated up-front (6 today:
cross_tool_lru_regime, cross_tool_saturation,
cross_tool_slope_ordering, cross_tool_slope_universality,
cross_tool_winners, anchor_cross_tool_agreement) and additions
must add a CROSS_TOOL_AGGREGATORS entry before landing.

7 rules S1-S7:

  S1: every CROSS_TOOL_AGGREGATORS entry's artifact exists on
      disk (path is relative to wiki/data/).
  S2: every artifact is valid JSON with a top-level object
      (not a bare array or scalar).
  S3: every declared top-level key is present in the on-disk
      JSON (extra keys are allowed — schemas can grow but
      cannot silently shrink).
  S4: every declared evidence-bearing path
      (cells / per_tool / checks / shared_cells / tool_results)
      resolves to a non-empty list-or-dict in the on-disk JSON
      — every cross-tool aggregator must emit at least one
      row of evidence.
  S5: every declared cell-row required-key (e.g. ``app``,
      ``graph``, ``policy``) is present in every row of the
      evidence list.
  S6: every aggregator that declares a ``tools_path`` has a
      ``tools`` list (or dict whose keys are tools) under that
      path whose values are a subset of the canonical tool
      vocabulary {cache_sim, gem5, sniper}. (Reuses gate 257
      backend vocab.)
  S7: every aggregator declared to carry a verdict at
      ``verdict_path`` has a value at that path of the declared
      verdict type (bool / str / dict).

Today: 6 aggregators, 0 violations. The 6 aggregators between
them carry 4 cells-style list aggregations (winners, saturation,
shared_cells, slope_ordering.per_tool[tool]) + 2
verdict-summary-only aggregators (lru_regime, slope_universality).
"""
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
WIKI_DATA = REPO_ROOT / "wiki" / "data"

# Canonical tool names — gate 257's CANONICAL_BACKENDS vocab,
# including both underscore (cache_sim) and hyphen (cache-sim)
# punctuation siblings since both are paper-faithful spellings.
# Imported lazily inside ``_canonical_tools()`` to avoid a hard
# module-level cycle if backend registry imports this gate.
CANONICAL_TOOLS_FALLBACK = ("cache_sim", "cache-sim", "gem5", "sniper")


def _canonical_tools() -> tuple[str, ...]:
    """Pull the canonical tool token allow-list from gate 257 if
    available; fall back to the local list if backend registry is
    unimportable (e.g. during isolated unit tests)."""

    try:
        import sys
        sys.path.insert(0, str(REPO_ROOT / "scripts" / "experiments" / "ecg"))
        import lit_faith_backend_registry as br  # type: ignore
        return tuple(b.name for b in br.CANONICAL_BACKENDS)
    except Exception:
        return CANONICAL_TOOLS_FALLBACK


CANONICAL_TOOLS = _canonical_tools()


@dataclass(frozen=True)
class Aggregator:
    """Schema declaration for one cross-tool aggregator artifact."""

    name: str
    """Short name (matches the JSON file basename without .json)."""

    purpose: str
    """One-line summary of what cross-tool join this aggregator emits."""

    top_keys: tuple[str, ...]
    """Top-level keys that MUST be present in the on-disk JSON."""

    evidence_path: tuple[str, ...]
    """Dotted path (as a tuple) to the evidence list-or-dict; must be
    non-empty on disk. Empty tuple = no evidence required (aggregator
    is summary-only)."""

    row_required_keys: tuple[str, ...] = ()
    """If ``evidence_path`` is a list-of-dicts, every row must contain
    these keys. Empty tuple = no per-row check."""

    tools_path: tuple[str, ...] = ()
    """If non-empty, dotted path to a list/dict whose entries name
    canonical tools. Validated under S6."""

    verdict_path: tuple[str, ...] = ()
    """If non-empty, dotted path to the aggregator's verdict value."""

    verdict_type: type | None = None
    """The expected Python type of the value at ``verdict_path``."""


CROSS_TOOL_AGGREGATORS: tuple[Aggregator, ...] = (
    Aggregator(
        name="cross_tool_lru_regime",
        purpose=(
            "Reports the LRU-vs-GRASP sub-WSS vs post-WSS regime "
            "inversion across all 3 tools (does LRU climb out of the "
            "deficit once the working-set spills L3?)."
        ),
        top_keys=("meta",),
        evidence_path=("meta", "tool_results"),
        row_required_keys=(),
        tools_path=("meta", "tools"),
        verdict_path=("meta", "verdict"),
        verdict_type=str,
    ),
    Aggregator(
        name="cross_tool_saturation",
        purpose=(
            "Reports the doubly-saturated cell census: which "
            "(graph, app, L3) tuples have BOTH cache_sim AND "
            "anchor (gem5 or sniper) collapsing all policies to "
            "within sat_floor_pp."
        ),
        top_keys=("cells", "schema_version", "summary"),
        evidence_path=("cells",),
        row_required_keys=("app", "anchor_l3", "cache_sim_l3"),
        tools_path=(),
        verdict_path=("summary", "doubly_saturated_agree"),
        verdict_type=int,
    ),
    Aggregator(
        name="cross_tool_slope_ordering",
        purpose=(
            "Reports the strict per-tool slope ordering "
            "(POPT > GRASP > SRRIP > LRU at the corpus-median "
            "level) across all 3 tools."
        ),
        top_keys=("meta", "per_tool"),
        evidence_path=("per_tool",),
        row_required_keys=(),
        tools_path=("meta", "tools"),
        verdict_path=("meta", "verdict"),
        verdict_type=str,
    ),
    Aggregator(
        name="cross_tool_slope_universality",
        purpose=(
            "Reports whether the steepness-band universality "
            "claim (every policy's POPT-vs-LRU steepness lies "
            "in the same band across cache_sim/gem5/sniper) "
            "holds today."
        ),
        top_keys=("meta",),
        evidence_path=("meta", "tool_policies"),
        row_required_keys=(),
        tools_path=("meta", "tools"),
        verdict_path=("meta", "violations"),
        verdict_type=list,
    ),
    Aggregator(
        name="cross_tool_winners",
        purpose=(
            "Reports the per-cell winner-policy across "
            "cache_sim/gem5/sniper (does GRASP win in cache_sim "
            "AND in sniper, or is it split?)."
        ),
        top_keys=("cells", "summary"),
        evidence_path=("cells",),
        row_required_keys=("app", "classification"),
        tools_path=(),
        verdict_path=("summary", "n_cells"),
        verdict_type=int,
    ),
    Aggregator(
        name="anchor_cross_tool_agreement",
        purpose=(
            "Reports the gem5-vs-sniper anchor-cell agreement: "
            "for the shared anchor cells, does sniper's POPT slope "
            "and gem5's POPT slope agree in sign and magnitude "
            "within thresholds?"
        ),
        top_keys=(
            "checks",
            "meta",
            "schema",
            "shared_cells",
            "summary",
            "verdict_ok",
        ),
        evidence_path=("shared_cells",),
        row_required_keys=("app", "graph", "policy"),
        tools_path=(),
        verdict_path=("verdict_ok",),
        verdict_type=bool,
    ),
)


def _resolve(obj: Any, path: tuple[str, ...]) -> tuple[bool, Any]:
    """Return (found, value) walking obj down the dotted path."""

    cur = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return False, None
        cur = cur[k]
    return True, cur


def _evidence_nonempty(value: Any) -> bool:
    """A list-or-dict is non-empty if it has ≥1 element/key."""

    if isinstance(value, (list, tuple, dict, set)):
        return len(value) > 0
    return False


def audit() -> dict[str, Any]:
    """Run all 7 rules and return the registry shape."""

    aggregator_status: list[dict[str, Any]] = []
    violations: list[dict[str, Any]] = []
    n_evidence_rows_total = 0

    for agg in CROSS_TOOL_AGGREGATORS:
        path = WIKI_DATA / f"{agg.name}.json"
        status: dict[str, Any] = {
            "name": agg.name,
            "exists": path.exists(),
            "top_keys_ok": False,
            "evidence_nonempty": False,
            "evidence_row_count": 0,
            "row_keys_ok": True,
            "tools_ok": True,
            "verdict_ok": True,
        }

        # S1
        if not path.exists():
            violations.append(
                {"rule": "S1", "aggregator": agg.name, "reason": f"missing artifact {path}"}
            )
            aggregator_status.append(status)
            continue

        # S2
        try:
            doc = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            violations.append(
                {"rule": "S2", "aggregator": agg.name, "reason": f"invalid JSON: {exc}"}
            )
            aggregator_status.append(status)
            continue
        if not isinstance(doc, dict):
            violations.append(
                {"rule": "S2", "aggregator": agg.name,
                 "reason": f"top-level is {type(doc).__name__}, expected dict"}
            )
            aggregator_status.append(status)
            continue

        # S3
        missing_top = [k for k in agg.top_keys if k not in doc]
        status["top_keys_ok"] = not missing_top
        for k in missing_top:
            violations.append(
                {"rule": "S3", "aggregator": agg.name, "reason": f"missing top-key {k!r}"}
            )

        # S4
        if agg.evidence_path:
            found, ev = _resolve(doc, agg.evidence_path)
            if not found:
                violations.append(
                    {"rule": "S4", "aggregator": agg.name,
                     "reason": f"evidence path {'.'.join(agg.evidence_path)} not found"}
                )
            elif not _evidence_nonempty(ev):
                violations.append(
                    {"rule": "S4", "aggregator": agg.name,
                     "reason": f"evidence path {'.'.join(agg.evidence_path)} is empty"}
                )
            else:
                status["evidence_nonempty"] = True
                status["evidence_row_count"] = len(ev)
                n_evidence_rows_total += len(ev)

                # S5 (only on list-of-dicts)
                if agg.row_required_keys and isinstance(ev, list):
                    for i, row in enumerate(ev):
                        if not isinstance(row, dict):
                            violations.append(
                                {"rule": "S5", "aggregator": agg.name,
                                 "reason": f"row[{i}] is {type(row).__name__}, expected dict"}
                            )
                            status["row_keys_ok"] = False
                            continue
                        missing_row = [k for k in agg.row_required_keys if k not in row]
                        if missing_row:
                            violations.append(
                                {"rule": "S5", "aggregator": agg.name,
                                 "reason": f"row[{i}] missing keys {missing_row}"}
                            )
                            status["row_keys_ok"] = False

        # S6
        if agg.tools_path:
            found, tools = _resolve(doc, agg.tools_path)
            if not found:
                violations.append(
                    {"rule": "S6", "aggregator": agg.name,
                     "reason": f"tools path {'.'.join(agg.tools_path)} not found"}
                )
                status["tools_ok"] = False
            else:
                tools_iter: list[str] = []
                if isinstance(tools, dict):
                    tools_iter = list(tools.keys())
                elif isinstance(tools, (list, tuple)):
                    # Allow either plain strings or dicts with a 'name' key
                    # (cross_tool_lru_regime uses [{"name": ..., "l3_min_kb": ...}]).
                    for t in tools:
                        if isinstance(t, str):
                            tools_iter.append(t)
                        elif isinstance(t, dict) and "name" in t:
                            tools_iter.append(t["name"])
                        else:
                            violations.append(
                                {"rule": "S6", "aggregator": agg.name,
                                 "reason": f"tools[i] is {type(t).__name__} without 'name' key"}
                            )
                            status["tools_ok"] = False
                else:
                    violations.append(
                        {"rule": "S6", "aggregator": agg.name,
                         "reason": f"tools value is {type(tools).__name__}, expected list or dict"}
                    )
                    status["tools_ok"] = False
                bad = [t for t in tools_iter if t not in CANONICAL_TOOLS]
                if bad:
                    violations.append(
                        {"rule": "S6", "aggregator": agg.name,
                         "reason": (f"non-canonical tool names {bad} "
                                    f"(canonical={list(CANONICAL_TOOLS)})")}
                    )
                    status["tools_ok"] = False

        # S7
        if agg.verdict_path and agg.verdict_type is not None:
            found, v = _resolve(doc, agg.verdict_path)
            if not found:
                violations.append(
                    {"rule": "S7", "aggregator": agg.name,
                     "reason": f"verdict path {'.'.join(agg.verdict_path)} not found"}
                )
                status["verdict_ok"] = False
            elif not isinstance(v, agg.verdict_type):
                violations.append(
                    {"rule": "S7", "aggregator": agg.name,
                     "reason": (f"verdict at {'.'.join(agg.verdict_path)} is "
                                f"{type(v).__name__}, expected {agg.verdict_type.__name__}")}
                )
                status["verdict_ok"] = False

        aggregator_status.append(status)

    return {
        "status": "active",
        "n_aggregators": len(CROSS_TOOL_AGGREGATORS),
        "n_evidence_rows_total": n_evidence_rows_total,
        "canonical_tools": list(CANONICAL_TOOLS),
        "aggregators": [
            {
                "name": agg.name,
                "purpose": agg.purpose,
                "top_keys": list(agg.top_keys),
                "evidence_path": list(agg.evidence_path),
                "row_required_keys": list(agg.row_required_keys),
                "tools_path": list(agg.tools_path),
                "verdict_path": list(agg.verdict_path),
                "verdict_type": agg.verdict_type.__name__ if agg.verdict_type else None,
            }
            for agg in CROSS_TOOL_AGGREGATORS
        ],
        "aggregator_status": aggregator_status,
        "rules": {
            "S1": "every CROSS_TOOL_AGGREGATORS entry's artifact exists on disk",
            "S2": "every artifact is valid JSON with a top-level object",
            "S3": "every declared top-level key is present in the on-disk JSON",
            "S4": "every declared evidence path resolves to a non-empty list/dict",
            "S5": "every declared cell-row required-key is present in every evidence row",
            "S6": ("every declared tools-path value is a subset of "
                   f"canonical {list(CANONICAL_TOOLS)}"),
            "S7": "every declared verdict-path value has the declared type",
        },
        "violations": violations,
    }


def _emit_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _emit_md(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Gate 262 — ECG cross-tool aggregator schema registry")
    lines.append("")
    lines.append(f"Status: **{data['status']}**")
    lines.append("")
    lines.append("## Totals")
    lines.append("")
    lines.append(f"- n_aggregators: {data['n_aggregators']}")
    lines.append(f"- n_evidence_rows_total: {data['n_evidence_rows_total']}")
    lines.append(f"- canonical_tools: {data['canonical_tools']}")
    lines.append("")
    lines.append("## Rules")
    lines.append("")
    for rid, desc in data["rules"].items():
        lines.append(f"- **{rid}** — {desc}")
    lines.append("")
    lines.append("## Aggregators")
    lines.append("")
    lines.append("| name | purpose | top_keys | evidence_path | tools_path | verdict_path | verdict_type |")
    lines.append("|---|---|---|---|---|---|---|")
    for agg in data["aggregators"]:
        lines.append(
            f"| `{agg['name']}` | {agg['purpose']} | "
            f"`{agg['top_keys']}` | `{'.'.join(agg['evidence_path']) or '-'}` | "
            f"`{'.'.join(agg['tools_path']) or '-'}` | "
            f"`{'.'.join(agg['verdict_path']) or '-'}` | "
            f"{agg['verdict_type'] or '-'} |"
        )
    lines.append("")
    lines.append("## Aggregator status (today)")
    lines.append("")
    lines.append("| name | exists | top_keys_ok | evidence_nonempty | rows | row_keys_ok | tools_ok | verdict_ok |")
    lines.append("|---|---|---|---|---:|---|---|---|")
    for s in data["aggregator_status"]:
        lines.append(
            f"| `{s['name']}` | {s['exists']} | {s['top_keys_ok']} | "
            f"{s['evidence_nonempty']} | {s['evidence_row_count']} | "
            f"{s['row_keys_ok']} | {s['tools_ok']} | {s['verdict_ok']} |"
        )
    lines.append("")
    if data["violations"]:
        lines.append("## Violations")
        lines.append("")
        for v in data["violations"]:
            lines.append(f"- **{v['rule']}** `{v['aggregator']}` — {v['reason']}")
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
        for agg in data["aggregators"]:
            w.writerow(("aggregator", agg["name"], ".".join(agg["evidence_path"]) or ""))
        for s in data["aggregator_status"]:
            w.writerow(("status", s["name"], f"rows={s['evidence_row_count']}"))
        for v in data["violations"]:
            w.writerow(("violation", v["aggregator"], f"{v['rule']}: {v['reason']}"))


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
        f"[lit-faith-cross-tool-schema] status={data['status']} "
        f"aggregators={data['n_aggregators']} "
        f"evidence_rows={data['n_evidence_rows_total']} "
        f"violations={len(data['violations'])}"
    )
    return 1 if data["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
