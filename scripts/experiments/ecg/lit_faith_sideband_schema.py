#!/usr/bin/env python3
"""Gate 248 — gem5/Sniper/cache_sim sideband-schema registry.

GraphBrew's three simulator overlays each emit a single
``[graphctx] register region ...`` log line per parsed region from
the JSON sideband file. Tier-A regression tests (gate 1) parse those
lines for sanity checks (PR/BC: 2 regions, hot_pct=50, grasp_region=1;
BellmanFord: 1 region, hot_pct=100). Those tests anchor on the exact
*key=value* token order — if any simulator silently re-orders, renames,
or drops a field, the regex stops matching and downstream sideband-
parity tests can RED for unrelated reasons.

This gate codifies the wire-format schema in a hand-curated
``SCHEMA_REGISTRY`` and asserts that every emit site declares the
fields in the same canonical order.

Source-of-truth (two halves on disk):

  1. Three C++ headers / source files that emit ``logGraphCtxRegistration``::

       bench/include/gem5_sim/overlays/mem/cache/replacement_policies/graph_cache_context_gem5.hh
       bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/graph_cache_context_sniper.cc
       bench/include/cache_sim/graph_cache_context.h

  2. A hand-curated ``SCHEMA_REGISTRY`` in this file declaring:

       - the canonical *ordered* schema field list,
       - the printf format-specifier for each field (``%s`` / ``%lx`` / ``%u`` / ``%d``),
       - the formal C++ parameter list of ``logGraphCtxRegistration``,
       - the regex anchor that Tier-A tests use.

Rules:

  S1 — every registered emit-site file exists;
  S2 — the printf format string in the file matches the canonical
       schema field order byte-for-byte (modulo C string-literal wrap);
  S3 — the C++ function signature ``logGraphCtxRegistration(...)``
       parameter list matches the canonical parameter list;
  S4 — every emit-site uses the same canonical literal prefix
       ``[graphctx] register region``;
  S5 — every schema field has a documented printf specifier (no
       silent ``%p``/``%X``/``%llu`` drift);
  S6 — the Tier-A parser regex anchor compiles AND matches a sample
       line built from the canonical schema (round-trip);
  S7 — no emit-site contains a *second* ``register region`` printf
       (would mean somebody added a divergent emit path).

Today the schema is::

    source=%s name=%s base=0x%lx upper=0x%lx hot_pct=%u grasp_region=%d
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2].parent
WIKI_DATA = ROOT / "wiki" / "data"


# ----------------------------------------------------------- registry --

EMIT_SITES: list[str] = [
    "bench/include/gem5_sim/overlays/mem/cache/replacement_policies/graph_cache_context_gem5.hh",
    "bench/include/sniper_sim/overlays/common/core/memory_subsystem/cache/graph_cache_context_sniper.cc",
    "bench/include/cache_sim/graph_cache_context.h",
]

# Canonical schema — field name, printf specifier, C++ parameter type.
# Order matters: this is the wire-format order.
SCHEMA_REGISTRY: list[dict] = [
    {"name": "source",       "fmt": "%s",  "cpp_type": "const char*"},
    {"name": "name",         "fmt": "%s",  "cpp_type": "const char*"},
    {"name": "base",         "fmt": "0x%lx", "cpp_type": "uint64_t"},
    {"name": "upper",        "fmt": "0x%lx", "cpp_type": "uint64_t"},
    {"name": "hot_pct",      "fmt": "%u",  "cpp_type": "uint32_t"},
    {"name": "grasp_region", "fmt": "%d",  "cpp_type": "bool"},
]

EMIT_PREFIX = "[graphctx] register region"

# The Tier-A parser anchor — built from canonical schema, used by
# gate 1 to validate sideband registration. Kept here so any schema
# drift forces a parallel update to gate 1's parser.
TIER_A_REGEX = (
    r"\[graphctx\] register region "
    r"source=(?P<source>\S+) name=(?P<name>\S+) "
    r"base=0x(?P<base>[0-9a-fA-F]+) upper=0x(?P<upper>[0-9a-fA-F]+) "
    r"hot_pct=(?P<hot_pct>\d+) grasp_region=(?P<grasp_region>\d+)"
)


# ----------------------------------------------------------- helpers --

_FUNC_RE = re.compile(
    r"(?:inline\s+)?void\s+logGraphCtxRegistration\s*\(([^)]*)\)",
    re.DOTALL,
)


def _canonical_format_string() -> str:
    """Reconstruct the expected printf format substring from the registry."""
    parts = [f"{f['name']}={f['fmt']}" for f in SCHEMA_REGISTRY]
    return f"{EMIT_PREFIX} " + " ".join(parts) + r"\n"


def _normalize_format_literal(text: str) -> str:
    """Concatenate adjacent C string literals like ``"a " "b\\n"`` -> ``"a b\\n"``."""
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == '"':
            # consume one string literal
            j = i + 1
            while j < len(text):
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == '"':
                    out.append(text[i + 1:j])
                    i = j + 1
                    break
                j += 1
            else:
                break
        else:
            i += 1
    return "".join(out)


def _extract_format_string(src: str) -> str | None:
    """Pull the format string from the fprintf inside logGraphCtxRegistration."""
    m = re.search(
        r"std::fprintf\s*\(\s*stderr\s*,\s*((?:\"(?:[^\"\\]|\\.)*\"\s*)+),",
        src,
        re.DOTALL,
    )
    if not m:
        return None
    return _normalize_format_literal(m.group(1))


def _extract_param_types(src: str) -> list[str] | None:
    """Pull the parameter type list of logGraphCtxRegistration."""
    m = _FUNC_RE.search(src)
    if not m:
        return None
    raw = m.group(1)
    params = []
    for chunk in raw.split(","):
        c = " ".join(chunk.split())
        if not c:
            continue
        # split type from name: take everything except the last token
        toks = c.split()
        if len(toks) >= 2:
            ptype = " ".join(toks[:-1])
        else:
            ptype = c
        params.append(ptype)
    return params


def _count_emit_calls(src: str) -> int:
    """Count occurrences of the literal emit prefix in a fprintf format."""
    # Count distinct fprintf-with-prefix occurrences, not raw substring,
    # because the prefix may show up in comments.
    n = 0
    for m in re.finditer(
        r"std::fprintf\s*\(\s*stderr\s*,\s*((?:\"(?:[^\"\\]|\\.)*\"\s*)+),",
        src,
        re.DOTALL,
    ):
        if EMIT_PREFIX in _normalize_format_literal(m.group(1)):
            n += 1
    return n


# ----------------------------------------------------------- audit --

_ALLOWED_SPECIFIERS = {"%s", "0x%lx", "%u", "%d"}


def audit() -> dict:
    canonical_param_types = [f["cpp_type"] for f in SCHEMA_REGISTRY]
    canonical_fmt = _canonical_format_string()

    site_rows = []
    violations = []

    # S5 — every registry entry has a documented printf specifier
    for entry in SCHEMA_REGISTRY:
        if entry["fmt"] not in _ALLOWED_SPECIFIERS:
            violations.append({
                "rule": "S5",
                "site": "(registry)",
                "detail": f"field {entry['name']} fmt={entry['fmt']} "
                          f"not in {sorted(_ALLOWED_SPECIFIERS)}",
            })

    for rel in EMIT_SITES:
        path = ROOT / rel
        row = {"site": rel, "exists": path.exists()}

        if not path.exists():
            violations.append({"rule": "S1", "site": rel,
                               "detail": "file missing"})
            site_rows.append(row)
            continue

        src = path.read_text()

        # S2 — format string matches canonical
        fmt = _extract_format_string(src)
        row["format_string"] = fmt
        if fmt is None:
            violations.append({"rule": "S2", "site": rel,
                               "detail": "no fprintf found"})
        elif fmt != canonical_fmt:
            violations.append({
                "rule": "S2", "site": rel,
                "detail": f"got={fmt!r} want={canonical_fmt!r}",
            })

        # S3 — function signature parameter types match canonical
        param_types = _extract_param_types(src)
        row["param_types"] = param_types
        if param_types is None:
            violations.append({"rule": "S3", "site": rel,
                               "detail": "no logGraphCtxRegistration sig found"})
        elif param_types != canonical_param_types:
            violations.append({
                "rule": "S3", "site": rel,
                "detail": f"got={param_types} want={canonical_param_types}",
            })

        # S4 — canonical literal prefix appears in the format string
        if fmt is None or EMIT_PREFIX not in fmt:
            violations.append({"rule": "S4", "site": rel,
                               "detail": f"missing prefix {EMIT_PREFIX!r}"})

        # S7 — exactly one register-region fprintf per file
        n = _count_emit_calls(src)
        row["emit_call_count"] = n
        if n != 1:
            violations.append({"rule": "S7", "site": rel,
                               "detail": f"emit_call_count={n} (want 1)"})

        site_rows.append(row)

    # S6 — Tier-A regex compiles AND round-trips a canonical sample line
    sample_line = (
        f"{EMIT_PREFIX} source=gem5 name=property base=0x10000 "
        f"upper=0x10100 hot_pct=50 grasp_region=1"
    )
    s6_ok = True
    try:
        m = re.search(TIER_A_REGEX, sample_line)
        if not m:
            s6_ok = False
            violations.append({"rule": "S6", "site": "(parser)",
                               "detail": "regex did not match sample line"})
        else:
            for field in [f["name"] for f in SCHEMA_REGISTRY]:
                if field not in m.groupdict():
                    s6_ok = False
                    violations.append({
                        "rule": "S6", "site": "(parser)",
                        "detail": f"named group {field!r} missing"})
    except re.error as e:
        s6_ok = False
        violations.append({"rule": "S6", "site": "(parser)",
                           "detail": f"regex compile error: {e}"})

    return {
        "status": "active",
        "schema_field_count": len(SCHEMA_REGISTRY),
        "emit_site_count": len(EMIT_SITES),
        "canonical_param_types": canonical_param_types,
        "canonical_format_string": canonical_fmt,
        "tier_a_regex": TIER_A_REGEX,
        "tier_a_regex_round_trip_ok": s6_ok,
        "site_rows": site_rows,
        "violations": violations,
    }


# ----------------------------------------------------------- writers --

def _write_json(out: Path, data: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2) + "\n")


def _write_md(out: Path, data: dict) -> None:
    buf = io.StringIO()
    buf.write("# Gate 248 — gem5/Sniper/cache_sim sideband-schema registry\n\n")
    buf.write(f"- status: **{data['status']}**\n")
    buf.write(f"- schema fields: {data['schema_field_count']}\n")
    buf.write(f"- emit sites: {data['emit_site_count']}\n")
    buf.write(f"- Tier-A regex round-trip ok: {data['tier_a_regex_round_trip_ok']}\n")
    buf.write(f"- violations: {len(data['violations'])}\n\n")
    buf.write("## Canonical schema\n\n")
    buf.write(f"```\n{data['canonical_format_string']}\n```\n\n")
    buf.write("## Per-site\n\n")
    buf.write("| site | exists | emit calls | fmt ok |\n")
    buf.write("|---|---|---|---|\n")
    canonical_fmt = data["canonical_format_string"]
    for r in data["site_rows"]:
        fmt_ok = r.get("format_string") == canonical_fmt
        buf.write(f"| `{r['site']}` | {r['exists']} | "
                  f"{r.get('emit_call_count','?')} | {fmt_ok} |\n")
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
        w.writerow(["site", "exists", "emit_call_count", "format_string_ok"])
        canonical_fmt = data["canonical_format_string"]
        for r in data["site_rows"]:
            w.writerow([
                r["site"],
                int(r["exists"]),
                r.get("emit_call_count", ""),
                int(r.get("format_string") == canonical_fmt),
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
    print(f"[lit-faith-sideband-schema] status={data['status']} "
          f"fields={data['schema_field_count']} "
          f"sites={data['emit_site_count']} "
          f"violations={len(data['violations'])}")
    return 0 if not data["violations"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
