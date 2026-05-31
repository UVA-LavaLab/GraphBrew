"""Gate 278 — receiver CLI registry.

Locks the argparse CLI surface of the two paper-matrix RECEIVERS:

* ``scripts/experiments/ecg/proof_matrix.py`` (parse_args, 17 flags)
* ``scripts/experiments/ecg/roi_matrix.py``   (parse_args, 48 flags)

Gate 277 locked the SENDER side — the ``--flag`` string literals built
into argv lists by ``paper_pipeline.run_profile`` and
``final_paper_run.make_{proof,roi}_job``. Gate 278 locks the RECEIVER
side — the live ``parser.add_argument(...)`` calls in each receiver's
``parse_args()`` — and adds a cross-side parity check (rule E7) that
verifies every sender-side flag is present in the corresponding
receiver's registered surface. Together gates 277 + 278 guarantee
**both** sides of every subprocess invocation agree on the flag
contract, so renames + removals on either side fail loud at audit
time rather than silently at job time.

Catches the silent-drift cases gate 277 alone can't catch:

* Someone renames ``--out-dir`` → ``--output-dir`` in BOTH sender and
  receiver simultaneously (gate 277 passes because the sender still
  builds a valid argv; gate 278 catches it because the registry pin
  on receiver side still expects ``--out-dir``).
* Someone adds a new optional flag to ``proof_matrix.parse_args``
  without teaching ``make_proof_job`` to pass it (no runtime error —
  argparse just uses the default — but the receiver's contract has
  silently widened and the registry pin fails).
* Someone changes ``--l3-sizes`` from ``nargs="+"`` to a single value
  in ``roi_matrix.parse_args`` (every existing sender that passes
  multiple values silently swallows only the last one). E5 catches.
* Someone changes ``--allow-gem5-ecg-pfx`` from ``store_true`` to
  ``store`` (sender stops appending value-less flag; receiver gets
  None default). E4 catches.

7 rules E1-E7:

* **E1** — every registered receiver module exists and ast.parses.
* **E2** — every registered receiver has a ``parse_args()`` top-level fn.
* **E3** — every registered flag exists in live ``add_argument`` calls.
* **E4** — every registered flag ``action`` matches live (default ``store``).
* **E5** — every registered flag ``nargs`` matches live.
* **E6** — registry exhaustive — no surprise live flag in receiver.
* **E7** — **cross-side parity:** every sender-side flag from gate 277
  (``make_proof_job`` → ``proof_matrix``,
  ``make_roi_job`` → ``roi_matrix``) is present in the corresponding
  receiver registry.
"""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]


PROOF_PATH = "scripts/experiments/ecg/proof_matrix.py"
ROI_PATH = "scripts/experiments/ecg/roi_matrix.py"


PROOF_FLAGS: list[dict] = [
    {"flag": "--ablations",      "action": "store", "nargs": "+"},
    {"flag": "--benchmarks",     "action": "store", "nargs": "+"},
    {"flag": "--bfs-options",    "action": "store"},
    {"flag": "--dry-run",        "action": "store_true"},
    {"flag": "--graph-path",     "action": "store"},
    {"flag": "--l1d-size",       "action": "store"},
    {"flag": "--l2-size",        "action": "store"},
    {"flag": "--l3-sizes",       "action": "store", "nargs": "+"},
    {"flag": "--l3-ways",        "action": "store"},
    {"flag": "--line-size",      "action": "store"},
    {"flag": "--no-build",       "action": "store_true"},
    {"flag": "--omp-threads",    "action": "store"},
    {"flag": "--out-dir",        "action": "store"},
    {"flag": "--pfx-window",     "action": "store"},
    {"flag": "--pr-options",     "action": "store"},
    {"flag": "--sssp-options",   "action": "store"},
    {"flag": "--timeout-cache",  "action": "store"},
]


ROI_FLAGS: list[dict] = [
    {"flag": "--all-policies",                       "action": "store_true"},
    {"flag": "--allow-gem5-ecg-pfx",                 "action": "store_true"},
    {"flag": "--allow-sniper-benchmark-workload",    "action": "store_true"},
    {"flag": "--allow-sniper-sg-kernel-workload",    "action": "store_true"},
    {"flag": "--benchmark",                          "action": "store"},
    {"flag": "--droplet-indirect-degree",            "action": "store"},
    {"flag": "--droplet-prefetch-degree",            "action": "store"},
    {"flag": "--droplet-stride-table-size",          "action": "store"},
    {"flag": "--dry-run",                            "action": "store_true"},
    {"flag": "--ecg-pfx-delivery",                   "action": "store"},
    {"flag": "--ecg-pfx-hint-filter",                "action": "store"},
    {"flag": "--ecg-pfx-lookahead",                  "action": "store"},
    {"flag": "--ecg-pfx-mode",                       "action": "store"},
    {"flag": "--ecg-pfx-window",                     "action": "store"},
    {"flag": "--l1d-size",                           "action": "store"},
    {"flag": "--l1d-ways",                           "action": "store"},
    {"flag": "--l2-size",                            "action": "store"},
    {"flag": "--l2-ways",                            "action": "store"},
    {"flag": "--l3-sizes",                           "action": "store", "nargs": "+"},
    {"flag": "--l3-ways",                            "action": "store"},
    {"flag": "--line-size",                          "action": "store"},
    {"flag": "--no-build",                           "action": "store_true"},
    {"flag": "--options",                            "action": "store"},
    {"flag": "--out-dir",                            "action": "store"},
    {"flag": "--policies",                           "action": "store", "nargs": "+"},
    {"flag": "--popt-active-columns",                "action": "store"},
    {"flag": "--popt-min-data-ways",                 "action": "store"},
    {"flag": "--popt-num-epochs",                    "action": "store"},
    {"flag": "--popt-property-bytes",                "action": "store"},
    {"flag": "--prefetcher",                         "action": "store"},
    {"flag": "--prefetcher-level",                   "action": "store"},
    {"flag": "--sniper-address-domain",              "action": "store"},
    {"flag": "--sniper-base-config",                 "action": "store"},
    {"flag": "--sniper-config",                      "action": "store", "nargs": "*"},
    {"flag": "--sniper-cores",                       "action": "store"},
    {"flag": "--sniper-enable-graph-policies",       "action": "store_true"},
    {"flag": "--sniper-frontend",                    "action": "store"},
    {"flag": "--sniper-memory-limit-gb",             "action": "store"},
    {"flag": "--sniper-mimicos-kernel-mb",           "action": "store"},
    {"flag": "--sniper-mimicos-memory-mb",           "action": "store"},
    {"flag": "--sniper-omp-wait-policy",             "action": "store"},
    {"flag": "--sniper-root",                        "action": "store"},
    {"flag": "--sniper-workload",                    "action": "store"},
    {"flag": "--suite",                              "action": "store"},
    {"flag": "--threads",                            "action": "store", "nargs": "+"},
    {"flag": "--timeout-cache",                      "action": "store"},
    {"flag": "--timeout-gem5",                       "action": "store"},
    {"flag": "--timeout-sniper",                     "action": "store"},
]


RECEIVER_CLI_REGISTRY: dict[str, list[dict]] = {
    PROOF_PATH: PROOF_FLAGS,
    ROI_PATH:   ROI_FLAGS,
}


# Cross-side parity (E7): which sender fn pairs with which receiver
SENDER_RECEIVER_PAIRS: list[tuple[str, str, str]] = [
    # (sender_module_path, sender_fn_name, receiver_module_path)
    ("scripts/experiments/ecg/final_paper_run.py", "make_proof_job", PROOF_PATH),
    ("scripts/experiments/ecg/final_paper_run.py", "make_roi_job",   ROI_PATH),
]


# --------------------------------------------------------------------
# AST helpers
# --------------------------------------------------------------------


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text("utf-8"))


def _top_level_fn(module: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in module.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return node  # type: ignore[return-value]
    return None


def _ast_const(node: ast.AST) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    return None


def _collect_add_argument_calls(fn: ast.FunctionDef) -> list[dict]:
    """Return [{flag, action, nargs}] for every parser.add_argument(...)."""
    out: list[dict] = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if not node.args:
            continue
        first = node.args[0]
        flag = _ast_const(first)
        if not isinstance(flag, str) or not flag.startswith("--"):
            continue
        action = "store"
        nargs: object | None = None
        for kw in node.keywords:
            if kw.arg == "action":
                v = _ast_const(kw.value)
                if isinstance(v, str):
                    action = v
            elif kw.arg == "nargs":
                v = _ast_const(kw.value)
                nargs = v
        entry: dict = {"flag": flag, "action": action}
        if nargs is not None:
            entry["nargs"] = nargs
        out.append(entry)
    return out


def _collect_sender_flag_literals(fn: ast.FunctionDef) -> set[str]:
    """Walk fn and return every --flag string literal (gate 277 style)."""
    out: set[str] = set()
    for node in ast.walk(fn):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.value.startswith("--"):
                out.add(node.value)
    return out


# --------------------------------------------------------------------
# Rules
# --------------------------------------------------------------------


def _check_e1_importable() -> list[dict]:
    out = []
    for rel in RECEIVER_CLI_REGISTRY:
        p = REPO_ROOT / rel
        if not p.is_file():
            out.append({"rule": "E1", "path": rel, "issue": "missing"})
            continue
        try:
            _parse_module(p)
        except SyntaxError as exc:
            out.append({"rule": "E1", "path": rel,
                        "issue": f"syntax error: {exc}"})
    return out


def _check_e2_parse_args_present() -> list[dict]:
    out = []
    for rel in RECEIVER_CLI_REGISTRY:
        p = REPO_ROOT / rel
        if not p.is_file():
            continue
        mod = _parse_module(p)
        if _top_level_fn(mod, "parse_args") is None:
            out.append({"rule": "E2", "path": rel,
                        "issue": "no parse_args() fn"})
    return out


def _live_flags(rel: str) -> list[dict]:
    p = REPO_ROOT / rel
    if not p.is_file():
        return []
    mod = _parse_module(p)
    fn = _top_level_fn(mod, "parse_args")
    if fn is None:
        return []
    return _collect_add_argument_calls(fn)


def _check_e3_flag_presence() -> list[dict]:
    out = []
    for rel, want_flags in RECEIVER_CLI_REGISTRY.items():
        live = _live_flags(rel)
        live_set = {e["flag"] for e in live}
        for entry in want_flags:
            if entry["flag"] not in live_set:
                out.append({"rule": "E3", "path": rel,
                            "missing_flag": entry["flag"]})
    return out


def _check_e4_action_match() -> list[dict]:
    out = []
    for rel, want_flags in RECEIVER_CLI_REGISTRY.items():
        live = {e["flag"]: e for e in _live_flags(rel)}
        for entry in want_flags:
            got = live.get(entry["flag"])
            if got is None:
                continue
            if got["action"] != entry["action"]:
                out.append({"rule": "E4", "path": rel,
                            "flag": entry["flag"],
                            "want_action": entry["action"],
                            "got_action": got["action"]})
    return out


def _check_e5_nargs_match() -> list[dict]:
    out = []
    for rel, want_flags in RECEIVER_CLI_REGISTRY.items():
        live = {e["flag"]: e for e in _live_flags(rel)}
        for entry in want_flags:
            got = live.get(entry["flag"])
            if got is None:
                continue
            want_nargs = entry.get("nargs")
            got_nargs = got.get("nargs")
            if want_nargs != got_nargs:
                out.append({"rule": "E5", "path": rel,
                            "flag": entry["flag"],
                            "want_nargs": want_nargs,
                            "got_nargs": got_nargs})
    return out


def _check_e6_exhaustive() -> list[dict]:
    out = []
    for rel, want_flags in RECEIVER_CLI_REGISTRY.items():
        want_set = {e["flag"] for e in want_flags}
        live_set = {e["flag"] for e in _live_flags(rel)}
        for extra in sorted(live_set - want_set):
            out.append({"rule": "E6", "path": rel, "flag": extra,
                        "issue": "live flag not in registry"})
    return out


def _check_e7_cross_side_parity() -> list[dict]:
    """Every sender-side --flag literal must be in receiver registry."""
    out = []
    for sender_path, sender_fn, recv_path in SENDER_RECEIVER_PAIRS:
        sp = REPO_ROOT / sender_path
        if not sp.is_file():
            out.append({"rule": "E7", "sender_path": sender_path,
                        "issue": "sender missing"})
            continue
        try:
            mod = _parse_module(sp)
        except SyntaxError as exc:
            out.append({"rule": "E7", "sender_path": sender_path,
                        "issue": f"sender syntax error: {exc}"})
            continue
        fn = _top_level_fn(mod, sender_fn)
        if fn is None:
            out.append({"rule": "E7", "sender_path": sender_path,
                        "sender_fn": sender_fn,
                        "issue": "sender fn missing"})
            continue
        sender_flags = _collect_sender_flag_literals(fn)
        receiver_set = {e["flag"]
                        for e in RECEIVER_CLI_REGISTRY.get(recv_path, [])}
        for flag in sorted(sender_flags - receiver_set):
            out.append({"rule": "E7", "sender_path": sender_path,
                        "sender_fn": sender_fn,
                        "receiver_path": recv_path,
                        "missing_in_receiver": flag})
    return out


# --------------------------------------------------------------------
# Audit + emit
# --------------------------------------------------------------------


def audit() -> dict:
    viols: list[dict] = []
    viols += _check_e1_importable()
    viols += _check_e2_parse_args_present()
    viols += _check_e3_flag_presence()
    viols += _check_e4_action_match()
    viols += _check_e5_nargs_match()
    viols += _check_e6_exhaustive()
    viols += _check_e7_cross_side_parity()

    counts = {
        "receivers": len(RECEIVER_CLI_REGISTRY),
        "proof_flags": len(PROOF_FLAGS),
        "roi_flags": len(ROI_FLAGS),
        "total_flags": sum(len(v) for v in RECEIVER_CLI_REGISTRY.values()),
        "cross_pairs": len(SENDER_RECEIVER_PAIRS),
    }
    return {
        "schema": "lit-faith-receiver-cli-registry/1",
        "status": "active",
        "counts": counts,
        "registry": {rel: list(flags)
                     for rel, flags in RECEIVER_CLI_REGISTRY.items()},
        "cross_pairs": [
            {"sender_path": sp, "sender_fn": fn, "receiver_path": rp}
            for sp, fn, rp in SENDER_RECEIVER_PAIRS
        ],
        "rules": {
            "E1": "receiver module importable (ast parses)",
            "E2": "parse_args() top-level fn present",
            "E3": "every registered flag exists in live add_argument calls",
            "E4": "every registered flag action matches live",
            "E5": "every registered flag nargs matches live",
            "E6": "registry exhaustive — no surprise live flag",
            "E7": "cross-side parity: sender --flag literal ⊆ receiver registry",
        },
        "violations": viols,
    }


def write_json(doc: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n", "utf-8")


def write_md(doc: dict, path: Path) -> None:
    lines: list[str] = [
        "# Receiver CLI registry (gate 278)",
        "",
        "_Auto-generated by `lit_faith_receiver_cli_registry.py`._",
        "",
        f"- receivers: **{doc['counts']['receivers']}**",
        f"- proof_matrix flags: **{doc['counts']['proof_flags']}**",
        f"- roi_matrix flags: **{doc['counts']['roi_flags']}**",
        f"- total flags: **{doc['counts']['total_flags']}**",
        f"- cross-side pairs (E7): **{doc['counts']['cross_pairs']}**",
        f"- violations: **{len(doc['violations'])}**",
        "",
    ]
    for rel, flags in sorted(doc["registry"].items()):
        lines.append(f"## `{rel}`")
        lines.append("")
        lines.append("| flag | action | nargs |")
        lines.append("| --- | --- | --- |")
        for e in flags:
            lines.append(f"| `{e['flag']}` | `{e['action']}` | "
                         f"`{e.get('nargs', '—')}` |")
        lines.append("")
    lines.append("## Cross-side parity pairs (E7)")
    lines.append("")
    lines.append("| sender | sender_fn | receiver |")
    lines.append("| --- | --- | --- |")
    for pair in doc["cross_pairs"]:
        lines.append(f"| `{pair['sender_path']}` | "
                     f"`{pair['sender_fn']}` | "
                     f"`{pair['receiver_path']}` |")
    lines.append("")
    if doc["violations"]:
        lines.append("## Violations")
        lines.append("")
        for v in doc["violations"]:
            lines.append(f"- {json.dumps(v, sort_keys=True)}")
    else:
        lines.append("## ✅ No violations")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", "utf-8")


def write_csv(doc: dict, path: Path) -> None:
    rows = ["path,flag,action,nargs"]
    for rel, flags in sorted(doc["registry"].items()):
        for e in flags:
            rows.append(
                f"{rel},{e['flag']},{e['action']},{e.get('nargs','')}"
            )
    for v in doc["violations"]:
        rows.append(
            f"violation,{v.get('path', v.get('sender_path', ''))},"
            f"{v.get('flag', v.get('missing_in_receiver', ''))},"
            f"{v.get('rule', '')}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows) + "\n", "utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--md-out", type=Path)
    ap.add_argument("--csv-out", type=Path)
    args = ap.parse_args(list(argv) if argv is not None else None)

    doc = audit()
    if args.json_out:
        write_json(doc, args.json_out)
    if args.md_out:
        write_md(doc, args.md_out)
    if args.csv_out:
        write_csv(doc, args.csv_out)

    print(
        f"[lit-faith-receiver-cli-registry] "
        f"status={doc['status']} "
        f"receivers={doc['counts']['receivers']} "
        f"flags={doc['counts']['total_flags']} "
        f"cross_pairs={doc['counts']['cross_pairs']} "
        f"violations={len(doc['violations'])}"
    )
    return 1 if doc["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
