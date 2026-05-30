"""Gate 261 — ECG arm catalog registry.

Eighth in the vocabulary-lock series (252 SBATCH, 255 policy, 256
profile, 257 backend, 258 graph, 259 build, 260 CLI, **261 arm-
catalog**). Locks the cross-file consistency of every ECG arm and
paper-shipping policy across the THREE places they appear:

  1. ``lit_faith_policy_registry.CANONICAL_ECG_ARMS`` — the
     registry side; ``ECG:`` prefixed namespace, 9 entries.
  2. ``paper_pipeline.POLICY_ORDER`` + ``POLICY_LABELS`` +
     ``POLICY_DESCRIPTIONS`` + ``POLICY_COLORS`` + ``POLICY_HATCHES``
     — the paper side; underscore-namespace (e.g. ``ECG_DBG_ONLY``),
     9 paper-shipping policies.
  3. ``proof_matrix.ABLATIONS`` + ``ADAPTIVE_SELECTORS`` — the
     measurement side; mixed-case proof-matrix labels (e.g.
     ``ECG_DBG_only``) whose ``policy`` field points back at the
     registry namespace (``ECG:DBG_ONLY``).

The three namespaces drift naturally over time. Two real examples
this gate catches:

* A new ``ECG_DBG_PRIMARY_CHARGED`` ablation is added to
  ``ABLATIONS`` but the paper-side ``POLICY_DESCRIPTIONS`` map
  forgets to add a description — every figure that joins on
  paper_label and policy will silently show a blank legend
  caption.
* A contributor renames ``ECG:DBG_PRIMARY`` to
  ``ECG:DBG_HEAD`` in the registry but the proof-matrix
  ``ECG_DBG_POPT`` ablation still references the old string —
  the ablation row is dropped from every chart with no error
  message; the missing bar is invisible in a grid plot.

7 rules A1-A7:
  A1: every ``paper_pipeline.POLICY_ORDER`` entry that is NOT in
      the four hard baselines (LRU / SRRIP / GRASP / POPT) has a
      matching registry-side entry (after namespace translation
      ``ECG_<X>`` → ``ECG:<X>`` and ``POPT_CHARGED`` → ``POPT_CHARGED``).
  A2: every ``paper_pipeline.POLICY_ORDER`` entry has a non-empty
      label in POLICY_LABELS, a non-empty description in
      POLICY_DESCRIPTIONS, and a color in POLICY_COLORS.
  A3: every "charged" paper policy (suffix ``_CHARGED``) has a
      hatch pattern in POLICY_HATCHES — required for grayscale
      legibility.
  A4: every ``proof_matrix.ABLATIONS`` entry's ``policy`` field is
      a known policy: either a canonical baseline (LRU/SRRIP/GRASP/
      POPT) or a registry ECG-arm key. No drift to ``ECG:NEW_ARM``
      without a registry entry.
  A5: every ``proof_matrix.ADAPTIVE_SELECTORS`` entry's
      ``candidates`` field references a real ABLATION label (no
      dangling string).
  A6: ``proof_matrix.ABLATIONS`` has no duplicate ``label``s
      (would cause silent row-merging in the rollup CSV).
  A7: the ECG-arm chain rule: every registry ECG arm whose parent
      is ``ECG`` has at least one ablation in ABLATIONS referencing
      it (no orphan registry entries); the inverse is enforced by
      A4.
"""
from __future__ import annotations

import argparse
import csv
import importlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ECG_DIR = REPO_ROOT / "scripts" / "experiments" / "ecg"

CANONICAL_BASELINES: tuple[str, ...] = ("LRU", "SRRIP", "GRASP", "POPT")


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _paper_to_registry_key(paper_name: str) -> str:
    """``ECG_DBG_PRIMARY_CHARGED`` → ``ECG:DBG_PRIMARY_CHARGED``;
    ``POPT_CHARGED`` and the baselines pass through unchanged."""
    if paper_name.startswith("ECG_"):
        return "ECG:" + paper_name[len("ECG_"):]
    return paper_name


def audit() -> dict[str, Any]:
    pp = _load("gate261_pp", ECG_DIR / "paper_pipeline.py")
    pm = _load("gate261_pm", ECG_DIR / "proof_matrix.py")
    pr = _load("gate261_pr", ECG_DIR / "lit_faith_policy_registry.py")

    arms_keys = set(pr.CANONICAL_ECG_ARMS.keys())  # noqa: PD003
    arm_entries = pr.CANONICAL_ECG_ARMS
    violations: list[dict[str, Any]] = []

    # A1: every POLICY_ORDER non-baseline maps to a registry entry.
    paper_to_registry: dict[str, str] = {}
    for pol in pp.POLICY_ORDER:
        if pol in CANONICAL_BASELINES:
            continue
        key = _paper_to_registry_key(pol)
        paper_to_registry[pol] = key
        if key not in arms_keys:
            violations.append({
                "rule": "A1",
                "paper_policy": pol,
                "expected_registry_key": key,
                "msg": (
                    f"paper_pipeline.POLICY_ORDER has {pol!r} but "
                    f"lit_faith_policy_registry.CANONICAL_ECG_ARMS "
                    f"is missing {key!r}"
                ),
            })

    # A2: every POLICY_ORDER has label + description + color.
    for pol in pp.POLICY_ORDER:
        for tbl_name in ("POLICY_LABELS", "POLICY_DESCRIPTIONS",
                         "POLICY_COLORS"):
            tbl = getattr(pp, tbl_name)
            val = tbl.get(pol)
            if not val:
                violations.append({
                    "rule": "A2",
                    "paper_policy": pol,
                    "table": tbl_name,
                    "msg": (
                        f"paper_pipeline.{tbl_name} is missing or "
                        f"empty for {pol!r}"
                    ),
                })

    # A3: every charged policy has a hatch.
    for pol in pp.POLICY_ORDER:
        if pol.endswith("_CHARGED") and pol not in pp.POLICY_HATCHES:
            violations.append({
                "rule": "A3",
                "paper_policy": pol,
                "msg": (
                    f"charged paper policy {pol!r} has no POLICY_HATCHES "
                    f"entry (required for grayscale legibility)"
                ),
            })

    # A4: every ABLATIONS.policy is canonical baseline or registry arm.
    ablation_labels: list[str] = []
    label_count: dict[str, int] = {}
    ablation_policies: dict[str, str] = {}
    for ab in pm.ABLATIONS:
        ablation_labels.append(ab.label)
        ablation_policies[ab.label] = ab.policy
        label_count[ab.label] = label_count.get(ab.label, 0) + 1
        ok = ab.policy in CANONICAL_BASELINES or ab.policy in arms_keys
        if not ok:
            violations.append({
                "rule": "A4",
                "ablation_label": ab.label,
                "policy": ab.policy,
                "msg": (
                    f"proof_matrix.ABLATIONS[{ab.label!r}].policy = "
                    f"{ab.policy!r} not in baselines or "
                    f"CANONICAL_ECG_ARMS"
                ),
            })

    # A5: every adaptive selector's candidates reference real labels.
    label_set = set(ablation_labels)
    for sel in pm.ADAPTIVE_SELECTORS:
        for cand in sel.candidates:
            if cand not in label_set:
                violations.append({
                    "rule": "A5",
                    "selector": sel.label,
                    "candidate": cand,
                    "msg": (
                        f"proof_matrix.ADAPTIVE_SELECTORS[{sel.label!r}]"
                        f" references unknown ablation label {cand!r}"
                    ),
                })

    # A6: no duplicate ablation labels.
    for lbl, n in label_count.items():
        if n > 1:
            violations.append({
                "rule": "A6",
                "ablation_label": lbl,
                "count": n,
                "msg": f"duplicate ablation label {lbl!r} (count={n})",
            })

    # A7: every ECG-parented registry arm has at least one ablation.
    # ``_CHARGED`` arms are post-hoc projections of their uncharged
    # parent (e.g. ``ECG:DBG_PRIMARY_CHARGED`` shares cache_sim runs
    # with ``ECG:DBG_PRIMARY``; see paper_pipeline.py PAIRS), so they
    # are satisfied by their parent's ablation count.
    arm_ablation_count: dict[str, int] = {a: 0 for a in arms_keys}
    for ab in pm.ABLATIONS:
        if ab.policy in arm_ablation_count:
            arm_ablation_count[ab.policy] += 1

    def _ablation_satisfied(arm: str) -> bool:
        if arm_ablation_count.get(arm, 0) > 0:
            return True
        if arm.endswith("_CHARGED"):
            parent = arm[: -len("_CHARGED")]
            return arm_ablation_count.get(parent, 0) > 0
        return False

    for arm, meta in arm_entries.items():
        if meta.get("parent") != "ECG":
            continue
        if not _ablation_satisfied(arm):
            violations.append({
                "rule": "A7",
                "registry_arm": arm,
                "msg": (
                    f"registry ECG arm {arm!r} has no matching "
                    f"proof_matrix.ABLATIONS entry (and no parent "
                    f"_CHARGED projection)"
                ),
            })

    return {
        "status": "active",
        "n_paper_policies": len(pp.POLICY_ORDER),
        "n_paper_charged": sum(
            1 for p in pp.POLICY_ORDER if p.endswith("_CHARGED")
        ),
        "n_registry_arms": len(arms_keys),
        "n_ablations": len(pm.ABLATIONS),
        "n_adaptive_selectors": len(pm.ADAPTIVE_SELECTORS),
        "paper_to_registry": dict(sorted(paper_to_registry.items())),
        "policy_order": list(pp.POLICY_ORDER),
        "ablations": [
            {"label": ab.label, "group": ab.group, "policy": ab.policy,
             "pfx_mode": ab.pfx_mode, "pfx_lookahead": ab.pfx_lookahead,
             "note": ab.note}
            for ab in pm.ABLATIONS
        ],
        "adaptive_selectors": [
            {"label": s.label, "candidates": list(s.candidates),
             "note": s.note}
            for s in pm.ADAPTIVE_SELECTORS
        ],
        "registry_arms": {
            arm: {
                "parent": meta.get("parent"),
                "purpose": meta.get("purpose"),
                "ablation_count": arm_ablation_count[arm],
            }
            for arm, meta in sorted(arm_entries.items())
        },
        "rules": {
            "A1": "every paper_pipeline.POLICY_ORDER non-baseline entry has a CANONICAL_ECG_ARMS entry after namespace translation",
            "A2": "every POLICY_ORDER entry has POLICY_LABELS + POLICY_DESCRIPTIONS + POLICY_COLORS",
            "A3": "every paper policy with suffix _CHARGED has a POLICY_HATCHES entry",
            "A4": "every proof_matrix.ABLATIONS.policy is a canonical baseline or a CANONICAL_ECG_ARMS key",
            "A5": "every proof_matrix.ADAPTIVE_SELECTORS.candidates entry references a real ablation label",
            "A6": "proof_matrix.ABLATIONS has no duplicate labels",
            "A7": "every CANONICAL_ECG_ARMS entry with parent='ECG' has at least one ABLATIONS row, OR (for _CHARGED arms) its uncharged parent has one",
        },
        "violations": violations,
    }


def _emit_json(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def _emit_md(data: dict[str, Any], path: Path) -> None:
    lines: list[str] = []
    lines.append("# Gate 261 — ECG arm catalog registry")
    lines.append("")
    lines.append(f"Status: **{data['status']}**")
    lines.append("")
    lines.append("## Totals")
    for k in ("n_paper_policies", "n_paper_charged", "n_registry_arms",
              "n_ablations", "n_adaptive_selectors"):
        lines.append(f"- {k}: {data[k]}")
    lines.append("")
    lines.append("## Rules")
    for rid, desc in data["rules"].items():
        lines.append(f"- **{rid}** — {desc}")
    lines.append("")
    lines.append("## Paper-policy → registry-arm map")
    lines.append("")
    lines.append("| paper_policy | registry_key |")
    lines.append("|---|---|")
    for pp, rk in data["paper_to_registry"].items():
        lines.append(f"| `{pp}` | `{rk}` |")
    lines.append("")
    lines.append("## Ablations")
    lines.append("")
    lines.append("| label | group | policy | pfx_mode | lookahead |")
    lines.append("|---|---|---|---:|---:|")
    for a in data["ablations"]:
        lines.append(
            f"| `{a['label']}` | `{a['group']}` | `{a['policy']}` | "
            f"{a['pfx_mode']} | {a['pfx_lookahead']} |"
        )
    lines.append("")
    lines.append("## Adaptive selectors")
    lines.append("")
    for s in data["adaptive_selectors"]:
        lines.append(
            f"- **{s['label']}** ← {', '.join(f'`{c}`' for c in s['candidates'])}"
        )
    lines.append("")
    lines.append("## Registry arms")
    lines.append("")
    lines.append("| arm | parent | ablation_count |")
    lines.append("|---|---|---:|")
    for arm, meta in data["registry_arms"].items():
        lines.append(
            f"| `{arm}` | `{meta['parent']}` | {meta['ablation_count']} |"
        )
    lines.append("")
    lines.append("## Violations")
    lines.append("")
    if data["violations"]:
        for v in data["violations"]:
            lines.append(f"- {v}")
    else:
        lines.append("None.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def _emit_csv(data: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("kind", "name", "extra"))
        for pp, rk in data["paper_to_registry"].items():
            w.writerow(("paper_policy", pp, rk))
        for a in data["ablations"]:
            w.writerow(("ablation", a["label"], a["policy"]))
        for s in data["adaptive_selectors"]:
            w.writerow(("selector", s["label"], "|".join(s["candidates"])))
        for arm, meta in data["registry_arms"].items():
            w.writerow(("registry_arm", arm, meta["parent"] or ""))
        for v in data["violations"]:
            w.writerow(("violation", str(v.get("rule", "")), str(v)))


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
        f"[lit-faith-arm-catalog] status={data['status']} "
        f"paper_policies={data['n_paper_policies']} "
        f"registry_arms={data['n_registry_arms']} "
        f"ablations={data['n_ablations']} "
        f"selectors={data['n_adaptive_selectors']} "
        f"violations={len(data['violations'])}"
    )
    return 1 if data["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
