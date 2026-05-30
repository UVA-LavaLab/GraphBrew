#!/usr/bin/env python3
"""LIT-Stat (gate 230): statistical-sanity audit of the lit-faith corpus.

For every per-claim row in ``literature_faithfulness_postfix.json``, the
comparator emits a ``delta_pct`` derived from the two miss-rate columns
that the row compares (LRU-vs-policy or POPT-vs-GRASP). This audit
re-derives the delta directly from the miss-rate columns and locks:

  * No NaN / inf / negative miss rates, no miss rates above 1.
  * The recomputed delta matches the stored ``delta_pct`` within
    rounding (4 dp) — a divergence means the comparator emitted
    inconsistent fields or someone hand-edited the JSON.
  * The recomputed delta carries the same sign as the stored field
    (catches sign-flip bugs that get rounded into the noise floor).
  * ``status`` is one of the allowed band labels and is consistent
    with the recomputed delta + the claim's tolerance bounds.
  * For LRU-vs-policy rows, both ``lru_miss_rate`` and
    ``policy_miss_rate`` are populated and 0 ≤ rate ≤ 1.
  * For POPT-vs-GRASP rows, both ``grasp_miss_rate`` and
    ``popt_miss_rate`` are populated and 0 ≤ rate ≤ 1.
  * Per-app delta distributions are nontrivial — at least one app
    must show absolute delta > 1 pp.

Emits a JSON / Markdown / CSV summary so the gate is reviewable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


ALLOWED_STATUSES = {
    "ok",
    "within_tolerance",
    "disagree",
    "known_deviation",
    "missing",
    "insufficient_data",
}

LRU_ROW_KEYS    = ("lru_miss_rate", "policy_miss_rate")
POPT_ROW_KEYS   = ("grasp_miss_rate", "popt_miss_rate")

DELTA_ROUNDING_TOL_PP = 0.001
SIGN_NOISE_FLOOR_PP   = 0.01


def _row_kind(row: dict[str, Any]) -> str:
    """Branch on the row's policy field, not on which keys are present —
    `POPT_NEAR_GRASP_IF_BIG_GAP` writes the same keys as `POPT_GE_GRASP`
    but stores `delta_pct = abs(popt - grasp) * 100` and the signed form
    in `signed_delta_pct`."""
    policy = row.get("policy")
    if policy == "POPT_NEAR_GRASP_IF_BIG_GAP":
        return "popt_near_grasp"
    if policy == "POPT_GE_GRASP":
        return "popt_ge_grasp"
    if all(k in row for k in LRU_ROW_KEYS):
        return "lru_vs_policy"
    return "unknown"


def _miss_rate_pair(row: dict[str, Any], kind: str) -> tuple[float, float]:
    if kind == "lru_vs_policy":
        return float(row["lru_miss_rate"]), float(row["policy_miss_rate"])
    if kind in ("popt_ge_grasp", "popt_near_grasp"):
        return float(row["grasp_miss_rate"]), float(row["popt_miss_rate"])
    raise ValueError(f"unknown row kind {kind!r}")


def _recompute_delta(row: dict[str, Any], kind: str) -> float | None:
    """Mirror the comparator's per-branch arithmetic:
       lru_vs_policy:   (policy - lru) * 100             (signed)
       popt_ge_grasp:   (popt - grasp) * 100             (signed)
       popt_near_grasp: abs((popt - grasp) * 100)        (magnitude)
    """
    if kind == "lru_vs_policy":
        a, b = float(row["lru_miss_rate"]), float(row["policy_miss_rate"])
        return (b - a) * 100.0
    if kind == "popt_ge_grasp":
        a, b = float(row["grasp_miss_rate"]), float(row["popt_miss_rate"])
        return (b - a) * 100.0
    if kind == "popt_near_grasp":
        a, b = float(row["grasp_miss_rate"]), float(row["popt_miss_rate"])
        return abs((b - a) * 100.0)
    return None


def _recompute_signed_delta(row: dict[str, Any], kind: str) -> float | None:
    """Always the signed (policy/popt - reference) * 100. Used for
    `signed_delta_pct` consistency checks on POPT_NEAR rows and for
    sign-flip detection on the other kinds."""
    if kind == "lru_vs_policy":
        a, b = float(row["lru_miss_rate"]), float(row["policy_miss_rate"])
        return (b - a) * 100.0
    if kind in ("popt_ge_grasp", "popt_near_grasp"):
        a, b = float(row["grasp_miss_rate"]), float(row["popt_miss_rate"])
        return (b - a) * 100.0
    return None


def _is_finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def build_audit(lit_path: Path) -> dict[str, Any]:
    payload = json.loads(lit_path.read_text(encoding="utf-8"))
    rows = payload["per_claim"]

    nan_inf:        list[dict[str, Any]] = []
    miss_oob:       list[dict[str, Any]] = []
    delta_mismatch: list[dict[str, Any]] = []
    sign_mismatch:  list[dict[str, Any]] = []
    signed_delta_mismatch: list[dict[str, Any]] = []
    status_bad:     list[dict[str, Any]] = []
    status_inconsistent: list[dict[str, Any]] = []
    unknown_kind:   list[dict[str, Any]] = []
    missing_pair:   list[dict[str, Any]] = []
    abs_delta_by_app:    dict[str, list[float]] = defaultdict(list)
    counts_by_kind:      dict[str, int] = defaultdict(int)
    counts_by_status:    dict[str, int] = defaultdict(int)

    for idx, row in enumerate(rows):
        cite = {
            "row_index": idx,
            "graph":   row.get("graph"),
            "app":     row.get("app"),
            "policy":  row.get("policy"),
            "l3_size": row.get("l3_size"),
        }

        kind = _row_kind(row)
        counts_by_kind[kind] += 1

        status = row.get("status")
        counts_by_status[status] += 1
        if status not in ALLOWED_STATUSES:
            status_bad.append({**cite, "status": status})

        if kind == "unknown":
            unknown_kind.append(cite)
            continue

        if kind == "lru_vs_policy":
            a_key, b_key = LRU_ROW_KEYS
        else:
            a_key, b_key = POPT_ROW_KEYS
        a, b = row.get(a_key), row.get(b_key)
        if a is None or b is None:
            missing_pair.append({**cite, "kind": kind,
                                 a_key: a, b_key: b})
            continue
        if not (_is_finite(a) and _is_finite(b)):
            nan_inf.append({**cite, a_key: a, b_key: b})
            continue
        if not (0.0 <= float(a) <= 1.0 and 0.0 <= float(b) <= 1.0):
            miss_oob.append({**cite, a_key: a, b_key: b})
            continue

        delta_stored = row.get("delta_pct")
        if delta_stored is None:
            continue
        if not _is_finite(delta_stored):
            nan_inf.append({**cite, "delta_pct": delta_stored})
            continue

        delta_recomp = _recompute_delta(row, kind)
        diff = abs(float(delta_stored) - float(delta_recomp))
        if diff > DELTA_ROUNDING_TOL_PP:
            delta_mismatch.append({
                **cite,
                "kind":          kind,
                "stored":        float(delta_stored),
                "recomputed":    round(float(delta_recomp), 6),
                "abs_diff_pp":   round(diff, 6),
            })

        # Sign-flip check applies only to signed-delta kinds.
        if kind in ("lru_vs_policy", "popt_ge_grasp"):
            if (abs(float(delta_stored)) > SIGN_NOISE_FLOOR_PP and
                    abs(float(delta_recomp)) > SIGN_NOISE_FLOOR_PP):
                if math.copysign(1.0, float(delta_stored)) != math.copysign(
                        1.0, float(delta_recomp)):
                    sign_mismatch.append({
                        **cite,
                        "kind":       kind,
                        "stored":     float(delta_stored),
                        "recomputed": round(float(delta_recomp), 6),
                    })

        # signed_delta_pct consistency on POPT_NEAR rows.
        if kind == "popt_near_grasp" and "signed_delta_pct" in row:
            signed_stored = row["signed_delta_pct"]
            if _is_finite(signed_stored):
                signed_recomp = _recompute_signed_delta(row, kind)
                if abs(float(signed_stored) - float(signed_recomp)) \
                        > DELTA_ROUNDING_TOL_PP:
                    signed_delta_mismatch.append({
                        **cite,
                        "stored":     float(signed_stored),
                        "recomputed": round(float(signed_recomp), 6),
                    })
                # The unsigned delta_pct should equal |signed|.
                if abs(abs(float(signed_stored)) - float(delta_stored)) \
                        > DELTA_ROUNDING_TOL_PP:
                    signed_delta_mismatch.append({
                        **cite,
                        "reason":       "|signed| != delta_pct",
                        "signed":       float(signed_stored),
                        "delta_pct":    float(delta_stored),
                    })

        # Status-vs-delta consistency: 'ok'/'within_tolerance' rows
        # should not exceed `max_abs_delta_pct + tolerance_pct`. We use
        # the unsigned-magnitude bound (which `_classify` enforces) and
        # skip rows that have a `min_abs_delta_pct` floor (those are
        # signed-band claims and the magnitude bound alone is too
        # generous). POPT_NEAR_GRASP rows enforce the bound only in
        # the phase-transition regime (grasp_gain_vs_lru > 10 pp) and
        # only when POPT is worse than GRASP (signed delta > 0) —
        # mirror that branch.
        tol     = float(row.get("tolerance_pct") or 0.0)
        max_abs = row.get("max_abs_delta_pct")
        min_abs = row.get("min_abs_delta_pct")
        if (status in ("ok", "within_tolerance") and
                max_abs is not None and min_abs is None):
            assertion_active = True
            if kind == "popt_near_grasp":
                gain = row.get("grasp_gain_vs_lru_pct")
                signed = row.get("signed_delta_pct")
                if (not _is_finite(gain) or float(gain) <= 10.0 or
                        not _is_finite(signed) or float(signed) <= 0.0):
                    assertion_active = False
            if assertion_active:
                magnitude = abs(float(delta_recomp))
                if magnitude > float(max_abs) + tol + DELTA_ROUNDING_TOL_PP:
                    status_inconsistent.append({
                        **cite,
                        "kind":        kind,
                        "status":      status,
                        "abs_delta":   round(magnitude, 6),
                        "max_abs":     float(max_abs),
                        "tolerance":   tol,
                    })

        if row.get("app"):
            abs_delta_by_app[row["app"]].append(abs(float(delta_recomp)))

    apps_with_signal = [
        app for app, vs in abs_delta_by_app.items() if max(vs) > 1.0
    ]
    apps_flat = [
        app for app, vs in abs_delta_by_app.items() if max(vs) <= 0.05
    ]

    summary = {
        "total_rows":            len(rows),
        "rows_lru_vs_policy":    counts_by_kind["lru_vs_policy"],
        "rows_popt_ge_grasp":    counts_by_kind["popt_ge_grasp"],
        "rows_popt_near_grasp":  counts_by_kind["popt_near_grasp"],
        "rows_unknown_kind":     counts_by_kind["unknown"],
        "rows_missing_pair":     len(missing_pair),
        "nan_inf":               len(nan_inf),
        "miss_rate_oob":         len(miss_oob),
        "delta_mismatch":        len(delta_mismatch),
        "sign_mismatch":         len(sign_mismatch),
        "signed_delta_mismatch": len(signed_delta_mismatch),
        "status_bad_label":      len(status_bad),
        "status_inconsistent":   len(status_inconsistent),
        "apps_count":            len(abs_delta_by_app),
        "apps_with_pp_signal":   sorted(apps_with_signal),
        "apps_flat":             sorted(apps_flat),
        "status_counts":         dict(sorted(counts_by_status.items())),
        "delta_rounding_tol_pp": DELTA_ROUNDING_TOL_PP,
        "sign_noise_floor_pp":   SIGN_NOISE_FLOOR_PP,
    }

    return {
        "schema_version":  1,
        "summary":         summary,
        "nan_inf":         nan_inf,
        "miss_rate_oob":   miss_oob,
        "delta_mismatch":  delta_mismatch,
        "sign_mismatch":   sign_mismatch,
        "signed_delta_mismatch": signed_delta_mismatch,
        "status_bad":      status_bad,
        "status_inconsistent": status_inconsistent,
        "unknown_kind":    unknown_kind,
        "missing_pair":    missing_pair,
    }


def _to_markdown(audit: dict[str, Any]) -> str:
    s = audit["summary"]
    lines = ["# Literature-faithfulness statistical-sanity audit (LIT-Stat)",
             "",
             "Re-derives `delta_pct` from the two miss-rate columns each row "
             "compares (LRU-vs-policy, POPT_GE_GRASP, POPT_NEAR_GRASP_IF_BIG_GAP) "
             "and checks for rounding drift, sign flips, out-of-bounds rates, "
             "NaN/inf, bad status labels, and status-vs-delta inconsistencies.",
             "",
             "## Summary",
             "",
             "| Metric | Value |",
             "|---|---|",
             f"| Total rows | {s['total_rows']} |",
             f"| LRU-vs-policy rows | {s['rows_lru_vs_policy']} |",
             f"| POPT_GE_GRASP rows | {s['rows_popt_ge_grasp']} |",
             f"| POPT_NEAR_GRASP rows | {s['rows_popt_near_grasp']} |",
             f"| Unknown-kind rows | {s['rows_unknown_kind']} |",
             f"| Missing-pair rows | {s['rows_missing_pair']} |",
             f"| NaN / inf values | {s['nan_inf']} |",
             f"| Miss-rate out of [0,1] | {s['miss_rate_oob']} |",
             f"| Delta mismatch (> {s['delta_rounding_tol_pp']} pp) | {s['delta_mismatch']} |",
             f"| Sign mismatch (above {s['sign_noise_floor_pp']} pp floor) | {s['sign_mismatch']} |",
             f"| Signed-delta inconsistencies | {s['signed_delta_mismatch']} |",
             f"| Bad status labels | {s['status_bad_label']} |",
             f"| Status-vs-delta inconsistencies | {s['status_inconsistent']} |",
             f"| Apps with > 1 pp signal | {len(s['apps_with_pp_signal'])} "
             f"({', '.join(s['apps_with_pp_signal'])}) |",
             f"| Apps flat (max abs Δ ≤ 0.05 pp) | {len(s['apps_flat'])} |",
             "",
             "## Status counts",
             "",
             "| Status | Count |",
             "|---|---|"]
    for status, n in s["status_counts"].items():
        lines.append(f"| `{status}` | {n} |")

    for section_key, label in [
        ("nan_inf",                "NaN / inf"),
        ("miss_rate_oob",          "Miss-rate out of [0,1]"),
        ("delta_mismatch",         "Delta mismatch"),
        ("sign_mismatch",          "Sign mismatch"),
        ("signed_delta_mismatch",  "Signed-delta inconsistencies"),
        ("status_bad",             "Bad status labels"),
        ("status_inconsistent",    "Status-vs-delta inconsistencies"),
        ("unknown_kind",           "Unknown row kind"),
        ("missing_pair",           "Missing miss-rate pair"),
    ]:
        rows = audit[section_key]
        if not rows:
            continue
        lines += ["", f"## {label} ({len(rows)})", "",
                  "| graph | app | policy | l3 | details |",
                  "|---|---|---|---|---|"]
        for r in rows[:50]:
            details = {k: v for k, v in r.items()
                       if k not in {"graph", "app", "policy", "l3_size",
                                    "row_index"}}
            lines.append(f"| {r.get('graph')} | {r.get('app')} | "
                         f"{r.get('policy')} | {r.get('l3_size')} | "
                         f"`{details}` |")

    return "\n".join(lines) + "\n"


def _to_csv(audit: dict[str, Any], csv_path: Path) -> None:
    s = audit["summary"]
    rows = [
        ("total_rows",            s["total_rows"]),
        ("rows_lru_vs_policy",    s["rows_lru_vs_policy"]),
        ("rows_popt_ge_grasp",    s["rows_popt_ge_grasp"]),
        ("rows_popt_near_grasp",  s["rows_popt_near_grasp"]),
        ("rows_unknown_kind",     s["rows_unknown_kind"]),
        ("rows_missing_pair",     s["rows_missing_pair"]),
        ("nan_inf",               s["nan_inf"]),
        ("miss_rate_oob",         s["miss_rate_oob"]),
        ("delta_mismatch",        s["delta_mismatch"]),
        ("sign_mismatch",         s["sign_mismatch"]),
        ("signed_delta_mismatch", s["signed_delta_mismatch"]),
        ("status_bad_label",      s["status_bad_label"]),
        ("status_inconsistent",   s["status_inconsistent"]),
        ("apps_count",            s["apps_count"]),
        ("apps_with_pp_signal",   ";".join(s["apps_with_pp_signal"])),
        ("apps_flat",             ";".join(s["apps_flat"])),
    ]
    for status, n in s["status_counts"].items():
        rows.append((f"status_{status}", n))
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["metric", "value"])
        for k, v in rows:
            w.writerow([k, v])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lit-faith-json", required=True, type=Path)
    ap.add_argument("--json-out",       required=True, type=Path)
    ap.add_argument("--md-out",         required=True, type=Path)
    ap.add_argument("--csv-out",        required=True, type=Path)
    args = ap.parse_args()

    audit = build_audit(args.lit_faith_json)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
    args.md_out.write_text(_to_markdown(audit), encoding="utf-8")
    _to_csv(audit, args.csv_out)

    s = audit["summary"]
    print(f"[lit-faith-stat] {s['total_rows']} rows; "
          f"delta_mismatch={s['delta_mismatch']} "
          f"sign_mismatch={s['sign_mismatch']} "
          f"signed_delta_mismatch={s['signed_delta_mismatch']} "
          f"miss_oob={s['miss_rate_oob']} "
          f"nan_inf={s['nan_inf']} "
          f"status_bad={s['status_bad_label']} "
          f"status_inconsistent={s['status_inconsistent']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
