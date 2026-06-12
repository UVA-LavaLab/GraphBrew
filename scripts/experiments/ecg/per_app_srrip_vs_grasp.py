"""Gate 73 — per-app SRRIP-vs-GRASP slope ordering.

Companion to gate 68 (per-app LRU-vs-GRASP). Gate 72 demonstrated
that the GLOBAL SRRIP-vs-GRASP slope ordering holds across all three
tools (cache-sim, gem5 anchor, sniper anchor). This gate breaks the
SRRIP-vs-GRASP comparison out per app on the cache-sim sweep, to
expose any per-kernel anomalies and ensure they are pinned rather
than silently absorbed by the global median.

Source: gate 68's per_app_capacity_slope.json (which already exposes
per-(app, policy) median slopes). This gate reads that artifact and
applies the SRRIP-specific ordering check.

PASS iff:
  (1) every per-app (SRRIP, GRASP) pair is present,
  (2) for every app NOT in PINNED_DEVIATING_APPS, SRRIP median is
      no more than ALLOW_SRRIP_SHALLOWER_BY_PP pp/octave shallower
      than GRASP,
  (3) no NEW deviating app appears beyond the pin set.

Pinned deviation:
  bfs : frontier-driven streaming access pattern produces near-flat
        miss curves (gate 65 flags bfs as the most-saturated kernel).
        Per-cell slopes are small and both LRU (gate 68 pin) and
        SRRIP (gate 73 pin) end up shallower than GRASP on this
        kernel. A real corpus property, not a measurement artefact.

Output schema:
  meta.apps                       : list of apps observed
  meta.allow_srrip_shallower_pp   : the gap-floor (1.0 pp/oct)
  meta.per_app[A]                 : grasp_median, srrip_median,
                                    srrip_minus_grasp_pp_oct
  meta.deviating_apps             : apps where SRRIP > GRASP
                                    + ALLOW_SRRIP_SHALLOWER_BY_PP
  meta.pinned_deviating_apps      : ["bfs"]
  meta.new_deviating_apps         : deviating minus pinned
  meta.verdict                    : PASS / FAIL
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PER_APP_JSON = REPO_ROOT / "wiki" / "data" / "per_app_capacity_slope.json"
DEFAULT_JSON_OUT = REPO_ROOT / "wiki" / "data" / "per_app_srrip_vs_grasp.json"
DEFAULT_MD_OUT   = REPO_ROOT / "wiki" / "data" / "per_app_srrip_vs_grasp.md"

ALLOW_SRRIP_SHALLOWER_BY_PP = 1.0
PINNED_DEVIATING_APPS: tuple[str, ...] = ()  # 2026-06-12: bfs well-behaved at array-relative 0.15 (single-thread)


def compute(per_app_path: Path) -> dict:
    doc = json.loads(per_app_path.read_text())
    per_app_src = doc["meta"]["per_app"]

    per_app: dict[str, dict] = {}
    apps: list[str] = []
    deviating: list[str] = []
    missing: list[str] = []

    for app, block in sorted(per_app_src.items()):
        apps.append(app)
        g = block.get("GRASP", {}).get("median_pp")
        s = block.get("SRRIP", {}).get("median_pp")
        if g is None or s is None:
            missing.append(app)
            per_app[app] = {
                "grasp_median_pp_oct": g,
                "srrip_median_pp_oct": s,
                "srrip_minus_grasp_pp_oct": None,
                "deviates": None,
            }
            continue
        delta = s - g
        deviates = delta > ALLOW_SRRIP_SHALLOWER_BY_PP
        if deviates:
            deviating.append(app)
        per_app[app] = {
            "grasp_median_pp_oct":       round(g, 4),
            "srrip_median_pp_oct":       round(s, 4),
            "srrip_minus_grasp_pp_oct":  round(delta, 4),
            "deviates":                  deviates,
        }

    new_deviating = [a for a in deviating if a not in PINNED_DEVIATING_APPS]

    verdict_checks = {
        "no_missing_apps":          len(missing) == 0,
        "no_new_deviating_apps":    len(new_deviating) == 0,
        "every_app_has_both_grasp_and_srrip":
            all(per_app[a]["srrip_minus_grasp_pp_oct"] is not None
                for a in apps),
    }
    verdict = "PASS" if all(verdict_checks.values()) else "FAIL"

    return {
        "meta": {
            "source":                       str(per_app_path.name),
            "apps":                         apps,
            "allow_srrip_shallower_by_pp":  ALLOW_SRRIP_SHALLOWER_BY_PP,
            "pinned_deviating_apps":        list(PINNED_DEVIATING_APPS),
            "deviating_apps":               deviating,
            "new_deviating_apps":           new_deviating,
            "missing_apps":                 missing,
            "per_app":                      per_app,
            "verdict_checks":               verdict_checks,
            "verdict":                      verdict,
        },
    }


def render_md(payload: dict) -> str:
    m = payload["meta"]
    lines = [
        "# Per-app SRRIP-vs-GRASP slope ordering",
        "",
        f"**Verdict:** {m['verdict']}  ",
        f"**Source:** `{m['source']}`  ",
        f"**Apps:** {len(m['apps'])}  ",
        f"**Pinned deviating:** {', '.join(m['pinned_deviating_apps']) or 'none'}  ",
        f"**Allowed SRRIP-shallower-than-GRASP slack:** "
        f"{m['allow_srrip_shallower_by_pp']} pp/octave",
        "",
        "## Per-app medians (pp/octave)",
        "",
        "| app | GRASP median | SRRIP median | SRRIP-GRASP | deviates |",
        "|---|---:|---:|---:|:---:|",
    ]
    for app in m["apps"]:
        e = m["per_app"][app]
        g = e["grasp_median_pp_oct"]
        s = e["srrip_median_pp_oct"]
        d = e["srrip_minus_grasp_pp_oct"]
        dv = e["deviates"]
        gs = f"{g:+.4f}" if g is not None else "—"
        ss = f"{s:+.4f}" if s is not None else "—"
        ds = f"{d:+.4f}" if d is not None else "—"
        pin = " (pinned)" if app in m["pinned_deviating_apps"] else ""
        dv_str = ("✅" if not dv else ("📌" if app in m["pinned_deviating_apps"] else "❌"))
        lines.append(
            f"| {app}{pin} | {gs} | {ss} | {ds} | {dv_str} |"
        )

    lines += [
        "",
        "## Verdict checks",
        "",
        "| check | result |",
        "|---|---|",
    ]
    for k, v in m["verdict_checks"].items():
        lines.append(f"| {k} | {'✅' if v else '❌'} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "Gate 72 verified the SRRIP-vs-GRASP slope ordering holds at the "
        "GLOBAL median across all three tools. This gate ensures the "
        "ordering also holds per app on the cache-sim sweep — modulo "
        "documented kernel deviations. bfs is pinned because its "
        "frontier-driven access pattern produces near-flat miss curves "
        "(gate 65 flagged it as the most-saturated kernel), so both "
        "LRU (gate 68 pin) and SRRIP (this gate's pin) appear shallower "
        "than GRASP on bfs. This is a real corpus property of the "
        "kernel, not a measurement artefact.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-app-json", type=Path, default=DEFAULT_PER_APP_JSON)
    ap.add_argument("--json-out",     type=Path, default=DEFAULT_JSON_OUT)
    ap.add_argument("--md-out",       type=Path, default=DEFAULT_MD_OUT)
    args = ap.parse_args()

    payload = compute(args.per_app_json)
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    args.md_out.write_text(render_md(payload))

    m = payload["meta"]
    print(
        f"per-app-srrip-vs-grasp: apps={len(m['apps'])} "
        f"deviating={len(m['deviating_apps'])} "
        f"new_deviating={len(m['new_deviating_apps'])} "
        f"verdict={m['verdict']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
