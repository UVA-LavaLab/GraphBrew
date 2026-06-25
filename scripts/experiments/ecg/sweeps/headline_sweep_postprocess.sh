#!/usr/bin/env bash
# headline_sweep_postprocess.sh — coordinated post-sweep workflow.
#
# Runs after the cache_sim/gem5/Sniper headline-1MB sweeps complete.
# Re-aggregates all artifacts, bumps the gate-282 ratchet, surfaces
# any downstream EXPECTED_* drift, and reports the state.
#
# Does NOT auto-commit or auto-bump baselines — those are deliberate
# review steps. Prints exactly what changed and what needs human attention.
#
# Usage:
#   bash scripts/experiments/ecg/sweeps/headline_sweep_postprocess.sh
#
# Idempotent: safe to re-run after any sweep extension.

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

log() { echo "[postprocess] $*" >&2; }

log "=== Phase 1: Aggregate sweep outputs ==="
make lit-faith 2>&1 | tail -2
make lit-faith-ecg 2>&1 | tail -2
make gem5-anchor-headline-1mb 2>&1 | tail -2 || true
make sniper-anchor-headline-1mb 2>&1 | tail -2 || true

log "=== Phase 2: Refresh registries ==="
make lit-catalog 2>&1 | tail -1
make lit-wiki-registry 2>&1 | tail -1
make lit-filename-grammar 2>&1 | tail -1
make lit-handoff-xref 2>&1 | tail -1
make lit-reproduce-smoke 2>&1 | tail -1 || true

log "=== Phase 3: Gate 282 coverage check ==="
python3 -m scripts.experiments.ecg.analysis.coverage --quiet \
    --json-out wiki/data/headline_coverage.json \
    --md-out   wiki/data/headline_coverage.md \
    --csv-out  wiki/data/headline_coverage.csv
python3 -m scripts.experiments.ecg.analysis.coverage

log "=== Phase 4: Gate 283 parity check ==="
python3 -m scripts.experiments.ecg.analysis.parity --quiet \
    --json-out wiki/data/headline_parity.json \
    --md-out   wiki/data/headline_parity.md \
    --csv-out  wiki/data/headline_parity.csv
python3 -c "
import json
d = json.load(open('wiki/data/headline_parity.json'))
s = d['summary']
print(f\"  cells_total={s['cells_total']}\")
print(f\"  cells_with_overlap={s['cells_with_overlap']}\")
print(f\"  agree={s['cells_agree']}  disagree={s['cells_disagree']}\")
print(f\"  single={s['cells_single_sim']}  empty={s['cells_empty']}\")
"

log "=== Phase 5: Baseline drift report ==="
python3 -m scripts.experiments.ecg.baseline_drift_bumper

log "=== Phase 6: Failing tests on current dashboard ==="
make confidence-fast 2>&1 | tail -2 || true
python3 -c "
import json
d = json.load(open('wiki/data/confidence_dashboard.json'))
bad = [s for s in d['suites'] if s.get('failed',0)>0 or s.get('errors',0)>0]
if not bad:
    print(f\"  ✅ GREEN — {len(d['suites'])} suites, 0 failures\")
else:
    print(f\"  ⛔ RED — {len(bad)} failing suites:\")
    for s in bad[:25]:
        print(f\"    {s.get('short')} F={s.get('failed')}\")
"

log "=== Phase 7: Summary ==="
log "If GREEN: bump ratchet then commit:"
log "  make headline-coverage-bump"
log "  git add -A && git commit -F .git/COMMIT_EDITMSG  # craft message"
log "If RED: review failing tests, bump EXPECTED_* constants per the drift report above."
