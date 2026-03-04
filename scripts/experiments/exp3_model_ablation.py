#!/usr/bin/env python3
"""
Experiment 3 — Comprehensive Model Ablation for LOGO CV Accuracy.

Grid:
  Models:    DT(d=2), DT(d=3), DT(d=5), Hybrid(d=3), RF(100), XGBoost(100)
  Classes:   17-individual, 8-family
  Features:  12D (legacy), 14D (current +dv,+apl)
  Criteria:  F-Reorder, F-Execution, E2E, Amortize
  Metrics:   strict top-1, ≤5% regret-bounded, family-level accuracy

Output:
  JSON results table + LaTeX-ready summary.

Usage:
  python scripts/experiments/exp3_model_ablation.py
  python scripts/experiments/exp3_model_ablation.py --json results.json
  python scripts/experiments/exp3_model_ablation.py --models dt rf xgboost
  python scripts/experiments/exp3_model_ablation.py --quick   # Only F-Execution
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ensure project root is on PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.core.utils import RESULTS_DIR, Logger
from scripts.lib.ml.model_tree import (
    Criterion,
    compute_oracle,
    criterion_value,
    extract_dt_features,
    cross_validate_logo_model_tree,
    cross_validate_logo_random_forest,
)

log = Logger()

# ===================================================================
# Constants
# ===================================================================

ALL_CRITERIA = [
    Criterion.FASTEST_REORDER,
    Criterion.FASTEST_EXECUTION,
    Criterion.BEST_ENDTOEND,
    Criterion.BEST_AMORTIZATION,
]

CRITERION_LABELS = {
    Criterion.FASTEST_REORDER: 'F-Reorder',
    Criterion.FASTEST_EXECUTION: 'F-Execution',
    Criterion.BEST_ENDTOEND: 'E2E',
    Criterion.BEST_AMORTIZATION: 'Amortize',
}

BENCHMARKS = ['bc', 'bfs', 'cc', 'cc_sv', 'pr', 'pr_spmv', 'sssp', 'tc']


# ===================================================================
# Ablation Configuration
# ===================================================================

@dataclass
class AblationConfig:
    """One point in the ablation grid."""
    model: str              # 'dt', 'hybrid', 'rf', 'xgboost'
    max_depth: int = 3
    n_estimators: int = 100
    learning_rate: float = 0.1
    min_child_weight: int = 3
    min_samples_leaf: int = 3
    use_families: bool = False
    feature_dim: int = 14   # 12 or 14

    @property
    def label(self) -> str:
        parts = [self.model.upper()]
        if self.model in ('dt', 'hybrid'):
            parts.append(f'd{self.max_depth}')
        elif self.model == 'rf':
            parts.append(f'{self.n_estimators}t')
        elif self.model == 'xgboost':
            parts.append(f'{self.n_estimators}t')
        parts.append(f'{self.feature_dim}D')
        if self.use_families:
            parts.append('fam8')
        else:
            parts.append('ind17')
        return '_'.join(parts)


def build_ablation_grid(
    models: Optional[List[str]] = None,
    feature_dims: Optional[List[int]] = None,
    family_modes: Optional[List[bool]] = None,
) -> List[AblationConfig]:
    """Build the full grid of ablation configurations."""
    if models is None:
        models = ['dt', 'hybrid', 'rf', 'xgboost']
    if feature_dims is None:
        feature_dims = [12, 14]
    if family_modes is None:
        family_modes = [False, True]

    configs = []

    for feat_dim in feature_dims:
        for use_fam in family_modes:
            for model in models:
                if model == 'dt':
                    # Ablate tree depth
                    for depth in [2, 3, 5]:
                        configs.append(AblationConfig(
                            model='dt',
                            max_depth=depth,
                            use_families=use_fam,
                            feature_dim=feat_dim,
                        ))
                elif model == 'hybrid':
                    configs.append(AblationConfig(
                        model='hybrid',
                        max_depth=3,
                        use_families=use_fam,
                        feature_dim=feat_dim,
                    ))
                elif model == 'rf':
                    configs.append(AblationConfig(
                        model='rf',
                        max_depth=3,
                        n_estimators=100,
                        use_families=use_fam,
                        feature_dim=feat_dim,
                    ))
                elif model == 'xgboost':
                    configs.append(AblationConfig(
                        model='xgboost',
                        max_depth=3,
                        n_estimators=100,
                        learning_rate=0.1,
                        min_child_weight=3,
                        use_families=use_fam,
                        feature_dim=feat_dim,
                    ))
    return configs


# ===================================================================
# Data Loading
# ===================================================================

def load_data() -> Tuple[list, dict]:
    """Load benchmark records and graph properties.

    Returns:
        (raw_records, graph_props) — raw_records is list of dicts,
        graph_props is dict of graph_name -> properties dict.
    """
    from scripts.lib.ml.eval_weights import (
        load_benchmark_entries,
        filter_to_benchmark_results,
    )
    from scripts.lib.ml.features import load_graph_properties_cache

    raw = load_benchmark_entries(str(RESULTS_DIR))
    if not raw:
        log.error(f"No benchmark data found in {RESULTS_DIR}")
        sys.exit(1)

    bench_results = filter_to_benchmark_results(raw)
    graph_props = load_graph_properties_cache(str(RESULTS_DIR))

    # Convert to raw dicts
    raw_records = []
    for r in bench_results:
        if r.success and r.time_seconds > 0:
            raw_records.append({
                'graph': r.graph,
                'benchmark': r.benchmark,
                'algorithm': r.algorithm,
                'time_seconds': r.time_seconds,
                'reorder_time': r.reorder_time,
                'success': True,
            })

    graphs = sorted(set(r['graph'] for r in raw_records if r['graph'] != 'tiny'))
    algos = sorted(set(r['algorithm'] for r in raw_records))

    log.info(f"Loaded {len(raw_records)} records, {len(graphs)} graphs, {len(algos)} algorithms, {len(graph_props)} graph props")

    return raw_records, graph_props


# ===================================================================
# Feature Dimension Patching
# ===================================================================

_original_extract = extract_dt_features


def _extract_12d(props: dict) -> List[float]:
    """Extract legacy 12D features (drop degree_variance, avg_path_length)."""
    full = _original_extract(props)
    return full[:12]   # indices 0-11


def _patch_feature_extractor(dim: int):
    """Monkey-patch extract_dt_features to return `dim` features.

    This affects all downstream callers (train_*, cross_validate_*) which
    import extract_dt_features from model_tree.
    """
    import scripts.lib.ml.model_tree as mt_mod

    if dim == 12:
        mt_mod.extract_dt_features = _extract_12d
    elif dim == 14:
        mt_mod.extract_dt_features = _original_extract
    else:
        raise ValueError(f"Unsupported feature_dim={dim}")


def _restore_feature_extractor():
    """Restore the original 14D extractor."""
    import scripts.lib.ml.model_tree as mt_mod
    mt_mod.extract_dt_features = _original_extract


# ===================================================================
# LOGO CV Runner
# ===================================================================

def run_logo_cv(
    config: AblationConfig,
    raw_records: list,
    graph_props: dict,
    criterion: Criterion,
) -> Dict:
    """Run LOGO CV for one (config, criterion) cell.

    Returns dict with: accuracy, within_5pct, family_acc, avg_regret, ...
    """
    _patch_feature_extractor(config.feature_dim)

    try:
        if config.model in ('dt', 'hybrid'):
            model_type = 'decision_tree' if config.model == 'dt' else 'hybrid'
            result = cross_validate_logo_model_tree(
                raw_records, graph_props,
                model_type=model_type,
                criterion=criterion,
                benchmarks=BENCHMARKS,
                use_families=config.use_families,
            )
        elif config.model == 'rf':
            result = cross_validate_logo_random_forest(
                raw_records, graph_props,
                criterion=criterion,
                benchmarks=BENCHMARKS,
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_samples_leaf=config.min_samples_leaf,
                use_families=config.use_families,
            )
        elif config.model == 'xgboost':
            from scripts.lib.ml.model_tree import cross_validate_logo_xgboost
            result = cross_validate_logo_xgboost(
                raw_records, graph_props,
                criterion=criterion,
                benchmarks=BENCHMARKS,
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                min_child_weight=config.min_child_weight,
                learning_rate=config.learning_rate,
                use_families=config.use_families,
            )
        else:
            raise ValueError(f"Unknown model: {config.model}")
    finally:
        _restore_feature_extractor()

    return result


# ===================================================================
# Result Aggregation
# ===================================================================

@dataclass
class AblationResult:
    """One cell in the ablation grid."""
    config_label: str
    model: str
    max_depth: int
    n_estimators: int
    use_families: bool
    feature_dim: int
    criterion: str
    accuracy: float        # strict top-1
    within_5pct: float     # regret ≤ 5%
    family_acc: float      # family-level
    avg_regret: float
    median_regret: float
    total: int
    correct: int
    elapsed_s: float

    def to_dict(self) -> dict:
        return asdict(self)


def format_pct(val: float) -> str:
    return f"{val * 100:.1f}%"


# ===================================================================
# Output: Summary Tables
# ===================================================================

def print_summary_table(results: List[AblationResult]):
    """Print compact summary table sorted by within_5pct, then accuracy."""
    if not results:
        print("No results to display.")
        return

    # Group by criterion
    by_crit = defaultdict(list)
    for r in results:
        by_crit[r.criterion].append(r)

    for crit_name in ['fastest-reorder', 'fastest-execution',
                       'best-endtoend', 'best-amortization']:
        rows = by_crit.get(crit_name, [])
        if not rows:
            continue
        rows.sort(key=lambda r: (-r.within_5pct, -r.accuracy))

        label = {
            'fastest-reorder': 'F-Reorder',
            'fastest-execution': 'F-Execution',
            'best-endtoend': 'E2E',
            'best-amortization': 'Amortize',
        }.get(crit_name, crit_name)

        print(f"\n{'=' * 100}")
        print(f"  LOGO CV — {label}")
        print(f"{'=' * 100}")
        hdr = f"{'Config':<32} {'Top-1':>7} {'≤5%':>7} {'Fam':>7} {'Regret':>9} {'Med':>7} {'N':>5}"
        print(hdr)
        print("-" * 100)
        for r in rows:
            line = (
                f"{r.config_label:<32} "
                f"{format_pct(r.accuracy):>7} "
                f"{format_pct(r.within_5pct):>7} "
                f"{format_pct(r.family_acc):>7} "
                f"{r.avg_regret:>8.1f}% "
                f"{r.median_regret:>6.1f}% "
                f"{r.total:>5}"
            )
            print(line)
        print()


def print_latex_table(results: List[AblationResult]):
    """Print LaTeX-formatted table for paper inclusion."""
    if not results:
        return

    print("\n% LaTeX Ablation Table")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{LOGO CV Ablation Results}")
    print("\\label{tab:ablation}")
    print("\\begin{tabular}{lrrrr}")
    print("\\toprule")
    print("Config & Top-1 & $\\leq$5\\% & Family & Avg Regret \\\\")
    print("\\midrule")

    # Group by criterion, print best per criterion
    by_crit = defaultdict(list)
    for r in results:
        by_crit[r.criterion].append(r)

    for crit_name in ['fastest-execution', 'best-endtoend']:
        rows = by_crit.get(crit_name, [])
        rows.sort(key=lambda r: (-r.within_5pct, -r.accuracy))
        label = 'F-Exec' if crit_name == 'fastest-execution' else 'E2E'
        for r in rows[:5]:
            esc_label = r.config_label.replace('_', '\\_')
            print(f"{esc_label} & {r.accuracy*100:.1f} & "
                  f"{r.within_5pct*100:.1f} & {r.family_acc*100:.1f} & "
                  f"{r.avg_regret:.1f} \\\\")
        print("\\midrule")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")


def print_best_configs(results: List[AblationResult]):
    """Print the single best config per criterion and metric."""
    if not results:
        return

    print("\n" + "=" * 80)
    print("  BEST CONFIGS PER CRITERION × METRIC")
    print("=" * 80)

    by_crit = defaultdict(list)
    for r in results:
        by_crit[r.criterion].append(r)

    for crit_name, crits in sorted(by_crit.items()):
        label = {
            'fastest-reorder': 'F-Reorder',
            'fastest-execution': 'F-Execution',
            'best-endtoend': 'E2E',
            'best-amortization': 'Amortize',
        }.get(crit_name, crit_name)

        best_top1 = max(crits, key=lambda r: r.accuracy)
        best_5pct = max(crits, key=lambda r: r.within_5pct)
        best_fam  = max(crits, key=lambda r: r.family_acc)

        print(f"\n  {label}:")
        print(f"    Best Top-1:  {best_top1.config_label:<32} {format_pct(best_top1.accuracy)}")
        print(f"    Best ≤5%:    {best_5pct.config_label:<32} {format_pct(best_5pct.within_5pct)}")
        print(f"    Best Family: {best_fam.config_label:<32}  {format_pct(best_fam.family_acc)}")
    print()


# ===================================================================
# Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive model ablation for LOGO CV accuracy")
    parser.add_argument('--json', type=str, default=None,
                        help='Save results to JSON file')
    parser.add_argument('--models', nargs='+',
                        choices=['dt', 'hybrid', 'rf', 'xgboost'],
                        default=None,
                        help='Subset of models to test')
    parser.add_argument('--features', nargs='+', type=int,
                        default=None,
                        help='Feature dimensions to test (12 and/or 14)')
    parser.add_argument('--quick', action='store_true',
                        help='Only F-Execution criterion (4× faster)')
    parser.add_argument('--no-family', action='store_true',
                        help='Skip family-class ablation (2× faster)')
    parser.add_argument('--latex', action='store_true',
                        help='Print LaTeX table')
    args = parser.parse_args()

    # Build grid
    family_modes = [False] if args.no_family else [False, True]
    configs = build_ablation_grid(
        models=args.models,
        feature_dims=args.features,
        family_modes=family_modes,
    )

    criteria = [Criterion.FASTEST_EXECUTION] if args.quick else ALL_CRITERIA

    total_runs = len(configs) * len(criteria)
    log.info(f"Ablation grid: {len(configs)} configs × {len(criteria)} criteria = {total_runs} LOGO CV runs")
    for c in configs:
        log.info(f"  {c.label}")

    # Load data
    raw_records, graph_props = load_data()

    # Run ablation
    all_results: List[AblationResult] = []
    run_idx = 0
    t_start = time.time()

    for criterion in criteria:
        crit_label = CRITERION_LABELS.get(criterion, criterion.value)

        for config in configs:
            run_idx += 1
            log.info(f"[{run_idx}/{total_runs}] {config.label} × {crit_label} ...")

            t0 = time.time()
            result = run_logo_cv(config, raw_records, graph_props, criterion)
            elapsed = time.time() - t0

            ar = AblationResult(
                config_label=config.label,
                model=config.model,
                max_depth=config.max_depth,
                n_estimators=config.n_estimators,
                use_families=config.use_families,
                feature_dim=config.feature_dim,
                criterion=criterion.value,
                accuracy=result.get('accuracy', 0),
                within_5pct=result.get('within_5pct', 0),
                family_acc=result.get('family_acc', 0),
                avg_regret=result.get('avg_regret', 0),
                median_regret=result.get('median_regret', 0),
                total=result.get('total', 0),
                correct=result.get('correct', 0),
                elapsed_s=round(elapsed, 1),
            )
            all_results.append(ar)

            log.info(f"  → Top-1={ar.accuracy*100:.1f}%  ≤5%={ar.within_5pct*100:.1f}%  Fam={ar.family_acc*100:.1f}%  Regret={ar.avg_regret:.1f}%  ({elapsed:.1f}s)")

    total_time = time.time() - t_start
    log.info(f"Ablation complete: {total_runs} runs in {total_time:.0f}s ({total_time/60:.1f} min)")

    # Print tables
    print_summary_table(all_results)
    print_best_configs(all_results)

    if args.latex:
        print_latex_table(all_results)

    # Save JSON
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_data = {
            'timestamp': datetime.now().isoformat(),
            'total_runs': total_runs,
            'total_time_s': round(total_time, 1),
            'configs': [asdict(c) for c in configs],
            'criteria': [c.value for c in criteria],
            'results': [r.to_dict() for r in all_results],
        }
        with open(out_path, 'w') as f:
            json.dump(out_data, f, indent=2)
        log.info(f"Saved JSON results to {out_path}")

    # Auto-save to results dir
    auto_path = Path(RESULTS_DIR) / 'vldb_experiments' / 'exp3_logo_cv' / 'ablation_results.json'
    auto_path.parent.mkdir(parents=True, exist_ok=True)
    auto_data = {
        'timestamp': datetime.now().isoformat(),
        'total_runs': total_runs,
        'total_time_s': round(total_time, 1),
        'results': [r.to_dict() for r in all_results],
    }
    with open(auto_path, 'w') as f:
        json.dump(auto_data, f, indent=2)
    log.info(f"Auto-saved to {auto_path}")


if __name__ == '__main__':
    main()
