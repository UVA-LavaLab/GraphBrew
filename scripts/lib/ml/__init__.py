"""Machine learning: feature schema, offline training, and emulation."""

from .feature_schema import (
    TIER0_FEATURE_COUNT,
    TIER0_FEATURE_INDEX,
    TIER0_FEATURE_NAMES,
    extract_tier0_features,
    informativeness_ratio,
    passes_informativeness_gate,
    residual_informativeness_ratio,
    feature_passes_acceptance_gate,
)
from .portfolio import (
    DEPLOYABLE_ARM_SPECS,
    apply_portfolio_guard,
    normalize_deployable_arm,
)
from .working_set import (
    KERNEL_CLASS,
    modeled_property_bytes,
    property_wsr_llc,
)
from .source_policy import (
    ADAPTIVE_SOURCE_COUNT,
    ADAPTIVE_SOURCE_MIN_REACHABILITY,
    ADAPTIVE_SOURCE_POLICY_ID,
    ADAPTIVE_SOURCE_SEED,
    ADAPTIVE_PORTFOLIO_VERIFICATION_GATE_ID,
    adaptive_source_record_eligible,
    aggregate_source_trial_times,
    require_portfolio_gate_coverage,
)

__all__ = [
    "TIER0_FEATURE_COUNT",
    "TIER0_FEATURE_INDEX",
    "TIER0_FEATURE_NAMES",
    "extract_tier0_features",
    "informativeness_ratio",
    "passes_informativeness_gate",
    "residual_informativeness_ratio",
    "feature_passes_acceptance_gate",
    "DEPLOYABLE_ARM_SPECS",
    "apply_portfolio_guard",
    "normalize_deployable_arm",
    "KERNEL_CLASS",
    "modeled_property_bytes",
    "property_wsr_llc",
    "ADAPTIVE_SOURCE_COUNT",
    "ADAPTIVE_SOURCE_MIN_REACHABILITY",
    "ADAPTIVE_SOURCE_POLICY_ID",
    "ADAPTIVE_SOURCE_SEED",
    "ADAPTIVE_PORTFOLIO_VERIFICATION_GATE_ID",
    "adaptive_source_record_eligible",
    "aggregate_source_trial_times",
    "require_portfolio_gate_coverage",
]
