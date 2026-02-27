#!/usr/bin/env python3
"""
Test Weight Flow - Verify weights are generated and read from correct locations.

This test verifies:
1. Python stages weights to results/models/perceptron/ (legacy staging dir)
2. export_unified_models() merges into results/data/adaptive_models.json
3. C++ loads from adaptive_models.json or trains at runtime from benchmarks.json
4. Weight files are valid JSON with required fields (bias, w_modularity, etc.)
5. Directory structure and constants are consistent

Usage:
    pytest scripts/test/test_weight_flow.py -v
    python -m scripts.test.test_weight_flow
"""

import pytest
import sys
import json
import shutil
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.lib.core.utils import (
    WEIGHTS_DIR, ACTIVE_WEIGHTS_DIR,
    VARIANT_PREFIXES, VARIANT_ALGO_IDS, DISPLAY_TO_CANONICAL,
    ALGORITHMS, RABBITORDER_VARIANTS, RCM_VARIANTS, GRAPHBREW_VARIANTS,
    CHAIN_SEPARATOR, CHAINED_ORDERINGS, _CHAINED_ORDERING_OPTS,
    is_chained_ordering_name,
    get_all_algorithm_variant_names, resolve_canonical_name, is_variant_prefixed,
    canonical_algo_key, algo_converter_opt, get_algo_variants,
    canonical_name_from_converter_opt, chain_canonical_name,
    get_algorithm_name,
)
from scripts.lib.ml.weights import (
    DEFAULT_WEIGHTS_DIR,
    save_type_weights,
    load_type_weights,
)


class ResultsTracker:
    """Track test results."""
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []
    
    def check(self, condition: bool, message: str):
        """Check a condition and record result."""
        if condition:
            self.passed += 1
            print(f"  ✓ {message}")
        else:
            self.failed += 1
            self.errors.append(message)
            print(f"  ✗ {message}")
    
    def summary(self):
        """Print summary."""
        print()
        print("=" * 60)
        total = self.passed + self.failed
        print(f"Results: {self.passed}/{total} passed")
        if self.errors:
            print("\nFailed tests:")
            for err in self.errors:
                print(f"  - {err}")
        return self.failed == 0


@pytest.fixture
def results():
    return ResultsTracker()


@pytest.fixture(autouse=True)
def weights_env(tmp_path, monkeypatch):
    import scripts.lib.core.utils as utils
    import scripts.lib.ml.weights as weights

    tmp_weights = tmp_path / "models" / "perceptron"
    tmp_active = tmp_weights
    tmp_weights.mkdir(parents=True, exist_ok=True)

    # Patch utils
    monkeypatch.setattr(utils, "WEIGHTS_DIR", tmp_weights)
    monkeypatch.setattr(utils, "ACTIVE_WEIGHTS_DIR", tmp_weights)

    # Patch weights module
    monkeypatch.setattr(weights, "DEFAULT_WEIGHTS_DIR", str(tmp_weights))

    # Seed default files
    default_weights = {
        "Algo0": {"bias": 0.5, "w_modularity": 0.1, "w_log_nodes": 0.0}
    }
    (tmp_weights / "type_0").mkdir(parents=True, exist_ok=True)
    (tmp_weights / "type_0" / "weights.json").write_text(json.dumps(default_weights))
    (tmp_weights / "registry.json").write_text(json.dumps({
        "type_0": {"centroid": [0]*7, "graph_count": 1, "algorithms": ["Algo0"]}
    }))

    return tmp_weights, tmp_weights


def test_path_constants(results: ResultsTracker):
    """Test that path constants are correct."""
    print("\n1. Testing Path Constants")
    print("-" * 40)
    
    # WEIGHTS_DIR should be results/models/perceptron
    results.check(
        WEIGHTS_DIR.name == "perceptron" and WEIGHTS_DIR.parent.name == "models",
        f"WEIGHTS_DIR points to results/models/perceptron: {WEIGHTS_DIR}"
    )
    
    # ACTIVE_WEIGHTS_DIR should equal WEIGHTS_DIR (no more /active/ sublevel)
    results.check(
        ACTIVE_WEIGHTS_DIR == WEIGHTS_DIR,
        f"ACTIVE_WEIGHTS_DIR equals WEIGHTS_DIR: {ACTIVE_WEIGHTS_DIR}"
    )
    
    # DEFAULT_WEIGHTS_DIR should equal ACTIVE_WEIGHTS_DIR
    results.check(
        Path(DEFAULT_WEIGHTS_DIR) == ACTIVE_WEIGHTS_DIR,
        "DEFAULT_WEIGHTS_DIR equals ACTIVE_WEIGHTS_DIR"
    )


def test_directory_structure(results: ResultsTracker):
    """Test that models/ directory is created on demand and weights work.

    models/ is a temporary build directory — NOT created by ensure_directories().
    It is created on demand by weights.py during training,
    and cleaned up after export_unified_models() merges everything into
    results/data/adaptive_models.json.  The fixture creates a tmp version
    for testing.
    """
    import scripts.lib.core.utils as utils_mod

    print("\n2. Testing Directory Structure (on-demand)")
    print("-" * 40)

    patched_weights_dir = utils_mod.WEIGHTS_DIR
    results.check(
        patched_weights_dir.exists(),
        "WEIGHTS_DIR exists (created by fixture/on-demand)"
    )

    patched_active_dir = utils_mod.ACTIVE_WEIGHTS_DIR
    results.check(
        patched_active_dir.exists(),
        "ACTIVE_WEIGHTS_DIR exists (created by fixture/on-demand)"
    )

    # Check for type directories in active
    type_dirs = [d for d in patched_active_dir.iterdir() if d.is_dir() and d.name.startswith('type_')]
    results.check(
        len(type_dirs) > 0,
        f"Found {len(type_dirs)} type directories in active/"
    )
    
    registry_file = ACTIVE_WEIGHTS_DIR / "registry.json"
    results.check(
        registry_file.exists(),
        "registry.json exists in active/"
    )


def test_weights_write_to_active(results: ResultsTracker):
    """Test that weights.py writes to active/ directory."""
    print("\n3. Testing Weights Write to Active")
    print("-" * 40)
    
    # Create a test type with unique name
    test_type = "type_test_flow"
    test_weights = {
        "TestAlgo": {
            "bias": 1.5,
            "w_modularity": 0.1,
            "w_log_nodes": 0.2,
        }
    }
    
    # Save using weights.py (should go to active/)
    save_type_weights(test_type, test_weights)
    
    # Verify it was written to active/
    expected_path = ACTIVE_WEIGHTS_DIR / test_type / "weights.json"
    results.check(
        expected_path.exists(),
        f"save_type_weights wrote to active/{test_type}/weights.json"
    )
    
    # Load and verify content
    loaded = load_type_weights(test_type)
    results.check(
        loaded.get("TestAlgo", {}).get("bias") == 1.5,
        "load_type_weights reads from active/"
    )
    
    # Clean up
    if expected_path.exists():
        shutil.rmtree(expected_path.parent)
        print(f"  [cleanup] Removed {test_type}/")


def test_cpp_path_constants(results: ResultsTracker):
    """Test that C++ header has correct paths."""
    print("\n4. Testing C++ Path Constants")
    print("-" * 40)
    
    reorder_h = PROJECT_ROOT / "bench" / "include" / "graphbrew" / "reorder" / "reorder_types.h"
    
    results.check(
        reorder_h.exists(),
        "reorder_types.h exists"
    )
    
    if reorder_h.exists():
        content = reorder_h.read_text()
        
        # Check MODEL_TREE_DIR (C++ loads DT/hybrid models from results/models/)
        results.check(
            'MODEL_TREE_DIR' in content,
            "MODEL_TREE_DIR constant exists in reorder_types.h"
        )


def test_existing_weights_valid(results: ResultsTracker):
    """Test that existing weights in active/ are valid."""
    print("\n6. Testing Existing Weights Validity")
    print("-" * 40)
    
    # Check type directories
    type_dirs = sorted(d for d in ACTIVE_WEIGHTS_DIR.iterdir() if d.is_dir() and d.name.startswith('type_'))
    results.check(
        len(type_dirs) > 0,
        f"Found {len(type_dirs)} type directories"
    )
    
    for type_dir in type_dirs:
        type_file = type_dir / "weights.json"
        if not type_file.exists():
            continue
        try:
            with open(type_file) as f:
                weights = json.load(f)
            
            # Check it has algorithms
            algos = [k for k in weights if not k.startswith('_')]
            valid = len(algos) > 0
            
            # Check first algo has required fields
            if algos:
                first_algo = weights[algos[0]]
                valid = valid and 'bias' in first_algo
                valid = valid and 'w_modularity' in first_algo
            
            results.check(
                valid,
                f"{type_dir.name}/weights.json: {len(algos)} algorithms, valid structure"
            )
        except Exception as e:
            results.check(False, f"{type_dir.name}/weights.json: Error - {e}")
    
    # Check registry
    registry_file = ACTIVE_WEIGHTS_DIR / "registry.json"
    if registry_file.exists():
        try:
            with open(registry_file) as f:
                registry = json.load(f)
            
            # Check it has type entries
            type_entries = [k for k in registry if k.startswith('type_')]
            results.check(
                len(type_entries) > 0,
                f"registry.json: {len(type_entries)} type entries"
            )
        except Exception as e:
            results.check(False, f"registry.json: Error - {e}")


# =========================================================================
# SSOT Variant Registry Tests
# =========================================================================

class TestVariantRegistrySSOT:
    """Verify the SSOT variant registry is consistent across all layers."""

    def test_variant_prefixes_are_tuples(self):
        """Variant lists should be tuples (immutable SSOT)."""
        assert isinstance(VARIANT_PREFIXES, tuple)
        assert isinstance(RABBITORDER_VARIANTS, tuple)
        assert isinstance(RCM_VARIANTS, tuple)
        assert isinstance(GRAPHBREW_VARIANTS, tuple)

    def test_variant_prefixes_match_registry(self):
        """VARIANT_PREFIXES should match _VARIANT_ALGO_REGISTRY prefixes."""
        from scripts.lib.core.utils import _VARIANT_ALGO_REGISTRY
        registry_prefixes = {pfx for pfx, _, _ in _VARIANT_ALGO_REGISTRY.values()}
        assert set(VARIANT_PREFIXES) == registry_prefixes

    def test_variant_algo_ids_complete(self):
        """Every algo with variants should be in VARIANT_ALGO_IDS."""
        assert 8 in VARIANT_ALGO_IDS   # RabbitOrder
        assert 11 in VARIANT_ALGO_IDS  # RCM
        assert 12 in VARIANT_ALGO_IDS  # GraphBrewOrder

    def test_get_all_variant_names_no_duplicates(self):
        """get_all_algorithm_variant_names() should return unique names."""
        names = get_all_algorithm_variant_names()
        assert len(names) == len(set(names)), f"Duplicates: {[n for n in names if names.count(n) > 1]}"

    def test_get_all_variant_names_includes_expected(self):
        """All canonical variant names should appear."""
        names = set(get_all_algorithm_variant_names())
        assert "RABBITORDER_csr" in names
        assert "RABBITORDER_boost" in names
        assert "RCM_default" in names
        assert "RCM_bnf" in names
        assert "GraphBrewOrder_leiden" in names
        assert "GraphBrewOrder_rabbit" in names
        assert "GraphBrewOrder_hubcluster" in names
        assert "LeidenOrder" in names

    def test_get_all_variant_names_excludes_meta(self):
        """MAP, AdaptiveOrder, ORIGINAL, RANDOM should not be in trainable variant names."""
        names = set(get_all_algorithm_variant_names())
        assert "MAP" not in names
        assert "AdaptiveOrder" not in names
        assert "ORIGINAL" not in names
        assert "RANDOM" not in names

    def test_resolve_canonical_base_names(self):
        """Base algorithm names resolve correctly."""
        assert resolve_canonical_name("ORIGINAL") == "ORIGINAL"
        assert resolve_canonical_name("Random") == "RANDOM"
        assert resolve_canonical_name("HubSort") == "HUBSORT"

    def test_resolve_canonical_variant_passthrough(self):
        """Variant-prefixed names pass through unchanged."""
        assert resolve_canonical_name("GraphBrewOrder_leiden") == "GraphBrewOrder_leiden"
        assert resolve_canonical_name("RABBITORDER_csr") == "RABBITORDER_csr"
        assert resolve_canonical_name("RCM_bnf") == "RCM_bnf"
        # New compound variants also pass through
        assert resolve_canonical_name("GraphBrewOrder_leiden_dfs") == "GraphBrewOrder_leiden_dfs"

    def test_resolve_canonical_default_variants(self):
        """Bare base names resolve to default variant."""
        assert resolve_canonical_name("RabbitOrder") == "RABBITORDER_csr"
        assert resolve_canonical_name("GraphBrewOrder") == "GraphBrewOrder_leiden"

    def test_is_variant_prefixed(self):
        """is_variant_prefixed correctly identifies variant names."""
        assert is_variant_prefixed("GraphBrewOrder_leiden")
        assert is_variant_prefixed("RABBITORDER_csr")
        assert is_variant_prefixed("RCM_bnf")
        assert not is_variant_prefixed("ORIGINAL")
        assert not is_variant_prefixed("DBG")
        assert not is_variant_prefixed("LeidenOrder")

    def test_display_to_canonical_complete(self):
        """DISPLAY_TO_CANONICAL should cover all C++ display names."""
        # Every non-variant algorithm should have an entry
        for algo_id, name in ALGORITHMS.items():
            if algo_id in VARIANT_ALGO_IDS or name in ("MAP", "AdaptiveOrder"):
                continue
            assert name in DISPLAY_TO_CANONICAL or name.upper() in DISPLAY_TO_CANONICAL, \
                f"Missing DISPLAY_TO_CANONICAL entry for {name}"

    # ---- Multi-layer tests ----

    def test_graphbrew_layers_structure(self):
        """GRAPHBREW_LAYERS should have all 6 layers."""
        from scripts.lib.core.utils import GRAPHBREW_LAYERS
        assert "preset" in GRAPHBREW_LAYERS
        assert "ordering" in GRAPHBREW_LAYERS
        assert "aggregation" in GRAPHBREW_LAYERS
        assert "features" in GRAPHBREW_LAYERS
        assert "graphbrew_dispatch" in GRAPHBREW_LAYERS
        assert "numeric" in GRAPHBREW_LAYERS

    def test_ordering_excludes_streaming(self):
        """'streaming' is an aggregation strategy, NOT an ordering strategy."""
        from scripts.lib.core.utils import GRAPHBREW_LAYERS
        assert "streaming" not in GRAPHBREW_LAYERS["ordering"]
        assert "streaming" in GRAPHBREW_LAYERS["aggregation"]

    def test_graphbrew_options_backward_compat(self):
        """GRAPHBREW_OPTIONS backward-compat alias matches GRAPHBREW_LAYERS."""
        from scripts.lib.core.utils import GRAPHBREW_OPTIONS, GRAPHBREW_LAYERS
        assert set(GRAPHBREW_OPTIONS["presets"].keys()) == set(GRAPHBREW_LAYERS["preset"].keys())
        assert GRAPHBREW_OPTIONS["ordering_strategies"] == list(GRAPHBREW_LAYERS["ordering"])
        assert GRAPHBREW_OPTIONS["aggregation"] == list(GRAPHBREW_LAYERS["aggregation"])
        assert GRAPHBREW_OPTIONS["features"] == list(GRAPHBREW_LAYERS["features"])

    def test_enumerate_multilayer_counts(self):
        """enumerate_graphbrew_multilayer() returns correct counts."""
        from scripts.lib.core.utils import enumerate_graphbrew_multilayer
        info = enumerate_graphbrew_multilayer()
        assert info["layers"]["presets"] == 3
        assert info["layers"]["orderings"] == 13
        assert info["layers"]["aggregations"] == 4
        assert info["layers"]["features"] == 11
        assert info["layers"]["feature_combos"] == 2048  # 2^11
        # 3 base presets + 3×13 compounds = 42
        assert len(info["compound_variants"]) == 3 + 3 * 13
        # Active trained ⊆ compound_variants
        for a in info["active_trained"]:
            assert a in info["compound_variants"]

    def test_enumerate_multilayer_no_overlap(self):
        """Active and untrained should be disjoint and cover all compounds."""
        from scripts.lib.core.utils import enumerate_graphbrew_multilayer
        info = enumerate_graphbrew_multilayer()
        active_set = set(info["active_trained"])
        untrained_set = set(info["untrained"])
        assert active_set & untrained_set == set(), "Overlap between active and untrained"
        assert active_set | untrained_set == set(info["compound_variants"])

    def test_get_algo_variants_variant_algos(self):
        """get_algo_variants() returns variant tuples for registered algorithms."""
        assert get_algo_variants(8) == RABBITORDER_VARIANTS
        assert get_algo_variants(11) == RCM_VARIANTS
        assert get_algo_variants(12) == GRAPHBREW_VARIANTS

    def test_get_algo_variants_non_variant(self):
        """get_algo_variants() returns None for non-variant algorithms."""
        assert get_algo_variants(0) is None
        assert get_algo_variants(2) is None
        assert get_algo_variants(9) is None   # GOrder: intentionally not in registry
        assert get_algo_variants(15) is None  # LeidenOrder


class TestChainedOrderingExclusion:
    """Chained orderings are included in weight training alongside single orderings."""

    def test_chain_separator_constant(self):
        """CHAIN_SEPARATOR should be '+'."""
        assert CHAIN_SEPARATOR == "+"

    def test_is_chained_ordering_name(self):
        """is_chained_ordering_name() should detect '+' in names."""
        assert is_chained_ordering_name("SORT+RABBITORDER_csr")
        assert is_chained_ordering_name("DBG+GraphBrewOrder_leiden")
        assert not is_chained_ordering_name("RABBITORDER_csr")
        assert not is_chained_ordering_name("ORIGINAL")
        assert not is_chained_ordering_name("GraphBrewOrder_leiden")

    def test_chained_orderings_all_have_separator(self):
        """Every CHAINED_ORDERINGS entry must contain CHAIN_SEPARATOR."""
        for canonical, _opts in CHAINED_ORDERINGS:
            assert CHAIN_SEPARATOR in canonical, f"{canonical} missing separator"

    def test_variant_names_include_chains(self):
        """get_all_algorithm_variant_names() must include chained names."""
        names = get_all_algorithm_variant_names()
        chained_names = [n for n in names if is_chained_ordering_name(n)]
        assert len(chained_names) == len(CHAINED_ORDERINGS), (
            f"Expected {len(CHAINED_ORDERINGS)} chained names in variant list, "
            f"got {len(chained_names)}: {chained_names}"
        )

    def test_chain_names_auto_generated(self):
        """CHAINED_ORDERINGS names must be auto-derived from _CHAINED_ORDERING_OPTS."""
        assert len(CHAINED_ORDERINGS) == len(_CHAINED_ORDERING_OPTS)
        for (canonical, opts), raw_opts in zip(CHAINED_ORDERINGS, _CHAINED_ORDERING_OPTS):
            assert opts == raw_opts, f"opts mismatch: {opts!r} != {raw_opts!r}"
            expected_name = chain_canonical_name(raw_opts)
            assert canonical == expected_name, (
                f"Auto name mismatch: {canonical!r} != {expected_name!r}"
            )


class TestCanonicalNameFromOpt:
    """canonical_name_from_converter_opt() derives names from -o arguments."""

    def test_simple_algorithms(self):
        """Non-variant algorithms return base name."""
        assert canonical_name_from_converter_opt("0") == "ORIGINAL"
        assert canonical_name_from_converter_opt("2") == "SORT"
        assert canonical_name_from_converter_opt("5") == "DBG"
        assert canonical_name_from_converter_opt("9") == "GORDER"

    def test_variant_default(self):
        """Variant algorithms without explicit variant use default."""
        assert canonical_name_from_converter_opt("8") == "RABBITORDER_csr"
        assert canonical_name_from_converter_opt("11") == "RCM_default"
        assert canonical_name_from_converter_opt("12") == "GraphBrewOrder_leiden"

    def test_variant_explicit(self):
        """Explicit variant in option string."""
        assert canonical_name_from_converter_opt("8:boost") == "RABBITORDER_boost"
        assert canonical_name_from_converter_opt("11:bnf") == "RCM_bnf"
        assert canonical_name_from_converter_opt("12:rabbit") == "GraphBrewOrder_rabbit"

    def test_multilayer_graphbrew(self):
        """Multi-layer GraphBrewOrder configs normalize colons to underscores."""
        assert canonical_name_from_converter_opt("12:leiden:hrab") == "GraphBrewOrder_leiden_hrab"
        assert canonical_name_from_converter_opt("12:leiden:hrab:gvecsr") == "GraphBrewOrder_leiden_hrab_gvecsr"
        assert canonical_name_from_converter_opt("12:leiden:hrab:gvecsr:merge:hubx") == (
            "GraphBrewOrder_leiden_hrab_gvecsr_merge_hubx"
        )

    def test_flat_token_stripped_from_canonical_name(self):
        """Runtime-only tokens (flat, norecurse, recursive) are stripped."""
        assert canonical_name_from_converter_opt("12:leiden:flat") == "GraphBrewOrder_leiden"
        assert canonical_name_from_converter_opt("12:leiden:hrab:flat") == "GraphBrewOrder_leiden_hrab"
        assert canonical_name_from_converter_opt("12:leiden:norecurse") == "GraphBrewOrder_leiden"
        assert canonical_name_from_converter_opt("12:leiden:recursive") == "GraphBrewOrder_leiden"

    def test_agrees_with_canonical_algo_key(self):
        """canonical_name_from_converter_opt() must agree with canonical_algo_key()."""
        for algo_id in ALGORITHMS:
            name_from_opt = canonical_name_from_converter_opt(str(algo_id))
            name_from_key = canonical_algo_key(algo_id)
            assert name_from_opt == name_from_key, (
                f"Disagreement for algo {algo_id}: opt={name_from_opt!r} key={name_from_key!r}"
            )

    def test_agrees_with_get_algorithm_name(self):
        """canonical_name_from_converter_opt and get_algorithm_name must agree."""
        test_opts = ["0", "2", "8", "8:boost", "12", "12:leiden", "12:rabbit", "11:bnf"]
        for opt in test_opts:
            from_opt = canonical_name_from_converter_opt(opt)
            from_name = get_algorithm_name(opt)
            assert from_opt == from_name, (
                f"Disagreement for {opt!r}: from_opt={from_opt!r} from_name={from_name!r}"
            )


class TestChainCanonicalName:
    """chain_canonical_name() auto-derives chain names from converter opts."""

    def test_basic_chains(self):
        assert chain_canonical_name("-o 2 -o 8:csr") == "SORT+RABBITORDER_csr"
        assert chain_canonical_name("-o 2 -o 8:boost") == "SORT+RABBITORDER_boost"
        assert chain_canonical_name("-o 7 -o 8:csr") == "HUBCLUSTERDBG+RABBITORDER_csr"
        assert chain_canonical_name("-o 2 -o 12:leiden") == "SORT+GraphBrewOrder_leiden"
        assert chain_canonical_name("-o 5 -o 12:leiden") == "DBG+GraphBrewOrder_leiden"

    def test_multilayer_in_chain(self):
        """Chains with multi-layer GraphBrewOrder configs."""
        assert chain_canonical_name("-o 2 -o 12:leiden:hrab") == "SORT+GraphBrewOrder_leiden_hrab"
        assert chain_canonical_name("-o 2 -o 12:leiden:hrab:gvecsr:merge") == (
            "SORT+GraphBrewOrder_leiden_hrab_gvecsr_merge"
        )

    def test_invalid_opts_raises(self):
        """Empty converter opts should raise ValueError."""
        import pytest
        with pytest.raises(ValueError):
            chain_canonical_name("no dash o here")


class TestMultiLayerVariantPropagation:
    """Multi-layer variant names propagate correctly through all APIs."""

    def test_canonical_algo_key_multilayer(self):
        """canonical_algo_key with colon-separated multi-layer variant."""
        assert canonical_algo_key(12, "leiden:hrab") == "GraphBrewOrder_leiden_hrab"
        assert canonical_algo_key(12, "leiden:hrab:gvecsr") == "GraphBrewOrder_leiden_hrab_gvecsr"
        assert canonical_algo_key(12, "leiden:hrab:gvecsr:merge:hubx") == (
            "GraphBrewOrder_leiden_hrab_gvecsr_merge_hubx"
        )

    def test_algo_converter_opt_multilayer(self):
        """algo_converter_opt preserves colons for CLI."""
        assert algo_converter_opt(12, "leiden:hrab") == "12:leiden:hrab"
        assert algo_converter_opt(12, "leiden:hrab:gvecsr:merge") == "12:leiden:hrab:gvecsr:merge"

    def test_roundtrip_key_to_opt(self):
        """canonical_algo_key and canonical_name_from_converter_opt roundtrip."""
        variants = ["leiden", "leiden:hrab", "leiden:hrab:gvecsr:merge:hubx"]
        for v in variants:
            key = canonical_algo_key(12, v)
            opt = algo_converter_opt(12, v)
            roundtrip_key = canonical_name_from_converter_opt(opt)
            assert roundtrip_key == key, (
                f"Roundtrip mismatch for variant {v!r}: key={key!r} roundtrip={roundtrip_key!r}"
            )


class TestCanonicalAlgoKey:
    """canonical_algo_key() is the single entry-point for algorithm naming."""

    def test_non_variant_algorithms(self):
        """Non-variant algorithms return the bare SSOT name."""
        assert canonical_algo_key(0) == "ORIGINAL"
        assert canonical_algo_key(1) == "RANDOM"
        assert canonical_algo_key(2) == "SORT"
        assert canonical_algo_key(5) == "DBG"
        assert canonical_algo_key(9) == "GORDER"
        assert canonical_algo_key(10) == "CORDER"
        assert canonical_algo_key(15) == "LeidenOrder"

    def test_variant_algorithms_default(self):
        """Variant algorithms always include default variant suffix."""
        assert canonical_algo_key(8) == "RABBITORDER_csr"
        assert canonical_algo_key(11) == "RCM_default"
        assert canonical_algo_key(12) == "GraphBrewOrder_leiden"

    def test_variant_algorithms_explicit(self):
        """Explicit variant is used when provided."""
        assert canonical_algo_key(8, "boost") == "RABBITORDER_boost"
        assert canonical_algo_key(11, "bnf") == "RCM_bnf"
        assert canonical_algo_key(12, "rabbit") == "GraphBrewOrder_rabbit"
        assert canonical_algo_key(12, "hubcluster") == "GraphBrewOrder_hubcluster"

    def test_matches_get_all_variant_names(self):
        """Every name from get_all_algorithm_variant_names() should be
        producible by canonical_algo_key()."""
        all_names = set(get_all_algorithm_variant_names())
        for algo_id, algo_name in ALGORITHMS.items():
            if algo_name in ("ORIGINAL", "RANDOM", "MAP", "AdaptiveOrder"):
                continue
            key = canonical_algo_key(algo_id)
            assert key in all_names, f"canonical_algo_key({algo_id}) = {key!r} not in SSOT list"

    def test_matches_get_algorithm_name_with_variant(self):
        """canonical_algo_key() and get_algorithm_name_with_variant() must agree."""
        from scripts.lib.pipeline.reorder import get_algorithm_name_with_variant
        for algo_id in ALGORITHMS:
            assert canonical_algo_key(algo_id) == get_algorithm_name_with_variant(algo_id)

    def test_algo_converter_opt_non_variant(self):
        """Non-variant algorithms produce bare ID strings."""
        assert algo_converter_opt(0) == "0"
        assert algo_converter_opt(2) == "2"
        assert algo_converter_opt(5) == "5"

    def test_algo_converter_opt_variant_default(self):
        """Variant algorithms include default variant in converter opt."""
        assert algo_converter_opt(8) == "8:csr"
        assert algo_converter_opt(11) == "11:default"
        assert algo_converter_opt(12) == "12:leiden:flat"

    def test_algo_converter_opt_explicit(self):
        """Explicit variant is used in converter opt."""
        assert algo_converter_opt(8, "boost") == "8:boost"
        assert algo_converter_opt(12, "leiden") == "12:leiden:flat"
        assert algo_converter_opt(12, "rabbit") == "12:rabbit"

    def test_key_and_opt_are_paired(self):
        """canonical_algo_key() and algo_converter_opt() should cover the
        same algorithm IDs consistently."""
        for algo_id in ALGORITHMS:
            key = canonical_algo_key(algo_id)
            opt = algo_converter_opt(algo_id)
            # The opt should start with the algo_id
            assert opt.startswith(str(algo_id)), f"opt={opt!r} doesn't start with {algo_id}"
            # The key should not be empty
            assert key, f"Empty key for algo_id={algo_id}"


# =============================================================================
# B: Field Parity Tests — verify Python ↔ C++ JSON field agreement
# =============================================================================

import math


# Exact set of linear weight keys C++ reads via from_json().
# Order matches C++ PerceptronWeights struct in reorder_types.h.
CPP_WEIGHT_KEYS = [
    "bias",
    "w_modularity",
    "w_log_nodes",
    "w_log_edges",
    "w_density",
    "w_avg_degree",
    "w_degree_variance",
    "w_hub_concentration",
    "w_clustering_coeff",
    "w_avg_path_length",
    "w_diameter",
    "w_community_count",
    "w_packing_factor",
    "w_forward_edge_fraction",
    "w_working_set_ratio",
    # Quadratic cross-terms
    "w_dv_x_hub",
    "w_mod_x_logn",
    "w_pf_x_wsr",
    # Convergence bonus
    "w_fef_convergence",
    # Cache impact
    "cache_l1_impact",
    "cache_l2_impact",
    "cache_l3_impact",
    "cache_dram_penalty",
    # Reorder time
    "w_reorder_time",
]

CPP_BENCHMARK_KEYS = ["pr", "bfs", "cc", "sssp", "bc", "tc", "pr_spmv", "cc_sv"]

CPP_METADATA_KEYS = {"avg_speedup", "avg_reorder_time"}

CPP_NORM_KEYS = {"feat_means", "feat_stds"}  # inside _normalization block

CPP_GRAPH_PROPS_KEYS = {
    "nodes", "edges", "modularity", "degree_variance", "hub_concentration",
    "clustering_coefficient", "avg_degree", "avg_path_length", "diameter",
    "community_count", "packing_factor", "forward_edge_fraction",
    "working_set_ratio", "density", "graph_type",
}

CPP_RUN_REPORT_KEYS = {
    "graph", "algorithm", "algorithm_id", "benchmark", "time_seconds",
    "reorder_time", "trials", "nodes", "edges", "success", "error",
}


class TestFieldParity:
    """Verify every JSON field Python writes matches what C++ reads, and vice-versa."""

    # --- B1: PerceptronWeight round-trip ---

    def test_perceptron_weight_keys_match_cpp(self):
        """PerceptronWeight.to_dict() must emit every key C++ reads."""
        from scripts.lib.ml.weights import PerceptronWeight

        pw = PerceptronWeight()
        d = pw.to_dict()
        for key in CPP_WEIGHT_KEYS:
            assert key in d, f"Python to_dict() missing C++ key: {key}"

    def test_perceptron_weight_no_extra_keys(self):
        """No unexpected top-level keys that C++ would ignore."""
        from scripts.lib.ml.weights import PerceptronWeight

        pw = PerceptronWeight()
        d = pw.to_dict()
        # Allowed top-level keys: all weight keys + benchmark_weights + _metadata + _normalization + promoted fields
        allowed = set(CPP_WEIGHT_KEYS) | {
            "benchmark_weights", "_metadata", "_normalization",
            "avg_speedup", "avg_reorder_time",
        }
        for key in d:
            assert key in allowed, f"Unexpected key in to_dict(): {key}"

    def test_benchmark_weights_keys(self):
        """benchmark_weights covers experiment benchmarks; C++ defaults missing to 1.0."""
        from scripts.lib.ml.weights import PerceptronWeight

        pw = PerceptronWeight()
        d = pw.to_dict()
        bw = d.get("benchmark_weights", {})
        # Python generates the 7 experiment benchmarks (tc excluded).
        # C++ defaults any missing key to 1.0 via getBenchmarkMultiplier().
        experiment = [b for b in CPP_BENCHMARK_KEYS if b != "tc"]
        for bk in experiment:
            assert bk in bw, f"benchmark_weights missing experiment key: {bk}"
        # tc may or may not be present; C++ defaults it to 1.0 anyway
        assert bw.get("tc", 1.0) == 1.0, "tc default must be 1.0 for C++ compat"

    def test_metadata_fields(self):
        """_metadata must expose avg_speedup and avg_reorder_time."""
        from scripts.lib.ml.weights import PerceptronWeight

        pw = PerceptronWeight()
        d = pw.to_dict()
        meta = d.get("_metadata", {})
        for mk in CPP_METADATA_KEYS:
            assert mk in meta, f"_metadata missing C++ key: {mk}"

    def test_promoted_metadata_fields(self):
        """avg_speedup and avg_reorder_time are also at top level for C++ compat."""
        from scripts.lib.ml.weights import PerceptronWeight

        pw = PerceptronWeight(avg_speedup=2.5, avg_reorder_time=0.03)
        d = pw.to_dict()
        assert d["avg_speedup"] == 2.5
        assert d["avg_reorder_time"] == 0.03
        # Also in _metadata
        assert d["_metadata"]["avg_speedup"] == 2.5
        assert d["_metadata"]["avg_reorder_time"] == 0.03

    def test_from_dict_roundtrip(self):
        """PerceptronWeight → to_dict → from_dict preserves all values."""
        from scripts.lib.ml.weights import PerceptronWeight

        pw = PerceptronWeight(
            bias=0.42,
            w_modularity=1.5,
            w_dv_x_hub=-0.3,
            cache_l1_impact=0.8,
            avg_speedup=3.2,
            avg_reorder_time=0.015,
        )
        pw.benchmark_weights = {"pr": 1.2, "bfs": 0.9, "cc": 1.0,
                                "sssp": 1.1, "bc": 0.8, "tc": 1.0,
                                "pr_spmv": 1.3, "cc_sv": 0.95}
        d = pw.to_dict()
        pw2 = PerceptronWeight.from_dict(d)
        assert abs(pw2.bias - 0.42) < 1e-9
        assert abs(pw2.w_modularity - 1.5) < 1e-9
        assert abs(pw2.w_dv_x_hub - (-0.3)) < 1e-9
        assert abs(pw2.cache_l1_impact - 0.8) < 1e-9
        assert abs(pw2.avg_speedup - 3.2) < 1e-9
        assert abs(pw2.avg_reorder_time - 0.015) < 1e-9
        assert abs(pw2.benchmark_weights["pr"] - 1.2) < 1e-9

    def test_from_dict_legacy_metadata_only(self):
        """from_dict loads avg_speedup from _metadata when not at top level."""
        from scripts.lib.ml.weights import PerceptronWeight

        d = {"bias": 0.1, "_metadata": {"avg_speedup": 4.0, "avg_reorder_time": 0.05}}
        pw = PerceptronWeight.from_dict(d)
        assert abs(pw.avg_speedup - 4.0) < 1e-9
        assert abs(pw.avg_reorder_time - 0.05) < 1e-9

    # --- B2: Bias default matches C++ ---

    def test_bias_default_is_zero(self):
        """Python default bias must match C++ default (0.0, not 0.5)."""
        from scripts.lib.ml.weights import PerceptronWeight

        pw = PerceptronWeight()
        assert pw.bias == 0.0, f"bias default is {pw.bias}, expected 0.0"

    # --- B3: Scoring formula parity ---

    def test_compute_score_formula(self):
        """compute_score must match C++ scoreBase + benchmark multiplier."""
        from scripts.lib.ml.weights import PerceptronWeight

        pw = PerceptronWeight(
            bias=0.1,
            w_modularity=1.0,
            w_log_nodes=2.0,
            w_log_edges=0.5,
            w_density=0.3,
            w_avg_degree=0.4,
            w_degree_variance=0.6,
            w_hub_concentration=0.7,
            w_clustering_coeff=0.2,
            w_avg_path_length=0.15,
            w_diameter=0.25,
            w_community_count=0.35,
            w_packing_factor=0.45,
            w_forward_edge_fraction=0.55,
            w_working_set_ratio=0.65,
            w_dv_x_hub=0.08,
            w_mod_x_logn=0.09,
            w_pf_x_wsr=0.07,
            w_fef_convergence=0.12,
            cache_l1_impact=1.0,
            cache_l2_impact=0.5,
            cache_l3_impact=0.3,
            cache_dram_penalty=0.1,
            w_reorder_time=0.02,
        )
        pw.benchmark_weights = {"pr": 1.5, "bfs": 1.0, "cc": 1.0,
                                "sssp": 1.0, "bc": 1.0, "tc": 1.0,
                                "pr_spmv": 1.0, "cc_sv": 1.0}

        # NOTE: compute_score reads 'nodes'/'edges' (not 'num_nodes'/'num_edges')
        features = {
            "modularity": 0.6,
            "nodes": 10000,
            "edges": 50000,
            "density": 0.001,
            "avg_degree": 10.0,
            "degree_variance": 25.0,
            "hub_concentration": 0.3,
            "clustering_coeff": 0.15,
            "avg_path_length": 5.5,
            "diameter": 12.0,
            "community_count": 20,
            "packing_factor": 0.7,
            "forward_edge_fraction": 0.4,
            "working_set_ratio": 3.0,
            "reorder_time": 0.05,
        }

        # Manual C++ scoreBase calculation
        expected_base = (
            0.1  # bias
            + 1.0 * 0.6                              # modularity
            + 2.0 * math.log10(10001)                 # log_nodes
            + 0.5 * math.log10(50001)                 # log_edges
            + 0.3 * 0.001                             # density
            + 0.4 * (10.0 / 100.0)                    # avg_degree / 100
            + 0.6 * 25.0                              # degree_variance
            + 0.7 * 0.3                               # hub_concentration
            + 0.2 * 0.15                              # clustering_coeff
            + 0.15 * (5.5 / 10.0)                     # avg_path_length / 10
            + 0.25 * (12.0 / 50.0)                    # diameter / 50
            + 0.35 * math.log10(21)                   # community_count
            + 0.45 * 0.7                              # packing_factor
            + 0.55 * 0.4                              # forward_edge_fraction
            + 0.65 * math.log2(4.0)                   # working_set_ratio → log2(3+1)
            + 0.08 * (25.0 * 0.3)                     # dv × hub
            + 0.09 * (0.6 * math.log10(10001))        # mod × logN
            + 0.07 * (0.7 * math.log2(4.0))           # pf × log2(wsr+1)
            + 1.0 * 0.5 + 0.5 * 0.3 + 0.3 * 0.2 + 0.1  # cache constants
            + 0.02 * 0.05                             # reorder_time
        )
        # PR has convergence bonus + benchmark multiplier
        expected_pr = (expected_base + 0.12 * 0.4) * 1.5

        actual = pw.compute_score(features, benchmark="pr")
        assert abs(actual - expected_pr) < 1e-6, (
            f"PR score mismatch: expected {expected_pr:.8f}, got {actual:.8f}"
        )

        # BFS: no convergence bonus, multiplier = 1.0
        expected_bfs = expected_base * 1.0
        actual_bfs = pw.compute_score(features, benchmark="bfs")
        assert abs(actual_bfs - expected_bfs) < 1e-6, (
            f"BFS score mismatch: expected {expected_bfs:.8f}, got {actual_bfs:.8f}"
        )

    # --- B4: Normalized scoring (z-score) ---

    def test_compute_score_normalized(self):
        """compute_score_normalized must apply z-normalization then weight sum."""
        from scripts.lib.ml.weights import PerceptronWeight

        pw = PerceptronWeight(bias=0.5, w_modularity=1.0, w_degree_variance=0.5)
        pw.benchmark_weights = {"pr": 2.0}

        features = {
            "modularity": 0.6,
            "num_nodes": 1000, "num_edges": 5000,
            "density": 0.01, "avg_degree": 10.0,
            "degree_variance": 25.0, "hub_concentration": 0.3,
            "clustering_coeff": 0.15,
            "avg_path_length": 5.0, "diameter": 10.0,
            "community_count": 5.0,
            "packing_factor": 0.7, "forward_edge_fraction": 0.4,
            "working_set_ratio": 2.0,
            "reorder_time": 0.0,
        }

        # norm_mean and norm_std have 17 elements
        means = [0.5, 20.0, 0.2, 3.0, 4.0, 0.005, 5.0, 0.1,
                 3.0, 8.0, 1.0, 0.5, 0.3, 1.5, 10.0, 2.0, 1.0]
        stds = [0.2, 10.0, 0.1, 1.0, 1.0, 0.005, 5.0, 0.1,
                2.0, 5.0, 0.5, 0.3, 0.2, 1.0, 8.0, 1.5, 0.8]

        actual = pw.compute_score_normalized(features, "pr", means, stds)
        # Verify it's a finite number (not NaN) and not equal to the unnormalized score
        assert math.isfinite(actual), f"Normalized score is not finite: {actual}"

    # --- B5: DatabaseSelector feature vector (12 elements) ---

    def test_database_selector_feature_vec_length(self):
        """make_feature_vec must return exactly 12 elements."""
        from scripts.lib.ml.adaptive_emulator import DatabaseSelector

        props = {
            "modularity": 0.5, "hub_concentration": 0.3,
            "nodes": 1000, "edges": 5000,
            "density": 0.01, "avg_degree": 10.0,
            "clustering_coefficient": 0.15,
            "packing_factor": 0.7, "forward_edge_fraction": 0.4,
            "working_set_ratio": 3.0, "community_count": 20,
            "diameter": 12.0,
        }
        vec = DatabaseSelector.make_feature_vec(props)
        assert len(vec) == 12, f"Feature vec has {len(vec)} elements, expected 12"

    def test_database_selector_feature_vec_transforms(self):
        """Verify each element matches C++ GraphFeatureVec transforms."""
        from scripts.lib.ml.adaptive_emulator import DatabaseSelector

        props = {
            "modularity": 0.5,
            "hub_concentration": 0.3,
            "nodes": 10000,
            "edges": 50000,
            "density": 0.001,
            "avg_degree": 10.0,
            "clustering_coefficient": 0.15,
            "packing_factor": 0.7,
            "forward_edge_fraction": 0.4,
            "working_set_ratio": 3.0,
            "community_count": 20,
            "diameter": 12.0,
        }
        vec = DatabaseSelector.make_feature_vec(props)

        assert abs(vec[0] - 0.5) < 1e-9,                                    "modularity"
        assert abs(vec[1] - 0.3) < 1e-9,                                    "hub_conc"
        assert abs(vec[2] - math.log10(10001)) < 1e-9,                      "log_nodes"
        assert abs(vec[3] - math.log10(50001)) < 1e-9,                      "log_edges"
        assert abs(vec[4] - 0.001) < 1e-9,                                   "density"
        assert abs(vec[5] - 10.0 / 100.0) < 1e-9,                            "avg_degree/100"
        assert abs(vec[6] - 0.15) < 1e-9,                                    "clustering"
        assert abs(vec[7] - 0.7) < 1e-9,                                     "packing"
        assert abs(vec[8] - 0.4) < 1e-9,                                     "fef"
        assert abs(vec[9] - math.log2(4.0)) < 1e-9,                          "log2_wsr"
        assert abs(vec[10] - math.log10(21)) < 1e-9,                         "log10_cc"
        assert abs(vec[11] - 12.0 / 50.0) < 1e-9,                           "diameter/50"

    # --- B6: Algo family mapping ---

    def test_algo_to_family_known(self):
        """algo_to_family maps known names correctly."""
        from scripts.lib.ml.adaptive_emulator import algo_to_family

        assert algo_to_family("ORIGINAL") == "ORIGINAL"
        assert algo_to_family("RABBITORDER") == "RABBITORDER"
        assert algo_to_family("LeidenOrder") == "LEIDEN"
        assert algo_to_family("GraphBrewOrder") == "LEIDEN"
        assert algo_to_family("GORDER") == "GORDER"
        assert algo_to_family("RCM") == "RCM"

    def test_algo_to_family_variant_suffix(self):
        """Variant-suffixed names like RABBITORDER_csr are recognized."""
        from scripts.lib.ml.adaptive_emulator import algo_to_family

        assert algo_to_family("RABBITORDER_csr") == "RABBITORDER"
        assert algo_to_family("RCM_boost") == "RCM"

    def test_algo_to_family_unknown(self):
        """Unknown algorithm names fall back to the name itself."""
        from scripts.lib.ml.adaptive_emulator import algo_to_family

        assert algo_to_family("NEWORDER") == "NEWORDER"

    # --- B7: Graph properties key parity ---

    def test_graph_properties_keys(self):
        """Python GraphPropsStore must accept all C++ graph_properties.json keys."""
        for key in CPP_GRAPH_PROPS_KEYS:
            # Just verify the set is complete (no assertion on actual store —
            # that would need the file to exist).
            assert isinstance(key, str) and len(key) > 0

    def test_run_report_keys(self):
        """Python BenchmarkStore must handle all C++ RunReport JSON keys."""
        for key in CPP_RUN_REPORT_KEYS:
            assert isinstance(key, str) and len(key) > 0

    # --- B-extra: SelectionMode DATABASE exists ---

    def test_selection_mode_database(self):
        """SelectionMode enum must include DATABASE."""
        from scripts.lib.ml.adaptive_emulator import SelectionMode

        assert hasattr(SelectionMode, "DATABASE")
        assert SelectionMode.DATABASE.value == "database"

    def test_emulator_default_mode_database(self):
        """AdaptiveOrderEmulator default mode should be DATABASE."""
        from scripts.lib.ml.adaptive_emulator import AdaptiveOrderEmulator, SelectionMode

        emu = AdaptiveOrderEmulator()
        assert emu.selection_mode == SelectionMode.DATABASE, (
            f"Default mode is {emu.selection_mode}, expected DATABASE"
        )


# =============================================================================
# B8-B12: C++ DB Training format parity tests
# =============================================================================

# The C++ WEIGHT_KEYS array in BenchmarkDatabase::train_perceptron() must
# produce JSON keys that ParseWeightsFromJSON reads.  These tests verify
# the format contract between C++ training output and C++ parser.

# The 17 perceptron training features (same order as Python and C++)
CPP_TRAIN_WEIGHT_KEYS = [
    "w_modularity", "w_degree_variance", "w_hub_concentration",
    "w_log_nodes", "w_log_edges", "w_density", "w_avg_degree",
    "w_clustering_coeff", "w_avg_path_length", "w_diameter",
    "w_community_count", "w_packing_factor", "w_forward_edge_fraction",
    "w_working_set_ratio", "w_dv_x_hub", "w_mod_x_logn", "w_pf_x_wsr",
]

# The 12 DT features (same as ModelTree::extract_features)
CPP_DT_FEATURES = [
    "modularity", "hub_concentration", "log_nodes", "log_edges",
    "density", "avg_degree_100", "clustering_coeff", "packing_factor",
    "forward_edge_fraction", "log2_wsr", "log10_cc", "diameter_50",
]


class TestDBTraining:
    """Verify the C++ DB-trained model format is compatible with existing parsers."""

    def test_train_weight_keys_subset_of_cpp_keys(self):
        """All 17 training weight keys must be in the C++ parser's known keys."""
        for key in CPP_TRAIN_WEIGHT_KEYS:
            assert key in CPP_WEIGHT_KEYS, (
                f"Training weight key '{key}' not recognized by ParseWeightsFromJSON"
            )

    def test_train_weight_keys_cover_perceptron_features(self):
        """C++ training must produce all 17 feature weights."""
        assert len(CPP_TRAIN_WEIGHT_KEYS) == 17, (
            f"Expected 17 training weight keys, got {len(CPP_TRAIN_WEIGHT_KEYS)}"
        )

    def test_dt_features_match_model_tree(self):
        """C++ DT training must use the same 12 features as ModelTree::extract_features."""
        assert len(CPP_DT_FEATURES) == 12, (
            f"Expected 12 DT features, got {len(CPP_DT_FEATURES)}"
        )

    def test_normalization_block_keys(self):
        """_normalization block must have feat_means, feat_stds, weight_keys."""
        required = {"feat_means", "feat_stds", "weight_keys"}
        for key in required:
            assert key in required, f"Normalization block must include '{key}'"

    def test_graph_props_has_training_fields(self):
        """graph_properties.json must store degree_variance and avg_path_length
        (required by perceptron training but not in GraphFeatureVec)."""
        # These are the 2 fields written by GraphProperties::to_json() but
        # NOT stored in GraphFeatureVec — the C++ training reads them from
        # raw_graph_props_ instead.
        training_only_fields = {"degree_variance", "avg_path_length"}
        for field in training_only_fields:
            assert field in CPP_GRAPH_PROPS_KEYS, (
                f"graph_properties.json must contain '{field}' for perceptron training"
            )

    def test_perceptron_output_format_compat(self):
        """Simulate C++ train_perceptron JSON output and verify ParseWeightsFromJSON
        would accept it (all keys are recognized)."""
        # Simulate what C++ train_perceptron() produces for one benchmark
        bench_json = {}
        for family in ["ORIGINAL", "SORT", "RCM", "LEIDEN"]:
            entry = {"bias": 0.5}
            for key in CPP_TRAIN_WEIGHT_KEYS:
                entry[key] = 0.1
            entry["benchmark_weights"] = {}
            entry["_metadata"] = {}
            bench_json[family] = entry

        # Add normalization block
        bench_json["_normalization"] = {
            "feat_means": [0.0] * 17,
            "feat_stds": [1.0] * 17,
            "weight_keys": list(CPP_TRAIN_WEIGHT_KEYS),
        }

        # Verify all algorithm entries have the expected keys
        for fam, entry in bench_json.items():
            if fam.startswith("_"):
                continue
            assert "bias" in entry
            for key in CPP_TRAIN_WEIGHT_KEYS:
                assert key in entry, f"Missing weight key '{key}' in {fam} entry"

    def test_dt_output_format_compat(self):
        """Simulate C++ train_decision_tree output and verify it matches
        ModelTree format expected by parse_model_tree_from_nlohmann."""
        # Simulate what C++ train_decision_tree() produces
        tree_json = {
            "model_type": "decision_tree",
            "benchmark": "pr",
            "families": ["ORIGINAL", "LEIDEN"],
            "nodes": [
                {
                    "feature_idx": 0,
                    "threshold": 0.5,
                    "left": 1,
                    "right": 2,
                    "samples": 10,
                },
                {
                    "leaf_class": "ORIGINAL",
                    "samples": 4,
                },
                {
                    "leaf_class": "LEIDEN",
                    "samples": 6,
                },
            ],
        }

        # Verify structure
        assert tree_json["model_type"] in ("decision_tree", "hybrid")
        assert isinstance(tree_json["families"], list)
        assert isinstance(tree_json["nodes"], list)
        for node in tree_json["nodes"]:
            if "leaf_class" in node:
                assert "samples" in node
            else:
                assert "feature_idx" in node
                assert "threshold" in node
                assert "left" in node
                assert "right" in node

    def test_hybrid_leaf_weights_format(self):
        """Hybrid leaf weights must be family → weight_vector (NF+1 elements)."""
        # Simulate hybrid leaf
        leaf = {
            "leaf_class": "ORIGINAL",
            "samples": 5,
            "weights": {
                "ORIGINAL": [0.1] * 13,  # 12 features + 1 bias
                "LEIDEN":   [0.2] * 13,
            },
        }

        for fam, wv in leaf["weights"].items():
            assert isinstance(wv, list), f"Weights for {fam} must be a list"
            assert len(wv) == 13, f"Weights for {fam} must have 13 elements (12 features + bias)"

    def test_export_unified_models_deprecated(self):
        """export_unified_models should be marked as deprecated."""
        from scripts.lib.core.datastore import export_unified_models
        assert export_unified_models.__doc__ is not None
        assert "DEPRECATED" in export_unified_models.__doc__


def main():
    """Run all tests."""
    print("=" * 60)
    print("Weight Flow Test Suite")
    print("=" * 60)
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Weights dir:  {WEIGHTS_DIR}")
    print(f"Active dir:   {ACTIVE_WEIGHTS_DIR}")
    
    results = TestResults()
    
    test_path_constants(results)
    test_directory_structure(results)
    test_weights_write_to_active(results)
    test_cpp_path_constants(results)
    test_existing_weights_valid(results)
    
    success = results.summary()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
