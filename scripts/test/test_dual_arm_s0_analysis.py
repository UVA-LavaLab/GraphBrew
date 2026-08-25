from scripts.experiments.vldb.analyze_dual_arm_s0 import dominates


def record(mapping, reuse):
    return {
        "mapping_ratio_vector": mapping,
        "reuse_ratio_vector": reuse,
    }


def test_dominates_prefers_lower_mapping_and_higher_reuse():
    assert dominates(
        record([0.8, 0.9], [1.1, 1.2, 1.05, 1.08]),
        record([0.9, 1.0], [1.0, 1.1, 1.01, 1.02]),
    )


def test_dominates_requires_no_regression():
    assert not dominates(
        record([0.8, 0.9], [0.99, 1.2]),
        record([0.9, 1.0], [1.0, 1.1]),
    )


def test_dominates_requires_strict_improvement():
    candidate = record([0.8, 0.9], [1.1, 1.2])
    assert not dominates(candidate, candidate)
