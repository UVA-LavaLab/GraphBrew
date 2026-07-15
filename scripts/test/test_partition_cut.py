#!/usr/bin/env python3

import unittest

from scripts.experiments.partition_cut.phase1 import (
    add_relative_metrics,
    policy_is_stable,
    summarize_policy,
    validate_cross_policy,
)


def record(policy: str, suffix: str = "a", threads: int = 1) -> dict:
    return {
        "policy": policy,
        "deterministic_required": True,
        "threads": threads,
        "repeat": 0,
        "wall_seconds": 2.0,
        "bfs_seconds": 1.0,
        "reorder_seconds": 0.5,
        "partition_seconds": 0.25,
        "diagnostics_seconds": 0.1,
        "remote_out_fraction": 0.4,
        "remote_in_fraction": 0.5,
        "max_remote_out_fraction": 0.6,
        "max_remote_in_fraction": 0.7,
        "ghost_count": 10,
        "ghost_bytes": 80,
        "ghost_byte_fraction": 0.2,
        "total_shard_bytes": 400,
        "max_shard_bytes": 120,
        "vertex_imbalance": 1.1,
        "balance_imbalance": 1.05,
        "out_edge_imbalance": 1.2,
        "in_edge_imbalance": 1.3,
        "storage_imbalance": 1.4,
        "mapping_fingerprint": f"mapping-{suffix}",
        "source_topology_fingerprint": "topology",
        "shard_fingerprint": f"shards-{suffix}",
        "ghost_fingerprint": f"ghosts-{suffix}",
        "source_id": 0,
        "depth_fingerprint": "depth",
        "reachable_vertices": 5,
        "max_depth": 3,
    }


class TestPartitionCutPhase1(unittest.TestCase):
    def test_policy_stability_uses_all_fingerprints(self):
        self.assertTrue(policy_is_stable([record("p"), record("p")]))
        self.assertFalse(
            policy_is_stable([record("p"), record("p", suffix="b")])
        )

    def test_summary_uses_conservative_remote_and_imbalance(self):
        baseline = summarize_policy(
            [record("original"), record("original", threads=32)]
        )
        summary = summarize_policy(
            [record("p"), record("p", threads=32)]
        )
        summary["remote_fraction"] = 0.25
        summary["ghost_bytes"] = 40
        summary["max_shard_bytes"] = 125
        add_relative_metrics([baseline, summary])
        self.assertEqual(summary["remote_fraction"], 0.25)
        self.assertEqual(summary["max_remote_fraction"], 0.7)
        self.assertEqual(summary["balance_imbalance"], 1.05)
        self.assertEqual(summary["max_edge_imbalance"], 1.3)
        self.assertEqual(summary["remote_reduction"], 2.0)
        self.assertEqual(summary["ghost_reduction"], 2.0)
        self.assertAlmostEqual(summary["max_shard_ratio"], 125 / 120)
        self.assertTrue(summary["passes_widening_cut_gate"])
        self.assertTrue(summary["passes_work_balance_gate"])
        self.assertTrue(summary["passes_capacity_gate"])
        self.assertTrue(summary["stable"])
        self.assertEqual(summary["primary_threads"], 32)
        self.assertEqual(
            sorted(summary["timing_by_threads"]),
            ["1", "32"],
        )

    def test_cross_policy_requires_same_source_topology_and_depth(self):
        validate_cross_policy([record("a"), record("b", suffix="b")])
        changed = record("b")
        changed["depth_fingerprint"] = "different"
        with self.assertRaisesRegex(RuntimeError, "BFS source-depth"):
            validate_cross_policy([record("a"), changed])


if __name__ == "__main__":
    unittest.main()
