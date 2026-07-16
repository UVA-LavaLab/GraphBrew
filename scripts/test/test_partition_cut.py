#!/usr/bin/env python3

from pathlib import Path
import json
import unittest

from scripts.experiments.partition_cut.phase1 import (
    add_relative_metrics,
    policy_is_stable,
    summarize_policy,
    validate_cross_policy,
)
from scripts.experiments.partition_cut.phase2 import (
    DEFAULT_MAX_SHARD_BYTES,
    POLICIES as PHASE2_POLICIES,
    classify_determinism,
    converter_command,
    geometric_mean,
    graph_needs_download,
    parse_runtime_config,
    selected_graphs,
    summarize_graph_records,
    summarize_corpus,
    summarize_phase2_policy,
    validate_record_matrix,
    validate_runtime_config,
    validate_runtime_traffic,
)
from scripts.experiments.vldb.config import COMPOSE_VARIANTS
from scripts.experiments.partition_cut.freeze_phase2 import (
    compact_runtime_traffic,
    freeze_matrix,
    normalize_paths,
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
        "max_depth": 1,
        "runtime_config": {},
        "run_metadata": {
            "bfs_binary": "bench/bin/bfs_p",
        },
        "runtime_traffic": {
            "schema": "graphbrew.partition_runtime_traffic.v1",
            "ghost_slots": 10,
            "graphblox_projection": {
                "bfs_bytes_per_superstep": 80,
                "pr_bytes_per_iteration": 40,
                "cc_bytes_per_iteration": 40,
                "spmv_initial_bytes": 40,
                "shards": [
                    {
                        "shard_id": 0,
                        "ghost_slots": 10,
                        "bfs_bytes_per_superstep": 80,
                        "pr_bytes_per_iteration": 40,
                        "cc_bytes_per_iteration": 40,
                        "spmv_initial_bytes": 40,
                    },
                ],
            },
            "bfs": {
                "supersteps": 2,
                "cpu_ghost_sync_values": 10,
                "cpu_ghost_sync_bytes": 10,
                "remote_parent_messages": 5,
                "remote_parent_bytes": 20,
                "graphblox_halo_values": 40,
                "graphblox_halo_bytes": 160,
                "steps": [
                    {
                        "step": 0,
                        "phase": "p-bsp-td",
                        "cpu_ghost_sync_bytes": 0,
                        "remote_parent_messages": 5,
                        "remote_parent_bytes": 20,
                        "graphblox_halo_bytes": 80,
                        "shards": [
                            {
                                "shard_id": 0,
                                "cpu_ghost_sync_bytes": 0,
                                "remote_parent_messages": 5,
                                "remote_parent_bytes": 20,
                                "graphblox_halo_bytes": 80,
                            },
                        ],
                    },
                    {
                        "step": 1,
                        "phase": "p-bsp-bu",
                        "cpu_ghost_sync_bytes": 10,
                        "remote_parent_messages": 0,
                        "remote_parent_bytes": 0,
                        "graphblox_halo_bytes": 80,
                        "shards": [
                            {
                                "shard_id": 0,
                                "cpu_ghost_sync_bytes": 10,
                                "remote_parent_messages": 0,
                                "remote_parent_bytes": 0,
                                "graphblox_halo_bytes": 80,
                            },
                        ],
                    },
                ],
            },
        },
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


class TestPartitionCutPhase2(unittest.TestCase):
    def test_smoke_preset_covers_requested_graph_classes(self):
        self.assertEqual(
            DEFAULT_MAX_SHARD_BYTES,
            512 * 1024 * 1024,
        )
        graphs = selected_graphs("smoke", None)
        self.assertEqual(
            {graph.category for graph in graphs},
            {"road", "mesh", "citation", "social"},
        )
        self.assertEqual(len(graphs), 4)

    def test_phase2_policy_tokens_cover_new_variants(self):
        self.assertEqual(
            PHASE2_POLICIES["sg_hilbert"].options,
            (
                "-o",
                "12:leiden:compose:"
                "sg_hilbert:comm_identity:intra_hubsort",
            ),
        )
        self.assertIn(
            "intra_hub2",
            PHASE2_POLICIES["intra_hub2"].options[1],
        )
        self.assertIn(
            "intra_rcmpp",
            PHASE2_POLICIES["intra_rcmpp"].options[1],
        )
        self.assertFalse(
            PHASE2_POLICIES["comm_cut_min"].deterministic_required
        )
        self.assertTrue(
            all(
                option.startswith("12:")
                for _label, option in COMPOSE_VARIANTS
            )
        )

    def test_converter_preserves_native_directed_order(self):
        road = selected_graphs(
            "smoke", ["roadNet-PA"])[0]
        citation = selected_graphs(
            "smoke", ["cit-HepPh"])[0]
        road_command = converter_command(
            Path("converter"),
            road,
            Path("road.mtx"),
            Path("road.sg"),
        )
        citation_command = converter_command(
            Path("converter"),
            citation,
            Path("citation.mtx"),
            Path("citation.sg"),
        )
        self.assertIn("-s", road_command)
        self.assertNotIn("-s", citation_command)
        self.assertNotIn("-o", road_command)
        self.assertNotIn("-o", citation_command)

    def test_missing_conversion_source_requires_download(self):
        import tempfile

        graph = selected_graphs(
            "smoke", ["cit-HepPh"])[0]
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            output = graph.path(root)
            output.parent.mkdir(parents=True)
            output.write_bytes(b"serialized")
            self.assertTrue(graph_needs_download(graph, root))
            (output.parent / "cit-HepPh.mtx").write_text(
                "%%MatrixMarket matrix coordinate pattern general\n"
            )
            self.assertFalse(graph_needs_download(graph, root))

    def test_geometric_mean_and_cross_class_gate(self):
        self.assertAlmostEqual(geometric_mean([1.0, 4.0]), 2.0)
        baseline = summarize_policy([record("original")])
        baseline["remote_reduction"] = 1.0
        baseline["ghost_reduction"] = 1.0
        baseline["bfs_halo_reduction"] = 1.0
        baseline["bfs_cpu_ghost_sync_reduction"] = 1.0
        baseline["bfs_remote_parent_reduction"] = 1.0
        baseline["bfs_cpu_total_reduction"] = 1.0
        baseline["pr_halo_reduction"] = 1.0
        baseline["cc_halo_reduction"] = 1.0
        baseline["spmv_halo_reduction"] = 1.0
        baseline["max_shard_ratio"] = 1.0
        baseline["preprocess_ratio"] = 1.0
        candidate_a = dict(baseline)
        candidate_a.update({
            "policy": "candidate",
            "remote_reduction": 2.0,
            "ghost_reduction": 2.0,
            "bfs_halo_reduction": 2.0,
            "bfs_cpu_ghost_sync_reduction": 2.0,
            "bfs_remote_parent_reduction": 2.0,
            "bfs_cpu_total_reduction": 2.0,
            "pr_halo_reduction": 2.0,
            "cc_halo_reduction": 2.0,
            "spmv_halo_reduction": 2.0,
            "max_shard_ratio": 1.05,
            "balance_imbalance": 1.02,
            "storage_imbalance": 1.02,
            "max_edge_imbalance": 1.02,
            "passes_absolute_capacity_gate": True,
            "preprocess_ratio": 2.0,
            "stable": True,
            "passes_widening_cut_gate": True,
        })
        candidate_b = dict(candidate_a)
        candidate_b.update({
            "remote_reduction": 1.5,
            "ghost_reduction": 1.6,
            "bfs_halo_reduction": 1.5,
            "bfs_cpu_ghost_sync_reduction": 1.5,
            "bfs_remote_parent_reduction": 1.5,
            "bfs_cpu_total_reduction": 1.5,
            "pr_halo_reduction": 1.5,
            "cc_halo_reduction": 1.5,
            "spmv_halo_reduction": 1.5,
        })
        graph_results = [
            {
                "category": "road",
                "summaries_by_policy": {
                    "original": baseline,
                    "candidate": candidate_a,
                },
            },
            {
                "category": "social",
                "summaries_by_policy": {
                    "original": baseline,
                    "candidate": candidate_b,
                },
            },
        ]
        aggregates = summarize_corpus(
            graph_results, ["original", "candidate"]
        )
        candidate = next(
            item
            for item in aggregates
            if item["policy"] == "candidate"
        )
        self.assertTrue(candidate["stable_on_all_graphs"])
        self.assertAlmostEqual(
            candidate["geomean_remote_reduction"],
            geometric_mean([2.0, 1.5]),
        )
        self.assertTrue(
            candidate["passes_universal_default_gate"]
        )

    def test_runtime_traffic_rejects_negative_and_inconsistent_values(self):
        valid = record("original")
        validate_runtime_traffic(valid)

        negative = record("original")
        negative["runtime_traffic"]["bfs"][
            "remote_parent_bytes"
        ] = -1
        with self.assertRaisesRegex(
            RuntimeError, "nonnegative integer"
        ):
            validate_runtime_traffic(negative)

        inconsistent = record("original")
        inconsistent["runtime_traffic"]["bfs"][
            "graphblox_halo_bytes"
        ] = 80
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            validate_runtime_traffic(inconsistent)

        bottom_up_remote = record("original")
        bottom_up_remote["runtime_traffic"]["bfs"]["steps"][1][
            "remote_parent_messages"
        ] = 1
        bottom_up_remote["runtime_traffic"]["bfs"]["steps"][1][
            "remote_parent_bytes"
        ] = 4
        bottom_up_remote["runtime_traffic"]["bfs"]["steps"][1][
            "shards"
        ][0]["remote_parent_messages"] = 1
        bottom_up_remote["runtime_traffic"]["bfs"]["steps"][1][
            "shards"
        ][0]["remote_parent_bytes"] = 4
        bottom_up_remote["runtime_traffic"]["bfs"][
            "remote_parent_messages"
        ] = 6
        bottom_up_remote["runtime_traffic"]["bfs"][
            "remote_parent_bytes"
        ] = 24
        with self.assertRaisesRegex(
            RuntimeError, "bottom-up remote-parent"
        ):
            validate_runtime_traffic(bottom_up_remote)

        wrong_depth = record("original")
        wrong_depth["max_depth"] = 2
        with self.assertRaisesRegex(RuntimeError, "supersteps"):
            validate_runtime_traffic(wrong_depth)

        too_large = record("original")
        too_large["runtime_traffic"]["ghost_slots"] = 1 << 64
        with self.assertRaisesRegex(
            RuntimeError, "nonnegative integer"
        ):
            validate_runtime_traffic(too_large)

    def test_zero_candidate_traffic_uses_json_safe_encoding(self):
        from scripts.experiments.partition_cut.phase2 import (
            add_runtime_relative_metrics,
        )

        baseline = summarize_phase2_policy([record("original")])
        candidate = dict(baseline)
        candidate["policy"] = "candidate"
        for field in (
            "bfs_cpu_ghost_sync_bytes",
            "bfs_remote_parent_bytes",
            "bfs_cpu_total_bytes",
            "bfs_graphblox_halo_bytes",
            "pr_halo_bytes_per_iteration",
            "cc_halo_bytes_per_iteration",
            "spmv_initial_halo_bytes",
        ):
            candidate[field] = 0
        add_runtime_relative_metrics([baseline, candidate])
        self.assertIsNone(candidate["bfs_cpu_total_reduction"])
        json.dumps(candidate, allow_nan=False)

    def test_runtime_traffic_changes_for_identical_fingerprints_fail(self):
        first = record("original")
        second = record("original", threads=32)
        second["runtime_traffic"]["bfs"]["steps"][0][
            "remote_parent_messages"
        ] = 4
        second["runtime_traffic"]["bfs"]["steps"][0][
            "remote_parent_bytes"
        ] = 16
        second["runtime_traffic"]["bfs"]["steps"][0]["shards"][0][
            "remote_parent_messages"
        ] = 4
        second["runtime_traffic"]["bfs"]["steps"][0]["shards"][0][
            "remote_parent_bytes"
        ] = 16
        second["runtime_traffic"]["bfs"][
            "remote_parent_messages"
        ] = 4
        second["runtime_traffic"]["bfs"][
            "remote_parent_bytes"
        ] = 16
        with self.assertRaisesRegex(
            RuntimeError, "changed for identical"
        ):
            summarize_phase2_policy([first, second])

    def test_frozen_runtime_traffic_is_compact_and_auditable(self):
        compact = compact_runtime_traffic(
            record("original")["runtime_traffic"])
        self.assertNotIn("steps", compact["bfs"])
        self.assertEqual(
            compact["bfs"]["phase_totals"]["p-bsp-td"]
            ["remote_parent_messages"],
            5,
        )
        self.assertEqual(
            compact["bfs"]["shards"][0]
            ["cpu_ghost_sync_bytes"],
            10,
        )
        self.assertEqual(
            len(compact["bfs"]["steps_sha256"]), 64)

    def test_freezer_rejects_stale_runtime_summaries(self):
        import tempfile

        project_root = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory(
            dir=project_root / "results"
        ) as raw_temp:
            root = Path(raw_temp)
            records = []
            for threads in (1, 32):
                for repeat in (0, 1):
                    item = record("original", threads=threads)
                    item["repeat"] = repeat
                    records.append(item)
                    run_dir = (
                        root
                        / "tiny"
                        / f"original-t{threads}-r{repeat}"
                    )
                    run_dir.mkdir(parents=True)
                    (run_dir / "stdout.log").write_text("ok\n")
                    (run_dir / "stderr.log").write_text("")
                    (run_dir / "benchmarks.json").write_text("[]\n")
            policies = [PHASE2_POLICIES["original"]]
            summaries = summarize_graph_records(
                records, policies, DEFAULT_MAX_SHARD_BYTES)
            graph_result = {
                "graph": "tiny",
                "category": "road",
                "size_tier": "smoke",
                "path": "tiny.sg",
                "graph_identity": {},
                "summaries": summaries,
                "summaries_by_policy": {
                    "original": summaries[0],
                },
                "records": records,
            }
            aggregate = summarize_corpus(
                [graph_result], ["original"])
            summary = {
                "correct": True,
                "determinism_complete": True,
                "schema": "graphbrew.partition_cut.phase2.v1",
                "preset": "smoke",
                "graphs": [{"name": "tiny"}],
                "partitions": 1,
                "balance": "total",
                "source": 0,
                "threads": [1, 32],
                "repeats": 2,
                "policies": [{"name": "original"}],
                "max_shard_bytes": DEFAULT_MAX_SHARD_BYTES,
                "valid": True,
                "determinism_passed": True,
                "graph_results": [graph_result],
                "aggregates": aggregate,
            }
            path = root / "phase2_summary.json"
            path.write_text(json.dumps(summary))
            freeze_matrix("test", path)
            summary["graph_results"][0]["summaries"][0][
                "bfs_cpu_total_bytes"
            ] += 1
            path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(
                RuntimeError, "stale policy summaries"
            ):
                freeze_matrix("test", path)

    def test_determinism_classifies_thread_and_repeat_variation(self):
        deterministic = [
            record("p", threads=1),
            record("p", threads=1),
            record("p", threads=32),
            record("p", threads=32),
        ]
        self.assertEqual(
            classify_determinism(deterministic)["classification"],
            "deterministic",
        )
        thread_variant = [
            record("p", suffix="a", threads=1),
            record("p", suffix="a", threads=1),
            record("p", suffix="b", threads=32),
            record("p", suffix="b", threads=32),
        ]
        self.assertEqual(
            classify_determinism(thread_variant)["classification"],
            "thread_variant",
        )
        repeat_variant = [
            record("p", suffix="a", threads=1),
            record("p", suffix="b", threads=1),
        ]
        self.assertEqual(
            classify_determinism(repeat_variant)["classification"],
            "repeat_variant",
        )
        self.assertEqual(
            classify_determinism(
                [record("p", threads=32)]
            )["classification"],
            "insufficient_evidence",
        )

    def test_runtime_config_detects_cut_min_fallback(self):
        parsed = parse_runtime_config(
            "GraphBrew: aggregation=gve-csr, ordering=compose, "
            "refinement=on (depth=0)\n"
            "GraphBrew: mComputation=total-edges (GVE style)\n"
            "GraphBrew: 3 passes, 8 iters, 5000 communities, "
            "time=1.0s\n"
            "  compose: sg=none "
            "comm=cut_min_fallback_degree_desc "
            "intra=hubsort refine=none, 5000 communities, "
            "0.1s\n"
            "Algorithm:           GraphBrewOrder\n"
        )
        self.assertEqual(parsed["communities"], 5000)
        self.assertTrue(parsed["cut_min_fallback"])
        validate_runtime_config(
            PHASE2_POLICIES["comm_cut_min"], parsed)
        validate_runtime_config(
            PHASE2_POLICIES["original"], {})
        validate_runtime_config(
            PHASE2_POLICIES["rcm_bnf"],
            {
                "algorithm": "RCMOrder",
                "rcm_variant": "bnf",
            },
        )
        validate_runtime_config(
            PHASE2_POLICIES["gorder_csr"],
            {
                "algorithm": "GOrder",
                "gorder_variant": "csr",
            },
        )
        with self.assertRaisesRegex(
            RuntimeError, "runtime configuration"
        ):
            validate_runtime_config(
                PHASE2_POLICIES["rcm_bnf"],
                {
                    "algorithm": "DefinitelyWrong",
                    "rcm_variant": "bnf",
                },
            )
        with self.assertRaisesRegex(
            RuntimeError, "runtime configuration"
        ):
            validate_runtime_config(
                PHASE2_POLICIES["gorder_csr"],
                {"algorithm": "GOrder"},
            )
        wrong_compose = dict(parsed)
        wrong_compose["m_computation"] = "half-edges"
        wrong_compose["refine"] = "2swap"
        with self.assertRaisesRegex(
            RuntimeError, "runtime configuration"
        ):
            validate_runtime_config(
                PHASE2_POLICIES["comm_cut_min"],
                wrong_compose,
            )

    def test_existing_record_matrix_is_exact_and_provenanced(self):
        metadata = {
            ("original", 1, 0): {"config": "expected"},
        }
        valid = record("original")
        valid["run_metadata"] = {"config": "expected"}
        self.assertEqual(
            validate_record_matrix([valid], metadata),
            [valid],
        )
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            validate_record_matrix([valid, valid], metadata)
        stale = dict(valid)
        stale["run_metadata"] = {"config": "stale"}
        with self.assertRaisesRegex(RuntimeError, "metadata"):
            validate_record_matrix([stale], metadata)

    def test_phase2_quality_uses_worst_primary_repeat(self):
        serial = record("p", threads=1)
        serial.update({
            "max_shard_bytes": 1000,
            "storage_imbalance": 1.3,
            "out_edge_imbalance": 1.5,
        })
        first = record("p", threads=32)
        second = record("p", threads=32)
        second.update({
            "remote_out_fraction": 0.8,
            "remote_in_fraction": 0.7,
            "ghost_bytes": 120,
            "max_shard_bytes": 180,
            "storage_imbalance": 1.2,
            "out_edge_imbalance": 1.4,
        })
        summary = summarize_phase2_policy(
            [serial, first, second])
        self.assertEqual(summary["remote_fraction"], 0.8)
        self.assertEqual(summary["ghost_bytes"], 120)
        self.assertEqual(summary["max_shard_bytes"], 1000)
        self.assertEqual(summary["storage_imbalance"], 1.4)
        self.assertEqual(summary["max_edge_imbalance"], 1.5)
        self.assertEqual(
            summary["quality_by_threads"]["32"]
            ["remote_fraction"]["median"],
            0.65,
        )

    def test_frozen_evidence_paths_are_repository_relative(self):
        root = Path(__file__).resolve().parents[2]
        value = {
            "path": str(root / "results/graphs/demo.sg"),
            "outside": "/tmp/external.log",
        }
        normalized = normalize_paths(value)
        self.assertEqual(
            normalized["path"],
            "results/graphs/demo.sg",
        )
        self.assertEqual(
            normalized["outside"],
            "/tmp/external.log",
        )


if __name__ == "__main__":
    unittest.main()
