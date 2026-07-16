#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graphbrew/partition/ownership_analysis.h"
#include "graphbrew/reorder/reorder_graphbrew.h"
#include "nlohmann_json.hpp"

namespace
{

template <typename Node>
nlohmann::json MetricsJson(
    const graphbrew::partition::OwnershipMetrics<Node> &metrics)
{
    return {
        {"owner_fingerprint", metrics.owner_fingerprint},
        {"partition_count", metrics.partition_count},
        {"remote_out_fraction", metrics.remote_out_fraction},
        {"remote_in_fraction", metrics.remote_in_fraction},
        {"max_remote_out_fraction", metrics.max_remote_out_fraction},
        {"max_remote_in_fraction", metrics.max_remote_in_fraction},
        {"ghost_slots", metrics.total_ghost_slots},
        {"ghost_bytes", metrics.total_ghost_slots * 8},
        {"ownership_metadata_bytes",
         metrics.total_ownership_metadata_bytes},
        {"bfs_bytes_per_superstep", metrics.total_ghost_slots * 8},
        {"pr_bytes_per_iteration", metrics.total_ghost_slots * 4},
        {"cc_bytes_per_iteration", metrics.total_ghost_slots * 4},
        {"spmv_initial_bytes", metrics.total_ghost_slots * 4},
        {"compact_total_storage_lower_bound_bytes",
         metrics.total_storage_bytes},
        {"compact_max_storage_lower_bound_bytes",
         metrics.max_storage_bytes},
        {"vertex_imbalance", metrics.vertex_imbalance},
        {"out_edge_imbalance", metrics.out_edge_imbalance},
        {"in_edge_imbalance", metrics.in_edge_imbalance},
        {"balance_imbalance", metrics.balance_imbalance},
        {"compact_storage_lower_bound_imbalance",
         metrics.storage_imbalance},
        {"per_shard", {
            {"owned_vertices", metrics.owned_vertices},
            {"out_edges", metrics.out_edges},
            {"in_edges", metrics.in_edges},
            {"ghost_slots", metrics.ghost_slots},
            {"ownership_metadata_bytes",
             metrics.ownership_metadata_bytes},
            {"compact_storage_lower_bound_bytes",
             metrics.storage_bytes},
        }},
    };
}

} // namespace

int main(int argc, char **argv)
{
    CLPartitionApp cli(argc, argv, "partition ownership analysis");
    if (!cli.ParseArgs())
        return -1;
    Builder builder(cli);
    Graph graph = builder.MakeGraph();
    if (graph.num_nodes() == 0)
        throw std::invalid_argument(
            "Ownership analysis requires a non-empty graph");
    const GraphPartitionBalance balance =
        ParseGraphPartitionBalance(cli.partition_balance());
    const std::size_t effective_partitions = std::min(
        static_cast<std::size_t>(cli.num_partitions()),
        static_cast<std::size_t>(graph.num_nodes()));

    graphbrew::GraphBrewConfig config =
        graphbrew::parseGraphBrewConfig({
            "gvecsr",
            "totalm",
            "refine0",
            "compose",
            "comm_degree_desc",
            "intra_rcmpp",
        });
    config.resolution =
        LeidenAutoResolution<NodeID, NodeID>(graph);
    config.deterministicCommunityDetection = true;
    auto result = graphbrew::runGraphBrew<std::uint32_t>(
        graph, config);
    pvector<NodeID> new_ids(graph.num_nodes());
    graphbrew::applyOrderingStrategy<std::uint32_t>(
        graph, new_ids, result, config);

    const auto contiguous_owners =
        graphbrew::partition::BuildContiguousOwners<NodeID>(
            graph,
            new_ids,
            effective_partitions,
            balance);
    const auto community_owners =
        graphbrew::partition::BuildCommunityOwners<NodeID>(
            graph,
            result.membership,
            effective_partitions,
            balance);
    const auto contiguous =
        graphbrew::partition::EvaluateOwnership<NodeID>(
            graph, contiguous_owners, balance, effective_partitions);
    const auto noncontiguous =
        graphbrew::partition::EvaluateOwnership<NodeID>(
            graph,
            community_owners.first,
            balance,
            effective_partitions,
            true);

    graphbrew::partition::OrderedFingerprint membership_fingerprint;
    membership_fingerprint.AddRange(result.membership);
    graphbrew::partition::OrderedFingerprint mapping_fingerprint;
    mapping_fingerprint.AddRange(new_ids);
    nlohmann::json output = {
        {"schema", "graphbrew.partition_ownership_analysis.v1"},
        {"requested_partitions", cli.num_partitions()},
        {"partitions", effective_partitions},
        {"vertices", graph.num_nodes()},
        {"balance", GraphPartitionBalanceName(balance)},
        {"analysis_only", true},
        {"graph_shard_v1_compatible", false},
        {"complete_per_bank_working_set_evaluated", false},
        {"communities", result.numCommunities},
        {"membership_fingerprint", membership_fingerprint.Hex()},
        {"mapping_fingerprint", mapping_fingerprint.Hex()},
        {"assignment_meets_work_balance_gate", community_owners.second},
        {"contiguous", MetricsJson(contiguous)},
        {"owner_by_vertex", MetricsJson(noncontiguous)},
    };
    std::cout << "[OWNERSHIP_ANALYSIS] "
              << output.dump() << std::endl;
    return 0;
}
