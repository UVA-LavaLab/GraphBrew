// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

// Standalone streaming exporter: reads an unweighted GAP serialized graph
// (`.sg`) through a read-only mmap view and writes a graph.shard.v1 package one
// shard at a time. Peak extra memory is O(N) scratch plus the single largest
// shard, never all shards at once, so very large graphs can be sharded without
// materialising a second in-memory CSR.
//
// The exporter preserves the vertex ordering already baked into the `.sg`
// file: `org_ids` is taken verbatim as the internal-to-source mapping, so the
// policy/mapping semantics recorded in the manifest reflect whatever reorder
// produced the `.sg`. This tool does not run a new reorder.

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

#include <getopt.h>

#include "partition/compact_csr.h"
#include "partition/diagnostics.h"
#include "partition/sg_mmap_view.h"
#include "partition/shard_manifest.h"

namespace
{

void PrintUsage(const char *program)
{
    std::cerr
        << "Usage: " << program << " -f <input.sg> -E <output_dir>"
        << " [-P <partitions>] [-B <balance>] [-i <graph_id>]"
        << " [-a <policy_name>] [-d <policy_id>] [-o <policy_option>]...\n"
        << "\n"
        << "  -f  input unweighted serialized graph (.sg)\n"
        << "  -E  output directory for the graph.shard.v1 package\n"
        << "  -P  number of partitions (default 1)\n"
        << "  -B  balance policy: vertices | out | total (default total)\n"
        << "  -i  graph id recorded in the manifest"
        << " (default: input file stem)\n"
        << "  -a  reorder policy name recorded in the manifest"
        << " (default: Original)\n"
        << "  -d  reorder policy id recorded in the manifest (default 0)\n"
        << "  -o  reorder policy option string (repeatable)\n";
}

} // namespace

int main(int argc, char *argv[])
{
    std::string input_path;
    std::string output_dir;
    std::size_t partitions = 1;
    std::string balance_name = "total";
    std::string graph_id;
    std::string policy_name;
    int policy_id = 0;
    std::vector<std::string> policy_options;

    int opt = 0;
    while ((opt = getopt(argc, argv, "f:E:P:B:i:a:d:o:h")) != -1)
    {
        switch (opt)
        {
        case 'f':
            input_path = optarg;
            break;
        case 'E':
            output_dir = optarg;
            break;
        case 'P':
            partitions = static_cast<std::size_t>(std::stoull(optarg));
            break;
        case 'B':
            balance_name = optarg;
            break;
        case 'i':
            graph_id = optarg;
            break;
        case 'a':
            policy_name = optarg;
            break;
        case 'd':
            policy_id = std::stoi(optarg);
            break;
        case 'o':
            policy_options.emplace_back(optarg);
            break;
        case 'h':
        default:
            PrintUsage(argv[0]);
            return opt == 'h' ? 0 : -1;
        }
    }

    if (input_path.empty() || output_dir.empty())
    {
        PrintUsage(argv[0]);
        return -1;
    }

    try
    {
        GraphPartitionBalance balance =
            ParseGraphPartitionBalance(balance_name);

        graphbrew::partition::SerializedGraphView<> view(input_path);

        const auto mapping =
            graphbrew::partition::BuildOriginalIdMapping(view);

        graphbrew::partition::ShardPackageMetadata metadata;
        metadata.graph_id = graph_id.empty()
            ? std::filesystem::path(input_path).stem().string()
            : graph_id;
        metadata.policy_name =
            policy_name.empty() ? std::string("Original") : policy_name;
        metadata.policy_id = policy_id;
        metadata.policy_options = std::move(policy_options);

        graphbrew::partition::ShardStreamStats stats;
        const std::filesystem::path manifest_path =
            graphbrew::partition::StreamShardPackage(
                output_dir,
                view,
                mapping,
                partitions,
                balance,
                metadata,
                &stats);

        std::cout
            << "Streamed graph.shard.v1 package: " << manifest_path << '\n'
            << "  nodes=" << view.num_nodes()
            << " edges_directed=" << view.num_edges_directed()
            << " directed=" << (view.directed() ? "true" : "false") << '\n'
            << "  partitions=" << partitions
            << " balance=" << balance_name << '\n'
            << "  peak_live_shards=" << stats.max_live_shards
            << " max_shard_bytes=" << stats.max_live_shard_bytes
            << " total_shard_bytes=" << stats.total_shard_bytes << '\n';
    }
    catch (const std::exception &error)
    {
        std::cerr << "graph_shard_export: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
