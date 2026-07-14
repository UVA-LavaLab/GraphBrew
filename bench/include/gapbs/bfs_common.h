#ifndef BFS_COMMON_H_
#define BFS_COMMON_H_

#include <cstdint>
#include <iostream>
#include <vector>

#include "benchmark.h"
#include "pvector.h"

inline void PrintBFSStats(
    const Graph &graph,
    const pvector<NodeID> &bfs_tree)
{
    std::int64_t tree_size = 0;
    std::int64_t edge_count = 0;
    for (NodeID vertex : graph.vertices())
    {
        if (bfs_tree[vertex] >= 0)
        {
            edge_count += graph.out_degree(vertex);
            ++tree_size;
        }
    }
    std::cout
        << "BFS Tree has " << static_cast<long long>(tree_size)
        << " nodes and " << static_cast<long long>(edge_count)
        << " edges" << std::endl;
}

inline bool BFSVerifier(
    const Graph &graph,
    NodeID source,
    const pvector<NodeID> &parent)
{
    pvector<int> depth(graph.num_nodes(), -1);
    depth[source] = 0;
    std::vector<NodeID> to_visit;
    to_visit.reserve(graph.num_nodes());
    to_visit.push_back(source);
    for (auto it = to_visit.begin(); it != to_visit.end(); ++it)
    {
        const NodeID vertex = *it;
        for (NodeID neighbor : graph.out_neigh(vertex))
        {
            if (depth[neighbor] == -1)
            {
                depth[neighbor] = depth[vertex] + 1;
                to_visit.push_back(neighbor);
            }
        }
    }

    for (NodeID vertex : graph.vertices())
    {
        if (depth[vertex] != -1 && parent[vertex] != -1)
        {
            if (vertex == source)
            {
                if (parent[vertex] != vertex || depth[vertex] != 0)
                {
                    std::cout << "Source wrong" << std::endl;
                    return false;
                }
                continue;
            }

            bool parent_found = false;
            for (NodeID predecessor : graph.in_neigh(vertex))
            {
                if (predecessor != parent[vertex])
                    continue;
                if (depth[predecessor] != depth[vertex] - 1)
                {
                    std::cout << "Wrong depths for " << vertex
                              << " & " << predecessor << std::endl;
                    return false;
                }
                parent_found = true;
                break;
            }
            if (!parent_found)
            {
                std::cout << "Couldn't find edge from "
                          << parent[vertex] << " to " << vertex
                          << std::endl;
                return false;
            }
        }
        else if (depth[vertex] != parent[vertex])
        {
            std::cout << "Reachability mismatch" << std::endl;
            return false;
        }
    }
    return true;
}

#endif // BFS_COMMON_H_
