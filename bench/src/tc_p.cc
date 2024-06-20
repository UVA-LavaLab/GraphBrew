// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

// Encourage use of gcc's parallel algorithms (for sort for relabeling)
// #ifdef _OPENMP
//   #define _GLIBCXX_PARALLEL
// #endif

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"

/*
   GAP Benchmark Suite
   Kernel: Triangle Counting (TC)
   Author: Scott Beamer

   Will count the number of triangles (cliques of size 3)

   Input graph requirements:
   - undirected
   - has no duplicate edges (or else will be counted as multiple triangles)
   - neighborhoods are sorted by vertex identifiers

   Other than symmetrizing, the rest of the requirements are done by SquishCSR
   during graph building.

   This implementation reduces the search space by counting each triangle only
   once. A naive implementation will count the same triangle six times because
   each of the three vertices (u, v, w) will count it in both ways. To count
   a triangle only once, this implementation only counts a triangle if u > v >
   w. Once the remaining unexamined neighbors identifiers get too big, it can
   break out of the loop, but this requires that the neighbors are sorted.

   This implementation relabels the vertices by degree. This optimization is
   beneficial if the average degree is sufficiently high and if the degree
   distribution is sufficiently non-uniform. To decide whether to relabel the
   graph, we use the heuristic in WorthRelabelling.
 */

using namespace std;

size_t OrderedCount(const Graph &g)
{
    size_t total = 0;
    #pragma omp parallel for reduction(+ : total) schedule(dynamic, 64)
    for (NodeID u = 0; u < g.num_nodes(); u++)
    {
        for (NodeID v : g.out_neigh(u))
        {
            if (v > u)
                break;
            auto it = g.out_neigh(v).begin();
            for (NodeID w : g.out_neigh(u))
            {
                if (w > v)
                    break;
                while (*it < w)
                    it++;
                if (w == *it)
                    total++;
            }
        }
    }
    return total;
}

size_t CrossOrderedCount(const Graph &g, const Graph &g2)
{
    size_t total = 0;
    #pragma omp parallel for reduction(+ : total) schedule(dynamic, 64)
    for (NodeID u = 0; u < g.num_nodes(); u++)
    {
        for (NodeID v : g.out_neigh(u))
        {
            if (v > u)
                break;
            auto it = g2.out_neigh(v).begin();
            for (NodeID w : g.out_neigh(u))
            {
                if (w > v)
                    break;
                while (*it < w)
                    it++;
                if (w == *it)
                    total++;
            }
        }
    }
    return total;
}

// Heuristic to see if sufficiently dense power-law graph
bool WorthRelabelling(const Graph &g)
{
    int64_t average_degree = g.num_edges() / g.num_nodes();
    if (average_degree < 10)
        return false;
    SourcePicker<Graph> sp(g);
    int64_t num_samples = min(int64_t(1000), g.num_nodes());
    int64_t sample_total = 0;
    pvector<int64_t> samples(num_samples);
    for (int64_t trial = 0; trial < num_samples; trial++)
    {
        samples[trial] = g.out_degree(sp.PickNext());
        sample_total += samples[trial];
    }
    sort(samples.begin(), samples.end());
    double sample_average = static_cast<double>(sample_total) / num_samples;
    double sample_median = samples[num_samples / 2];
    return sample_average / 1.3 > sample_median;
}

// Uses heuristic to see if worth relabeling
size_t Hybrid(const Graph &g)
{
    if (WorthRelabelling(g))
        return OrderedCount(Builder::RelabelByDegree(g));
    else
        return OrderedCount(g);
}

// // Uses heuristic to see if worth relabeling
// size_t Hybrid_partitioned(const PGraph &p_g,
//                           int p_n = 1, int p_m = 1) {
//   size_t total = 0;
//   size_t p_total = 0;
//   for (int col = 0; col < p_m; ++col) {
//     for (int row = 0; row < p_n; ++row) {
//       int idx = col * p_n + row;
//       Graph partition_g = p_g[idx];
//       if (WorthRelabelling(partition_g))
//         p_total = OrderedCount(Builder::RelabelByDegree(partition_g));
//       else
//         p_total = OrderedCount(partition_g);

//       total += p_total;
//     }
//   }

//   return total;
// }

void PrintTriangleStats(const Graph &g, size_t total_triangles)
{
    cout << total_triangles << " triangles" << endl;
}

// Compares with simple serial implementation that uses std::set_intersection
bool TCVerifier(const Graph &g, size_t test_total)
{
    size_t total = 0;
    vector<NodeID> intersection;
    intersection.reserve(g.num_nodes());
    for (NodeID u : g.vertices())
    {
        for (NodeID v : g.out_neigh(u))
        {
            auto new_end = set_intersection(
                               g.out_neigh(u).begin(), g.out_neigh(u).end(), g.out_neigh(v).begin(),
                               g.out_neigh(v).end(), intersection.begin());
            intersection.resize(new_end - intersection.begin());
            total += intersection.size();
        }
    }
    total = total / 6; // each triangle was counted 6 times
    if (total != test_total)
        cout << total << " != " << test_total << endl;
    return total == test_total;
}

int main(int argc, char *argv[])
{
    CLApp cli(argc, argv, "triangle count");
    if (!cli.ParseArgs())
        return -1;
    Builder b(cli);
    Graph g = b.MakeGraph();
    // if (g.directed()) {
    //   cout << "Input graph is directed but tc requires undirected" << endl;
    //   return -2;
    // }




    // g.PrintTopology();
    // b.PrintPartitionsTopology(p_g);

    // Create graphs from each partition in column-major order and add to
    // partitions_g
    std::vector<int>::const_iterator segment_iter = cli.segments().begin();
    // int p_type = *segment_iter;
    segment_iter++;
    int p_n = *segment_iter;
    segment_iter++;
    int p_m = *segment_iter;
    segment_iter++;

    PGraph p_g = b.MakePartitionedGraph();
    PFlatGraph pf_g;

    Timer tm;
    double tc_p_time = 0.0f;

    size_t total = 0;
    size_t p_total = 0;

    // local count
    for (int col = 0; col < p_m; ++col)
    {
        for (int row = 0; row < p_n; ++row)
        {
            int idx = row * p_m + col;
            std::cout << "Local TC_P: [" << row << "] [" << col << "]" << std::endl;
            // p_g[idx].PrintTopology();
            p_g[idx].PrintStats();
            tm.Start();
            p_total = OrderedCount(p_g[idx]);
            tm.Stop();
            tc_p_time += tm.Seconds();
            total += p_total;
        }
    }
    PrintTime("Local Time TC_P", tc_p_time);
    std::cout << "Local TC_P: " << total << std::endl;
    // cross count
    // Cross count for each column
    for (int col = 0; col < p_m; ++col)
    {
        for (int row1 = 0; row1 < p_n; ++row1)
        {
            int idx1 = row1 * p_m + col;
            for (int row2 = row1 + 1; row2 < p_n; ++row2)
            {
                int idx2 = row2 * p_m + col;
                std::cout << "Cross TC_P: [" << row1 << "] [" << col << "] with ["
                          << row2 << "] [" << col << "]" << std::endl;
                tm.Start();
                p_total = CrossOrderedCount(p_g[idx1], p_g[idx2]);
                tm.Stop();
                tc_p_time += tm.Seconds();
                total += p_total;
            }
        }
    }

    b.FlattenPartitions(p_g, pf_g);

    for (int col = 0; col < p_m; ++col)
    {
        for (int row = 0; row < p_n; ++row)
        {
            int idx = row * p_m + col;
            std::cout << "Local TC_P: [" << row << "] [" << col << "]" << std::endl;
            pf_g[idx].PrintStats();
            pf_g[idx].PrintTopology();
            pf_g[idx].display(std::string("graph"));
        }
    }

    PrintTime("Total Time TC_P", tc_p_time);
    std::cout << "Total TC_P: " << total << std::endl;

    BenchmarkKernel(cli, g, Hybrid, PrintTriangleStats, TCVerifier);
    return 0;
}
