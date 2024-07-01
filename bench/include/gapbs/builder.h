// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef BUILDER_H_
#define BUILDER_H_

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <fstream>
#include <functional>
#include <numeric>
#include <parallel/algorithm>
#include <set>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "command_line.h"
#include "generator.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "reader.h"
#include "sliding_queue.h"
#include "timer.h"
#include "util.h"

/*
   GAP Benchmark Suite
   Class:  BuilderBase
   Author: Scott Beamer

   Given arguments from the command line (cli), returns a built graph
   - MakeGraph() will parse cli and obtain edgelist to call
   MakeGraphFromEL(edgelist) to perform the actual graph construction
   - edgelist can be from file (Reader) or synthetically generated (Generator)
   - Common case: BuilderBase typedef'd (w/ params) to be Builder (benchmark.h)
 */

//
// A demo program of reordering using Rabbit Order.

#ifdef RABBIT_ENABLE
#include "edge_list.hpp"
#include "rabbit_order.hpp"
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/count.hpp>

using namespace edge_list;
#endif
// Author: ARAI Junya <arai.junya@lab.ntt.co.jp> <araijn@gmail.com>
//

/*
   MIT License

   Copyright (c) 2016, Hao Wei.

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.
 */
#include "GoGraph.h"
#include "GoUtil.h"
/*
 * @author Priyank Faldu <Priyank.Faldu@ed.ac.uk> <http://faldupriyank.com>
 *
 * Copyright 2019 The University of Edinburgh
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */
#include "vec2d.h"

#ifndef TYPE
/** Type of edge weights. */
#define TYPE float
#endif
#ifndef REPEAT_METHOD
/** Number of times to repeat each method. */
#define REPEAT_METHOD 1
#endif
#ifndef MAX_THREADS
/** Maximum number of threads to use. */
#define MAX_THREADS 16
#endif

#include "main.hxx"

template <typename NodeID_, typename DestID_ = NodeID_,
          typename WeightT_ = NodeID_, bool invert = true,
          typename FNodeID_ = NodeID_, typename FDestID_ = NodeID_>
class BuilderBase
{
    typedef EdgePair<NodeID_, DestID_> Edge;
    typedef pvector<Edge> EdgeList;

    const CLBase &cli_;
    bool symmetrize_;
    bool needs_weights_;
    bool in_place_ = false;
    int64_t num_nodes_ = -1;
    std::vector<std::pair<ReorderingAlgo, std::string>> reorder_options_;

public:
    explicit BuilderBase(const CLBase &cli) : cli_(cli)
    {

        symmetrize_ = cli_.symmetrize();
        needs_weights_ = !std::is_same<NodeID_, DestID_>::value;
        in_place_ = cli_.in_place();
        // reorder_options_(cli_.reorder_options());
        if (in_place_ && needs_weights_)
        {
            std::cout << "In-place building (-m) does not support weighted graphs"
                      << std::endl;
            exit(-30);
        }
    }

    DestID_ GetSource(EdgePair<NodeID_, NodeID_> e)
    {
        return e.u;
    }

    DestID_ GetSource(EdgePair<NodeID_, NodeWeight<NodeID_, WeightT_>> e)
    {
        return NodeWeight<NodeID_, WeightT_>(e.u, e.v.w);
    }

    NodeID_ FindMaxNodeID(const EdgeList &el)
    {
        NodeID_ max_seen = 0;
        #pragma omp parallel for reduction(max : max_seen)
        for (auto it = el.begin(); it < el.end(); it++)
        {
            Edge e = *it;
            max_seen = __gnu_parallel::max(max_seen, e.u);
            max_seen = __gnu_parallel::max(max_seen, (NodeID_)e.v);
        }
        return max_seen;
    }

    pvector<NodeID_> CountDegrees(const EdgeList &el, bool transpose)
    {
        pvector<NodeID_> degrees(num_nodes_, 0);
        #pragma omp parallel for
        for (auto it = el.begin(); it < el.end(); it++)
        {
            Edge e = *it;
            if (symmetrize_ || (!symmetrize_ && !transpose))
                fetch_and_add(degrees[e.u], 1);
            if ((symmetrize_ && !in_place_) || (!symmetrize_ && transpose))
                fetch_and_add(degrees[(NodeID_)e.v], 1);
        }
        return degrees;
    }

    static pvector<SGOffset> PrefixSum(const pvector<NodeID_> &degrees)
    {
        pvector<SGOffset> sums(degrees.size() + 1);
        SGOffset total = 0;
        for (size_t n = 0; n < degrees.size(); n++)
        {
            sums[n] = total;
            total += degrees[n];
        }
        sums[degrees.size()] = total;
        return sums;
    }

    static pvector<SGOffset> ParallelPrefixSum(const pvector<NodeID_> &degrees)
    {
        const size_t block_size = 1 << 20;
        const size_t num_blocks = (degrees.size() + block_size - 1) / block_size;
        pvector<SGOffset> local_sums(num_blocks);
        #pragma omp parallel for
        for (size_t block = 0; block < num_blocks; block++)
        {
            SGOffset lsum = 0;
            size_t block_end = std::min((block + 1) * block_size, degrees.size());
            for (size_t i = block * block_size; i < block_end; i++)
                lsum += degrees[i];
            local_sums[block] = lsum;
        }
        pvector<SGOffset> bulk_prefix(num_blocks + 1);
        SGOffset total = 0;
        for (size_t block = 0; block < num_blocks; block++)
        {
            bulk_prefix[block] = total;
            total += local_sums[block];
        }
        bulk_prefix[num_blocks] = total;
        pvector<SGOffset> prefix(degrees.size() + 1);
        #pragma omp parallel for
        for (size_t block = 0; block < num_blocks; block++)
        {
            SGOffset local_total = bulk_prefix[block];
            size_t block_end = std::min((block + 1) * block_size, degrees.size());
            for (size_t i = block * block_size; i < block_end; i++)
            {
                prefix[i] = local_total;
                local_total += degrees[i];
            }
        }
        prefix[degrees.size()] = bulk_prefix[num_blocks];
        return prefix;
    }

    // Removes self-loops and redundant edges
    // Side effect: neighbor IDs will be sorted
    void SquishCSR(const CSRGraph<NodeID_, DestID_, invert> &g, bool transpose,
                   DestID_ ***sq_index, DestID_ **sq_neighs)
    {
        pvector<NodeID_> diffs(g.num_nodes());
        DestID_ *n_start, *n_end;
        #pragma omp parallel for private(n_start, n_end)
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
        {
            if (transpose)
            {
                n_start = g.in_neigh(n).begin();
                n_end = g.in_neigh(n).end();
            }
            else
            {
                n_start = g.out_neigh(n).begin();
                n_end = g.out_neigh(n).end();
            }
            __gnu_parallel::stable_sort(n_start, n_end);
            DestID_ *new_end = std::unique(n_start, n_end);
            new_end = std::remove(n_start, new_end, n);
            diffs[n] = new_end - n_start;
        }
        pvector<SGOffset> sq_offsets = ParallelPrefixSum(diffs);
        *sq_neighs = new DestID_[sq_offsets[g.num_nodes()]];
        *sq_index = CSRGraph<NodeID_, DestID_>::GenIndex(sq_offsets, *sq_neighs);
        #pragma omp parallel for private(n_start)
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
        {
            if (transpose)
                n_start = g.in_neigh(n).begin();
            else
                n_start = g.out_neigh(n).begin();
            std::copy(n_start, n_start + diffs[n], (*sq_index)[n]);
        }
    }

    CSRGraph<NodeID_, DestID_, invert>
    SquishGraph(const CSRGraph<NodeID_, DestID_, invert> &g)
    {
        DestID_ **out_index, *out_neighs, **in_index, *in_neighs;
        SquishCSR(g, false, &out_index, &out_neighs);
        if (g.directed())
        {
            if (invert)
                SquishCSR(g, true, &in_index, &in_neighs);
            return CSRGraph<NodeID_, DestID_, invert>(
                       g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
        }
        else
        {
            return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index,
                    out_neighs);
        }
    }

    /*
       In-Place Graph Building Steps
       - sort edges and squish (remove self loops and redundant edges)
       - overwrite EdgeList's memory with outgoing neighbors
       - if graph not being symmetrized
        - finalize structures and make incoming structures if requested
       - if being symmetrized
        - search for needed inverses, make room for them, add them in place
     */
    void MakeCSRInPlace(EdgeList &el, DestID_ ***index, DestID_ **neighs,
                        DestID_ ***inv_index, DestID_ **inv_neighs)
    {
        // preprocess EdgeList - sort & squish in place
        __gnu_parallel::stable_sort(el.begin(), el.end());
        auto new_end = std::unique(el.begin(), el.end());
        el.resize(new_end - el.begin());
        auto self_loop = [](Edge e)
        {
            return e.u == e.v;
        };
        new_end = std::remove_if(el.begin(), el.end(), self_loop);
        el.resize(new_end - el.begin());
        // analyze EdgeList and repurpose it for outgoing edges
        pvector<NodeID_> degrees = CountDegrees(el, false);
        pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
        pvector<NodeID_> indegrees = CountDegrees(el, true);
        *neighs = reinterpret_cast<DestID_ *>(el.data());
        for (Edge e : el)
            (*neighs)[offsets[e.u]++] = e.v;
        size_t num_edges = el.size();
        el.leak();
        // revert offsets by shifting them down
        for (NodeID_ n = num_nodes_; n >= 0; n--)
            offsets[n] = n != 0 ? offsets[n - 1] : 0;
        if (!symmetrize_) // not going to symmetrize so no need to add edges
        {
            size_t new_size = num_edges * sizeof(DestID_);
            *neighs = static_cast<DestID_ *>(std::realloc(*neighs, new_size));
            *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
            if (invert) // create inv_neighs & inv_index for incoming edges
            {
                pvector<SGOffset> inoffsets = ParallelPrefixSum(indegrees);
                *inv_neighs = new DestID_[inoffsets[num_nodes_]];
                *inv_index =
                    CSRGraph<NodeID_, DestID_>::GenIndex(inoffsets, *inv_neighs);
                for (NodeID_ u = 0; u < num_nodes_; u++)
                {
                    for (DestID_ *it = (*index)[u]; it < (*index)[u + 1]; it++)
                    {
                        NodeID_ v = static_cast<NodeID_>(*it);
                        (*inv_neighs)[inoffsets[v]] = u;
                        inoffsets[v]++;
                    }
                }
            }
        }
        else   // symmetrize graph by adding missing inverse edges
        {
            // Step 1 - count number of needed inverses
            pvector<NodeID_> invs_needed(num_nodes_, 0);
            for (NodeID_ u = 0; u < num_nodes_; u++)
            {
                for (SGOffset i = offsets[u]; i < offsets[u + 1]; i++)
                {
                    DestID_ v = (*neighs)[i];
                    bool inv_found =
                        std::binary_search(*neighs + offsets[v], *neighs + offsets[v + 1],
                                           static_cast<DestID_>(u));
                    if (!inv_found)
                        invs_needed[v]++;
                }
            }
            // increase offsets to account for missing inverses, realloc neighs
            SGOffset total_missing_inv = 0;
            for (NodeID_ n = 0; n < num_nodes_; n++)
            {
                offsets[n] += total_missing_inv;
                total_missing_inv += invs_needed[n];
            }
            offsets[num_nodes_] += total_missing_inv;
            size_t newsize = (offsets[num_nodes_] * sizeof(DestID_));
            *neighs = static_cast<DestID_ *>(std::realloc(*neighs, newsize));
            if (*neighs == nullptr)
            {
                std::cout << "Call to realloc() failed" << std::endl;
                exit(-33);
            }
            // Step 2 - spread out existing neighs to make room for inverses
            //   copies backwards (overwrites) and inserts free space at starts
            SGOffset tail_index = offsets[num_nodes_] - 1;
            for (NodeID_ n = num_nodes_ - 1; n >= 0; n--)
            {
                SGOffset new_start = offsets[n] + invs_needed[n];
                for (SGOffset i = offsets[n + 1] - 1; i >= new_start; i--)
                {
                    (*neighs)[tail_index] = (*neighs)[i - total_missing_inv];
                    tail_index--;
                }
                total_missing_inv -= invs_needed[n];
                tail_index -= invs_needed[n];
            }
            // Step 3 - add missing inverse edges into free spaces from Step 2
            for (NodeID_ u = 0; u < num_nodes_; u++)
            {
                for (SGOffset i = offsets[u] + invs_needed[u]; i < offsets[u + 1];
                        i++)
                {
                    DestID_ v = (*neighs)[i];
                    bool inv_found = std::binary_search(
                                         *neighs + offsets[v] + invs_needed[v], *neighs + offsets[v + 1],
                                         static_cast<DestID_>(u));
                    if (!inv_found)
                    {
                        (*neighs)[offsets[v] + invs_needed[v] - 1] =
                            static_cast<DestID_>(u);
                        invs_needed[v]--;
                    }
                }
            }
            for (NodeID_ n = 0; n < num_nodes_; n++)
                __gnu_parallel::stable_sort(*neighs + offsets[n],
                                            *neighs + offsets[n + 1]);
            *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
        }
    }

    /*
       Graph Building Steps (for CSR):
       - Read edgelist once to determine vertex degrees (CountDegrees)
       - Determine vertex offsets by a prefix sum (ParallelPrefixSum)
       - Allocate storage and set points according to offsets (GenIndex)
       - Copy edges into storage
     */
    void MakeCSR(const EdgeList &el, bool transpose, DestID_ ***index,
                 DestID_ **neighs)
    {
        pvector<NodeID_> degrees = CountDegrees(el, transpose);
        pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
        *neighs = new DestID_[offsets[num_nodes_]];
        *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
        #pragma omp parallel for
        for (auto it = el.begin(); it < el.end(); it++)
        {
            Edge e = *it;
            if (symmetrize_ || (!symmetrize_ && !transpose))
                (*neighs)[fetch_and_add(offsets[e.u], 1)] = e.v;
            if (symmetrize_ || (!symmetrize_ && transpose))
                (*neighs)[fetch_and_add(offsets[static_cast<NodeID_>(e.v)], 1)] =
                    GetSource(e);
        }
    }

    CSRGraph<NodeID_, DestID_, invert> MakeGraphFromEL(EdgeList &el)
    {
        DestID_ **index = nullptr, **inv_index = nullptr;
        DestID_ *neighs = nullptr, *inv_neighs = nullptr;
        Timer t;
        t.Start();
        if (num_nodes_ == -1)
            num_nodes_ = FindMaxNodeID(el) + 1;
        if (needs_weights_)
            Generator<NodeID_, DestID_, WeightT_>::InsertWeights(el);
        if (in_place_)
        {
            MakeCSRInPlace(el, &index, &neighs, &inv_index, &inv_neighs);
        }
        else
        {
            MakeCSR(el, false, &index, &neighs);
            if (!symmetrize_ && invert)
            {
                MakeCSR(el, true, &inv_index, &inv_neighs);
            }
        }
        t.Stop();

        PrintTime("Build Time", t.Seconds());
        if (symmetrize_)
            return CSRGraph<NodeID_, DestID_, invert>(num_nodes_, index, neighs);
        else
            return CSRGraph<NodeID_, DestID_, invert>(num_nodes_, index, neighs,
                    inv_index, inv_neighs);
    }

    pvector<NodeID_> CountLocalDegrees(const EdgeList &el, bool transpose)
    {
        pvector<NodeID_> degrees(num_nodes_, 0);
        #pragma omp parallel for
        for (auto it = el.begin(); it < el.end(); it++)
        {
            Edge e = *it;
            if ((!transpose))
                fetch_and_add(degrees[e.u], 1);
            if ((transpose))
                fetch_and_add(degrees[(NodeID_)e.v], 1);
        }
        return degrees;
    }

    void MakeLocalCSR(const EdgeList &el, bool transpose, DestID_ ***index,
                      DestID_ **neighs)
    {
        pvector<NodeID_> degrees = CountLocalDegrees(el, transpose);
        pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
        *neighs = new DestID_[offsets[num_nodes_]];
        *index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, *neighs);
        #pragma omp parallel for
        for (auto it = el.begin(); it < el.end(); it++)
        {
            Edge e = *it;
            if ((!transpose))
                (*neighs)[fetch_and_add(offsets[e.u], 1)] = e.v;
            if ((transpose))
                (*neighs)[fetch_and_add(offsets[static_cast<NodeID_>(e.v)], 1)] =
                    GetSource(e);
        }
    }

    CSRGraph<NodeID_, DestID_, invert> MakeLocalGraphFromEL(EdgeList &el)
    {
        DestID_ **index = nullptr, **inv_index = nullptr;
        DestID_ *neighs = nullptr, *inv_neighs = nullptr;
        Timer t;
        t.Start();
        if (num_nodes_ == -1)
        {
            num_nodes_ = FindMaxNodeID(el) + 1;
        }

        MakeLocalCSR(el, false, &index, &neighs);
        MakeLocalCSR(el, true, &inv_index, &inv_neighs);
        CSRGraph<NodeID_, DestID_, invert> g = CSRGraph<NodeID_, DestID_, invert>(
                num_nodes_, index, neighs, inv_index, inv_neighs);
        // CSRGraph<NodeID_, DestID_, invert> g = CSRGraph<NodeID_, DestID_,
        // invert>(num_nodes_, index, neighs); g.PrintTopology();
        g = SquishGraph(g);
        // SquishCSR(g, false, &index, &neighs);
        // SquishCSR(g, true, &inv_index, &inv_neighs);
        t.Stop();
        PrintTime("Local Build Time", t.Seconds());
        return g;
        // return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
    }

    void FlattenPartitions(
        const std::vector<CSRGraph<NodeID_, DestID_, invert>> &partitions,
        std::vector<CSRGraphFlat<FNodeID_, WeightT_, FDestID_>> &partitions_flat,
        size_t alignment = 4096)
    {
        partitions_flat.reserve(
            partitions.size()); // Reserve space for the flattened partitions

        for (const auto &partition : partitions)
        {
            partitions_flat.push_back(partition.flattenGraphOut(alignment));
        }
    }

    void
    MakeOrientedELFromUniDirect(EdgeList &el,
                                const CSRGraph<NodeID_, DestID_, invert> &g)
    {
        int64_t num_edges = el.size();

        // Parallel loop to TRUST oriented edge list
        #pragma omp parallel for
        for (NodeID_ i = 0; i < num_edges; ++i)
        {
            Edge e = el[i];
            if (g.is_weighted())
            {
                NodeID_ src = e.u;
                NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(e.v).v;
                WeightT_ weight = static_cast<NodeWeight<NodeID_, WeightT_>>(e.v).w;

                if (src > dest)
                    el[i] = Edge(dest, NodeWeight<NodeID_, WeightT_>(src, weight));

            }
            else
            {
                NodeID_ src = e.u;
                NodeID_ dest = e.v;

                if (src > dest)
                    el[i] = Edge(dest, src);
            }
        }
    }

    EdgeList
    MakeUniDirectELFromGraph(const CSRGraph<NodeID_, DestID_, invert> &g)
    {
        int64_t num_edges = g.num_edges_directed();
        int64_t num_nodes = g.num_nodes();
        EdgeList el(num_edges * 2);
        el.resize(num_edges * 2);

        // Parallel loop to construct the edge list
        #pragma omp parallel for
        for (NodeID_ i = 0; i < num_nodes; ++i)
        {
            NodeID_ out_start = g.out_offset(i);
            NodeID_ in_start = out_start + num_edges;

            NodeID_ j = 0;
            for (DestID_ neighbor : g.out_neigh(i))
            {
                el[out_start + j] = Edge(i, neighbor);
                if (g.is_weighted())
                {
                    NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                    WeightT_ weight =
                        static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;
                    el[in_start + j] =
                        Edge(dest, NodeWeight<NodeID_, WeightT_>(i, weight));
                }
                else
                {
                    el[in_start + j] = Edge(neighbor, i);
                }

                ++j;
            }
        }

        return el;
    }

    void PrintEdgeList(const EdgeList &el)
    {
        for (const auto &edge : el)
        {
            std::cout << edge.u << " -> " << edge.v << std::endl;
        }
    }

    /**
     * @brief Partitions a graph based on input parameters and returns the
     * partitions.
     *
     * This function retrieves partitioning parameters from the CLI input, creates
     * a graph, and then partitions the graph using either the GRAPHIT/Cagra or
     * TRUST partitioning method based on the input type. The resulting partitions
     * are then returned.
     *
     * @return std::vector<CSRGraph<NodeID_, DestID_, invert>> The partitions of
     * the graph.
     */
    std::vector<CSRGraph<NodeID_, DestID_, invert>> MakePartitionedGraph()
    {
        std::vector<CSRGraph<NodeID_, DestID_, invert>> partitions;

        std::vector<int>::const_iterator segment_iter = cli_.segments().begin();
        int p_type = *segment_iter;
        segment_iter++;
        int p_n = *segment_iter;
        segment_iter++;
        int p_m = *segment_iter;
        segment_iter++;

        CSRGraph<NodeID_, DestID_, invert> g = MakeGraph();

        switch (p_type)
        {
        case 0: // <0:GRAPHIT/Cagra>
            partitions = MakeCagraPartitionedGraph(g, p_n, p_m);
            break;
        case 1: // <1:TRUST>
            partitions = MakeTrustPartitionedGraph(g, p_n, p_m);
            break;
        default:
            partitions = MakeCagraPartitionedGraph(g, p_n, p_m);
            break;
        }

        return partitions;
    }

    void PrintPartitionsTopology(
        const std::vector<CSRGraph<NodeID_, DestID_, invert>> &partitions)
    {
        for (size_t i = 0; i < partitions.size(); ++i)
        {
            std::cout << "Partition " << i << ":" << std::endl;
            partitions[i].PrintTopology();
        }
    }

    std::vector<CSRGraph<NodeID_, DestID_, invert>>
            MakeCagraPartitionedGraph(const CSRGraph<NodeID_, DestID_, invert> &g,
                                      int p_n = 1, int p_m = 1)
    {
        std::vector<CSRGraph<NodeID_, DestID_, invert>> partitions;
        int p_num = p_n * p_m;
        if (p_num <= 0)
        {
            throw std::invalid_argument(
                "Number of partitions must be greater than 0");
        }
        // Determine the number of nodes in each partition
        NodeID_ total_nodes = g.num_nodes();
        NodeID_ nodes_per_partition = total_nodes / p_num;
        NodeID_ remaining_nodes = total_nodes % p_num;

        NodeID_ startID = 0;
        for (int i = 0; i < p_num; ++i)
        {
            NodeID_ stopID = startID + nodes_per_partition;
            if (i < remaining_nodes)
            {
                stopID += 1; // Distribute remaining nodes
            }
            // Create a partition using graphSlicer
            CSRGraph<NodeID_, DestID_, invert> partition =
                graphSlicer(g, startID, stopID, cli_.use_out_degree());
            partitions.emplace_back(std::move(partition));
            startID = stopID;
        }

        return partitions;
    }

    std::pair<std::vector<EdgeList>, std::vector<pvector<NodeID_>>>
    MakeTrustPartitionedEL(const CSRGraph<NodeID_, DestID_, invert> &g,
                           int p_n = 1, int p_m = 1)
    {

        int num_partitions = p_n * p_m;
        int num_threads = omp_get_max_threads();
        std::vector<EdgeList> partitions_el(num_partitions);
        std::vector<EdgeList> partitions_el_dest(p_m);

        // Local edge lists for each thread
        std::vector<std::vector<EdgeList>> local_partitions_el(num_threads);
        std::vector<std::vector<EdgeList>> local_partitions_el_dest(num_threads);

        std::vector<CSRGraph<NodeID_, DestID_, invert>> partitions_g(p_m);
        std::vector<pvector<NodeID_>> partitions_new_ids(p_m);

        for (int thread_id = 0; thread_id < num_threads; ++thread_id)
        {
            local_partitions_el[thread_id].resize(num_partitions);
            local_partitions_el_dest[thread_id].resize(p_m);
        }

        #pragma omp parallel
        {

            int thread_id = omp_get_thread_num();

            // Each thread processes a portion of the nodes
            #pragma omp for schedule(static)
            for (NodeID_ i = 0; i < g.num_nodes(); ++i)
            {
                NodeID_ src = i;

                for (DestID_ j : g.out_neigh(i))
                {
                    if (g.is_weighted())
                    {
                        NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(j).v;
                        WeightT_ weight = static_cast<NodeWeight<NodeID_, WeightT_>>(j).w;
                        int partition_idx = (dest % p_m);
                        Edge e = Edge(src, NodeWeight<NodeID_, WeightT_>(dest, weight));
                        local_partitions_el_dest[thread_id][partition_idx].push_back(e);
                    }
                    else
                    {
                        NodeID_ dest = j;
                        int partition_idx = (dest % p_m);
                        Edge e = Edge(src, dest);
                        // std::cout << partition_idx << ": " <<  src << " -> " << dest <<
                        // std::endl;
                        local_partitions_el_dest[thread_id][partition_idx].push_back(e);
                    }
                }
            }
        }

        // Parallel merge of local partitions into the global partitions
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < p_m; ++i)
        {
            for (int t = 0; t < num_threads; ++t)
            {
                partitions_el_dest[i].reserve(partitions_el_dest[i].size() +
                                              local_partitions_el_dest[t][i].size());
            }
            for (int t = 0; t < num_threads; ++t)
            {
                partitions_el_dest[i].insert(partitions_el_dest[i].end(),
                                             local_partitions_el_dest[t][i].begin(),
                                             local_partitions_el_dest[t][i].end());
            }
        }

        // Create graphs from each column partition and generate new IDs
        for (int i = 0; i < p_m; ++i)
        {
            if (partitions_el_dest[i].empty())
            {
                partitions_g[i] = CSRGraph<NodeID_, DestID_, invert>();
                partitions_new_ids[i] = pvector<NodeID_>();
                continue;
            }
            CSRGraph<NodeID_, DestID_, invert> partition_g =
                MakeLocalGraphFromEL(partitions_el_dest[i]);
            pvector<NodeID_> new_ids(partition_g.num_nodes(), -1);
            // partition_g = SquishGraph(partition_g);
            GenerateSortMapping(partition_g, new_ids, true, false);
            partitions_new_ids[i] = std::move(new_ids);
            partitions_g[i] = std::move(partition_g);
        }

        for (int dest_p = 0; dest_p < p_m; ++dest_p)
        {
            CSRGraph<NodeID_, DestID_, invert> partition_g =
                std::move(partitions_g[dest_p]);
            // partition_g.PrintTopology();
            if (partition_g.num_nodes() == 0)
            {
                continue;
            }

            #pragma omp parallel
            {
                int thread_id = omp_get_thread_num();
                // Each thread processes a portion of the nodes
                #pragma omp for schedule(static)
                for (NodeID_ i = 0; i < partition_g.num_nodes(); ++i)
                {
                    NodeID_ src = i;
                    for (DestID_ j : partition_g.out_neigh(i))
                    {
                        if (partition_g.is_weighted())
                        {
                            NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(j).v;
                            WeightT_ weight = static_cast<NodeWeight<NodeID_, WeightT_>>(j).w;
                            int partition_idx = (src % p_n) * p_m + (dest % p_m);
                            Edge e = Edge(src, NodeWeight<NodeID_, WeightT_>(dest, weight));
                            local_partitions_el[thread_id][partition_idx].push_back(e);
                        }
                        else
                        {
                            NodeID_ dest = j;
                            int partition_idx = (src % p_n) * p_m + (dest % p_m);
                            Edge e = Edge(src, dest);
                            local_partitions_el[thread_id][partition_idx].push_back(e);
                        }
                    }
                }
            }
        }

        // #pragma omp parallel
        // {

        //     int thread_id = omp_get_thread_num();

        //     // Each thread processes a portion of the nodes
        //     #pragma omp for schedule(static)
        //     for (NodeID_ i = 0; i < g.num_nodes(); ++i)
        //     {
        //         NodeID_ src = i;

        //         for (DestID_ j : g.out_neigh(i))
        //         {
        //             if (g.is_weighted())
        //             {
        //                 NodeID_ dest = static_cast<NodeWeight<NodeID_,
        //                 WeightT_>>(j).v; WeightT_ weight =
        //                 static_cast<NodeWeight<NodeID_, WeightT_>>(j).w; int
        //                 partition_idx = (src % p_n) * p_m + (dest % p_m); Edge e =
        //                 Edge(src, NodeWeight<NodeID_, WeightT_>(dest, weight));
        //                 local_partitions_el[thread_id][partition_idx].push_back(e);
        //             }
        //             else
        //             {
        //                 NodeID_ dest = j;
        //                 int partition_idx = (src % p_n) * p_m + (dest % p_m);
        //                 Edge e = Edge(src, dest);
        //                 // std::cout << partition_idx << ": " <<  src << " -> " <<
        //                 dest << std::endl;
        //                 local_partitions_el[thread_id][partition_idx].push_back(e);
        //             }
        //         }
        //     }
        // }

        // Parallel merge of local partitions into the global partitions
        #pragma omp parallel for schedule(dynamic)
        for (int i = 0; i < num_partitions; ++i)
        {
            for (int t = 0; t < num_threads; ++t)
            {
                partitions_el[i].reserve(partitions_el[i].size() +
                                         local_partitions_el[t][i].size());
            }
            for (int t = 0; t < num_threads; ++t)
            {
                partitions_el[i].insert(partitions_el[i].end(),
                                        local_partitions_el[t][i].begin(),
                                        local_partitions_el[t][i].end());
            }
        }

        return std::make_pair(std::move(partitions_el),
                              std::move(partitions_new_ids));
    }

    std::vector<CSRGraph<NodeID_, DestID_, invert>>
            MakeTrustPartitionedGraph(const CSRGraph<NodeID_, DestID_, invert> &g,
                                      int p_n = 1, int p_m = 1)
    {

        if ((p_n * p_m) <= 0)
        {
            throw std::invalid_argument(
                "Number of partitions must be greater than 0");
        }

        std::vector<CSRGraph<NodeID_, DestID_, invert>> partitions_g(p_n * p_m);
        std::vector<EdgeList> partitions_el;
        std::vector<pvector<NodeID_>> partitions_new_ids;

        CSRGraph<NodeID_, DestID_, invert> uni_g;
        CSRGraph<NodeID_, DestID_, invert> org_g;
        EdgeList uni_el;

        pvector<NodeID_> new_ids_g(g.num_nodes(), -1);
        GenerateSortMapping(g, new_ids_g, true, true);
        org_g = RelabelByMapping(g, new_ids_g);

        uni_el = MakeUniDirectELFromGraph(org_g);
        // PrintEdgeList(uni_el);
        // std::cout << std::endl;
        MakeOrientedELFromUniDirect(uni_el, org_g);
        // PrintEdgeList(uni_el);
        // std::cout << std::endl;

        uni_g = MakeLocalGraphFromEL(uni_el);
        // uni_g = SquishGraph(uni_g);
        // uni_g.PrintTopology();
        // std::cout << std::endl;

        pvector<NodeID_> new_ids(uni_g.num_nodes(), -1);
        GenerateSortMapping(uni_g, new_ids, true, false);
        uni_g = RelabelByMapping(uni_g, new_ids);

        std::tie(partitions_el, partitions_new_ids) =
            MakeTrustPartitionedEL(uni_g, p_n, p_m);

        // Create graphs from each partition in column-major order and add to
        // partitions_g
        for (int row = 0; row < p_n; ++row)
        {
            for (int col = 0; col < p_m; ++col)
            {
                int idx = row * p_m + col;
                CSRGraph<NodeID_, DestID_, invert> partition_g =
                    MakeLocalGraphFromEL(partitions_el[idx]);
                // partition_g = RelabelByMapping(partition_g, partitions_new_ids[col]);
                // partition_g = SquishGraph(partition_g);
                partitions_g[idx] = std::move(partition_g);
            }
        }

        return partitions_g;
    }

    CSRGraph<NodeID_, DestID_, invert> MakeGraph()
    {
        CSRGraph<NodeID_, DestID_, invert> g;
        CSRGraph<NodeID_, DestID_, invert> g_final;
        bool gContinue_ = true; // Control variable to exit the scope
        {
            // extra scope to trigger earlier deletion of el (save memory)
            EdgeList el;
            if (cli_.filename() != "")
            {
                Reader<NodeID_, DestID_, WeightT_, invert> r(cli_.filename());
                if ((r.GetSuffix() == ".sg") || (r.GetSuffix() == ".wsg"))
                {
                    g_final = r.ReadSerializedGraph();
                    gContinue_ = false; // Control variable to exit the scope
                }
                else
                {
                    el = r.ReadFile(needs_weights_);
                }
            }
            else if (cli_.scale() != -1)
            {
                Generator<NodeID_, DestID_> gen(cli_.scale(), cli_.degree());
                el = gen.GenerateEL(cli_.uniform());
            }
            if (gContinue_)
            {
                g = MakeGraphFromEL(el);
            }
        }

        if (gContinue_)
        {
            if (in_place_)
                g_final = std::move(g);
            else
                g_final = SquishGraph(g);
        }
        // g_final.PrintTopology();
        pvector<NodeID_> new_ids(g_final.num_nodes(), -1);
        for (const auto &option : cli_.reorder_options())
        {
            new_ids.fill(-1);
            GenerateMapping(g_final, new_ids, option.first, cli_.use_out_degree(),
                            option.second);
            g_final = RelabelByMapping(g_final, new_ids);
        }

        // g_final = SquishGraph(g_final);
        // g_final.PrintTopology();
        // g_final.PrintTopologyOriginal();
        return g_final;
    }

    // Relabels (and rebuilds) graph by order of decreasing degree
    static CSRGraph<NodeID_, DestID_, invert>
    RelabelByDegree(const CSRGraph<NodeID_, DestID_, invert> &g)
    {
        if (g.directed())
        {
            std::cout << "Cannot relabel directed graph" << std::endl;
            std::exit(-11);
        }
        Timer t;
        t.Start();
        typedef std::pair<int64_t, NodeID_> degree_node_p;
        pvector<degree_node_p> degree_id_pairs(g.num_nodes());
        #pragma omp parallel for
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
            degree_id_pairs[n] = std::make_pair(g.out_degree(n), n);
        __gnu_parallel::stable_sort(degree_id_pairs.begin(), degree_id_pairs.end(),
                                    std::greater<degree_node_p>());
        pvector<NodeID_> degrees(g.num_nodes());
        pvector<NodeID_> new_ids(g.num_nodes());
        #pragma omp parallel for
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
        {
            degrees[n] = degree_id_pairs[n].first;
            new_ids[degree_id_pairs[n].second] = n;
        }
        pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
        DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
        DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
        #pragma omp parallel for
        for (NodeID_ u = 0; u < g.num_nodes(); u++)
        {
            for (NodeID_ v : g.out_neigh(u))
                neighs[offsets[new_ids[u]]++] = new_ids[v];
            std::sort(index[new_ids[u]], index[new_ids[u] + 1]);
        }
        t.Stop();
        PrintTime("Relabel", t.Seconds());
        return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
    }

    static CSRGraph<NodeID_, DestID_, invert>
    RelabelByMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                     pvector<NodeID_> &new_ids)
    {
        Timer t;
        t.Start();
        bool outDegree = true;
        // bool createOnlyDegList = true;
        CSRGraph<NodeID_, DestID_, invert> g_relabel;
        bool createBothCSRs = true;

        auto max_iter = __gnu_parallel::max_element(new_ids.begin(), new_ids.end());
        size_t max_id = *max_iter;

        #pragma omp parallel for
        for (NodeID_ v = 0; v < g.num_nodes(); ++v)
        {
            if (new_ids[v] == -1)
            {
                // Assigning new IDs starting from max_id atomically
                NodeID_ local_max = __sync_fetch_and_add(&max_id, 1);
                new_ids[v] = local_max + 1;
                // cerr << v << " " << new_ids[v] << " " << max_id << endl;
            }
        }

        if (g.directed() == true)
        {
            #pragma omp parallel for
            for (NodeID_ v = 0; v < g.num_nodes(); ++v)
                assert(new_ids[v] != -1);

            /* Step VI: generate degree to build a new graph */
            pvector<NodeID_> degrees(g.num_nodes());
            pvector<NodeID_> inv_degrees(g.num_nodes());
            if (outDegree == true)
            {
                #pragma omp parallel for
                for (NodeID_ n = 0; n < g.num_nodes(); n++)
                {
                    degrees[new_ids[n]] = g.out_degree(n);
                    inv_degrees[new_ids[n]] = g.in_degree(n);
                }
            }
            else
            {
                #pragma omp parallel for
                for (NodeID_ n = 0; n < g.num_nodes(); n++)
                {
                    degrees[new_ids[n]] = g.in_degree(n);
                    inv_degrees[new_ids[n]] = g.out_degree(n);
                }
            }

            /* Graph building phase */
            pvector<SGOffset> offsets = ParallelPrefixSum(inv_degrees);
            DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
            DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
            #pragma omp parallel for schedule(dynamic, 1024)
            for (NodeID_ u = 0; u < g.num_nodes(); u++)
            {
                if (outDegree == true)
                {
                    for (NodeID_ v : g.in_neigh(u))
                        neighs[offsets[new_ids[u]]++] = new_ids[v];
                }
                else
                {
                    for (NodeID_ v : g.out_neigh(u))
                        neighs[offsets[new_ids[u]]++] = new_ids[v];
                }
                std::sort(index[new_ids[u]],
                          index[new_ids[u] + 1]); // sort neighbors of each vertex
            }
            DestID_ *inv_neighs(nullptr);
            DestID_ **inv_index(nullptr);
            if (createBothCSRs == true)
            {
                // making the inverse list (in-degrees in this case)
                pvector<SGOffset> inv_offsets = ParallelPrefixSum(degrees);
                inv_neighs = new DestID_[inv_offsets[g.num_nodes()]];
                inv_index =
                    CSRGraph<NodeID_, DestID_>::GenIndex(inv_offsets, inv_neighs);
                if (createBothCSRs == true)
                {
                    #pragma omp parallel for schedule(dynamic, 1024)
                    for (NodeID_ u = 0; u < g.num_nodes(); u++)
                    {
                        if (outDegree == true)
                        {
                            for (NodeID_ v : g.out_neigh(u))
                                inv_neighs[inv_offsets[new_ids[u]]++] = new_ids[v];
                        }
                        else
                        {
                            for (NodeID_ v : g.in_neigh(u))
                                inv_neighs[inv_offsets[new_ids[u]]++] = new_ids[v];
                        }
                        std::sort(
                            inv_index[new_ids[u]],
                            inv_index[new_ids[u] + 1]); // sort neighbors of each vertex
                    }
                }
            }
            t.Stop();
            PrintTime("Relabel Map Time", t.Seconds());
            if (outDegree == true)
            {

                g_relabel = CSRGraph<NodeID_, DestID_, invert>(
                                g.num_nodes(), inv_index, inv_neighs, index, neighs);
            }
            else
            {
                g_relabel = CSRGraph<NodeID_, DestID_, invert>(
                                g.num_nodes(), index, neighs, inv_index, inv_neighs);
            }
        }
        else
        {
            /* Undirected graphs - no need to make separate lists for in and out
             * degree */

            #pragma omp parallel for
            for (NodeID_ v = 0; v < g.num_nodes(); ++v)
                assert(new_ids[v] != -1);

            /* Step VI: generate degree to build a new graph */
            pvector<NodeID_> degrees(g.num_nodes());
            #pragma omp parallel for
            for (NodeID_ n = 0; n < g.num_nodes(); n++)
            {
                degrees[new_ids[n]] = g.out_degree(n);
            }

            /* Graph building phase */
            pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
            DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
            DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
            #pragma omp parallel for schedule(dynamic, 1024)
            for (NodeID_ u = 0; u < g.num_nodes(); u++)
            {
                for (NodeID_ v : g.out_neigh(u))
                    neighs[offsets[new_ids[u]]++] = new_ids[v];
                std::sort(index[new_ids[u]], index[new_ids[u] + 1]);
            }
            t.Stop();
            PrintTime("Relabel Map Time", t.Seconds());
            g_relabel =
                CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
        }

        g_relabel.copy_org_ids(g.get_org_ids());
        g_relabel.update_org_ids(new_ids);
        return g_relabel;
    }

    static CSRGraph<NodeID_, DestID_, invert>
    RelabelByMapping_v2(const CSRGraph<NodeID_, DestID_, invert> &g,
                        pvector<NodeID_> &new_ids)
    {
        Timer t;
        DestID_ **out_index;
        DestID_ *out_neighs;
        DestID_ **in_index;
        DestID_ *in_neighs;
        CSRGraph<NodeID_, DestID_, invert> g_relabel;

        t.Start();
        pvector<NodeID_> out_degrees(g.num_nodes(), 0);

        auto max_iter = __gnu_parallel::max_element(new_ids.begin(), new_ids.end());
        size_t max_id = *max_iter;

        #pragma omp parallel for
        for (NodeID_ v = 0; v < g.num_nodes(); ++v)
        {
            if (new_ids[v] == -1)
            {
                // Assigning new IDs starting from max_id atomically
                NodeID_ local_max = __sync_fetch_and_add(&max_id, 1);
                new_ids[v] = local_max + 1;
            }
        }

        #pragma omp parallel for
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
        {
            out_degrees[new_ids[n]] = g.out_degree(n);
            // if(new_ids[n] > g.num_nodes())
            // cerr << new_ids[n] << endl;
        }
        pvector<SGOffset> out_offsets = ParallelPrefixSum(out_degrees);
        out_neighs = new DestID_[out_offsets[g.num_nodes()]];
        out_index = CSRGraph<NodeID_, DestID_>::GenIndex(out_offsets, out_neighs);
        #pragma omp parallel for
        for (NodeID_ u = 0; u < g.num_nodes(); u++)
        {
            for (NodeID_ v : g.out_neigh(u))
            {
                SGOffset out_offsets_local =
                    __sync_fetch_and_add(&(out_offsets[new_ids[u]]), 1);
                out_neighs[out_offsets_local] = new_ids[v];
            }
            std::sort(out_index[new_ids[u]], out_index[new_ids[u] + 1]);
        }

        if (g.directed())
        {
            pvector<NodeID_> in_degrees(g.num_nodes(), 0);
            #pragma omp parallel for
            for (NodeID_ n = 0; n < g.num_nodes(); n++)
            {
                in_degrees[new_ids[n]] = g.in_degree(n);
            }
            pvector<SGOffset> in_offsets = ParallelPrefixSum(in_degrees);
            in_neighs = new DestID_[in_offsets[g.num_nodes()]];
            in_index = CSRGraph<NodeID_, DestID_>::GenIndex(in_offsets, in_neighs);
            #pragma omp parallel for
            for (NodeID_ u = 0; u < g.num_nodes(); u++)
            {
                for (NodeID_ v : g.in_neigh(u))
                {
                    SGOffset in_offsets_local =
                        __sync_fetch_and_add(&(in_offsets[new_ids[u]]), 1);
                    in_neighs[in_offsets_local] = new_ids[v];
                }
                std::sort(in_index[new_ids[u]], in_index[new_ids[u] + 1]);
            }
            t.Stop();
            g_relabel = CSRGraph<NodeID_, DestID_, invert>(
                            g.num_nodes(), out_index, out_neighs, in_index, in_neighs);
        }
        else
        {
            t.Stop();
            g_relabel = CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), out_index,
                        out_neighs);
        }
        g_relabel.copy_org_ids(g.get_org_ids());
        g_relabel.update_org_ids(new_ids);
        PrintTime("Relabel Map Time", t.Seconds());
        return g_relabel;
    }

    const std::string ReorderingAlgoStr(ReorderingAlgo type)
    {
        switch (type)
        {
        case HubSort:
            return "HubSort";
        case DBG:
            return "DBG";
        case HubClusterDBG:
            return "HubClusterDBG";
        case HubSortDBG:
            return "HubSortDBG";
        case HubCluster:
            return "HubCluster";
        case Random:
            return "Random";
        case RabbitOrder:
            return "RabbitOrder";
        case GOrder:
            return "GOrder";
        case COrder:
            return "COrder";
        case RCMOrder:
            return "RCMOrder";
        case LeidenOrder:
            return "LeidenOrder";
        case GraphBrewOrder:
            return "GraphBrewOrder";
        case ORIGINAL:
            return "Original";
        case Sort:
            return "Sort";
        case MAP:
            return "MAP";
        default:
            std::cerr << "Unknown Reordering Algorithm type: " << type << std::endl;
            abort();
        }
    }

    void GenerateMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                         pvector<NodeID_> &new_ids,
                         ReorderingAlgo reordering_algo, bool useOutdeg,
                         std::vector<std::string> reordering_options)
    {
        switch (reordering_algo)
        {
        case HubSort:
            GenerateHubSortMapping(g, new_ids, useOutdeg);
            break;
        case Sort:
            GenerateSortMapping(g, new_ids, useOutdeg);
            break;
        case DBG:
            GenerateDBGMapping(g, new_ids, useOutdeg);
            break;
        case HubSortDBG:
            GenerateHubSortDBGMapping(g, new_ids, useOutdeg);
            break;
        case HubClusterDBG:
            GenerateHubClusterDBGMapping(g, new_ids, useOutdeg);
            break;
        case HubCluster:
            GenerateHubClusterMapping(g, new_ids, useOutdeg);
            break;
        case Random:
            GenerateRandomMapping(g, new_ids);
            // RandOrder(g, new_ids, false, false);
            break;
        case RabbitOrder:
            GenerateRabbitOrderMapping(g, new_ids);
            break;
        case GOrder:
            GenerateGOrderMapping(g, new_ids);
            break;
        case COrder:
            GenerateCOrderMapping(g, new_ids);
            break;
        case RCMOrder:
            GenerateRCMOrderMapping(g, new_ids);
            break;
        case LeidenOrder:
            GenerateLeidenMapping(g, new_ids, reordering_options);
            break;
        case GraphBrewOrder:
            GenerateGraphBrewMapping(g, new_ids, useOutdeg, reordering_options);
            break;
        case MAP:
            LoadMappingFromFile(g, new_ids, reordering_options);
            break;
        case ORIGINAL:
            GenerateOriginalMapping(g, new_ids);
            break;
        default:
            std::cout << "Unknown generateMapping type: " << reordering_algo
                      << std::endl;
            std::abort();
        }

        // std::cout << std::endl;
        // for (size_t i = 0; i < new_ids.size(); ++i)
        // {
        //     std::cout <<  i << "->" << new_ids[i] << " " << static_cast<size_t>(g.out_degree(i)) << std::endl;
        // }
        // std::cout << std::endl;
#ifdef _DEBUG
        VerifyMapping(g, new_ids);
        // exit(-1);
#endif
    }

    void
    GenerateMappingLocalEdgelist(EdgeList &el, pvector<NodeID_> &new_ids,
                                 ReorderingAlgo reordering_algo, bool useOutdeg,
                                 std::vector<std::string> reordering_options, int numLevels = 1, bool recursion = false)
    {

        CSRGraph<NodeID_, DestID_, invert> g = MakeLocalGraphFromEL(el);

        switch (reordering_algo)
        {
        case HubSort:
            GenerateHubSortMapping(g, new_ids, useOutdeg);
            break;
        case Sort:
            GenerateSortMapping(g, new_ids, useOutdeg);
            break;
        case DBG:
            GenerateDBGMapping(g, new_ids, useOutdeg);
            break;
        case HubSortDBG:
            GenerateHubSortDBGMapping(g, new_ids, useOutdeg);
            break;
        case HubClusterDBG:
            GenerateHubClusterDBGMapping(g, new_ids, useOutdeg);
            break;
        case HubCluster:
            GenerateHubClusterMapping(g, new_ids, useOutdeg);
            break;
        case Random:
            GenerateRandomMapping(g, new_ids);
            // RandOrder(g, new_ids, false, false);
            break;
        case RabbitOrder:
            GenerateRabbitOrderMapping(g, new_ids);
            break;
        case GOrder:
            GenerateGOrderMapping(g, new_ids);
            break;
        case COrder:
            GenerateCOrderMapping(g, new_ids);
            break;
        case RCMOrder:
            GenerateRCMOrderMapping(g, new_ids);
            break;
        case LeidenOrder:
            GenerateLeidenMapping(g, new_ids, reordering_options);
            break;
        case GraphBrewOrder:
            GenerateGraphBrewMapping(g, new_ids, useOutdeg, reordering_options, numLevels, recursion);
            break;
        case MAP:
            LoadMappingFromFile(g, new_ids, reordering_options);
            break;
        case ORIGINAL:
            GenerateOriginalMapping(g, new_ids);
            break;
        default:
            std::cout << "Unknown generateMapping type: " << reordering_algo
                      << std::endl;
            std::abort();
        }
#ifdef _DEBUG
        VerifyMapping(g, new_ids);
        // exit(-1);
#endif
    }

    ReorderingAlgo getReorderingAlgo(const char *arg)
    {
        int value = std::atoi(arg);
        switch (value)
        {
        case 0:
            return ORIGINAL;
        case 1:
            return Random;
        case 2:
            return Sort;
        case 3:
            return HubSort;
        case 4:
            return HubCluster;
        case 5:
            return DBG;
        case 6:
            return HubSortDBG;
        case 7:
            return HubClusterDBG;
        case 8:
            return RabbitOrder;
        case 9:
            return GOrder;
        case 10:
            return COrder;
        case 11:
            return RCMOrder;
        case 12:
            return LeidenOrder;
        case 13:
            return GraphBrewOrder;
        case 14:
            return MAP;
        default:
            std::cerr << "Invalid ReorderingAlgo value: " << value << std::endl;
            std::exit(EXIT_FAILURE);
        }
    }

    void VerifyMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                       const pvector<NodeID_> &new_ids)
    {
        NodeID_ *hist = alloc_align_4k<NodeID_>(g.num_nodes());
        int64_t num_nodes = g.num_nodes();

        #pragma omp parallel for
        for (long i = 0; i < num_nodes; i++)
        {
            hist[i] = new_ids[i];
        }

        __gnu_parallel::stable_sort(&hist[0], &hist[num_nodes]);

        NodeID_ count = 0;

        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; i++)
        {
            if (hist[i] != i)
            {
                __sync_fetch_and_add(&count, 1);
            }
        }

        if (count != 0)
        {
            std::cout << "Num of vertices did not match: " << count << std::endl;
            std::cout << "Mapping is invalid.!" << std::endl;
            std::abort();
        }
        else
        {
            std::cout << "Mapping is valid.!" << std::endl;
        }
        std::free(hist);
    }

    void printReorderingMethods(const std::string &filename, Timer t)
    {
        std::size_t last_slash = filename.rfind('/');
        std::string basename = filename.substr(last_slash + 1);
        std::size_t last_dot = basename.rfind('.');
        std::string stem = basename.substr(0, last_dot);

        std::vector<int> codes;
        std::istringstream iss(stem);
        std::string part;
        while (getline(iss, part, '_'))
        {
            int num;
            if (std::istringstream(part) >> num)
            {
                codes.push_back(num);
            }
        }

        // std::cout << "Reordering methods for file '" << filename
        //           << "':" << std::endl;
        for (int code : codes)
        {
            try
            {
                std::string algoStr =
                    ReorderingAlgoStr(static_cast<ReorderingAlgo>(code)) + " Map Time";
                PrintTime(algoStr, t.Seconds());
            }
            catch (...)
            {
                std::cerr << "Invalid code: " << code << std::endl;
            }
        }
    }

    void LoadMappingFromFile(const CSRGraph<NodeID_, DestID_, invert> &g,
                             pvector<NodeID_> &new_ids,
                             std::vector<std::string> reordering_options)
    {
        Timer t;
        int64_t num_nodes = g.num_nodes();
        std::string map_file = "mapping.lo";

        // std::cout << "Options: ";
        // for (const auto& param : reordering_options) {
        //   std::cout << param << " ";
        // }
        // std::cout << std::endl;

        if (!reordering_options.empty())
            map_file = reordering_options[0];

        t.Start();
        std::ifstream ifs(map_file, std::ifstream::in);
        if (!ifs.is_open())
        {
            std::cerr << "File " << map_file << " does not exist!" << std::endl;
            throw std::runtime_error("File not found.");
        }
        std::string file_suffix = map_file.substr(map_file.find_last_of('.'));
        if (file_suffix != ".so" && file_suffix != ".lo")
        {
            std::cerr << "Unsupported file format: " << file_suffix << std::endl;
            throw std::invalid_argument("Unsupported format.");
        }
        NodeID_ *label_ids = new NodeID_[num_nodes];
        if (file_suffix == ".so")
        {
            ifs.read(reinterpret_cast<char *>(label_ids),
                     g.num_nodes() * sizeof(NodeID_));
            #pragma omp parallel for
            for (int64_t i = 0; i < num_nodes; i++)
            {
                new_ids[i] = label_ids[i];
            }
        }
        else
        {
            for (int64_t i = 0; i < num_nodes; i++)
            {
                ifs >> new_ids[i];
            }
        }
        delete[] label_ids;
        ifs.close();
        t.Stop();

        printReorderingMethods(map_file, t);
        PrintTime("Load Map Time", t.Seconds());
    }

    void GenerateOriginalMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                 pvector<NodeID_> &new_ids)
    {
        int64_t num_nodes = g.num_nodes();

        Timer t;
        t.Start();
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; i++)
        {
            new_ids[i] = (NodeID_)i;
        }
        t.Stop();
        PrintTime("Original Map Time", t.Seconds());
    }

    void GenerateRandomMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                               pvector<NodeID_> &new_ids)
    {
        Timer t;
        t.Start();
        std::srand(0); // so that the random graph generated is the same
        int64_t num_nodes = g.num_nodes();
        // int64_t num_edges = g.num_edges_directed();

        NodeID_ granularity = 1;
        NodeID_ slice = (num_nodes - granularity + 1) / granularity;
        NodeID_ artificial_num_nodes = slice * granularity;
        assert(artificial_num_nodes <= num_nodes);
        pvector<NodeID_> slice_index;
        slice_index.resize(slice);

        #pragma omp parallel for
        for (NodeID_ i = 0; i < slice; i++)
        {
            slice_index[i] = i;
        }

        __gnu_parallel::random_shuffle(slice_index.begin(), slice_index.end());

        {
            #pragma omp parallel for
            for (NodeID_ i = 0; i < slice; i++)
            {
                NodeID_ new_index = slice_index[i] * granularity;
                for (NodeID_ j = 0; j < granularity; j++)
                {
                    NodeID_ v = (i * granularity) + j;
                    if (v < artificial_num_nodes)
                    {
                        new_ids[v] = new_index + j;
                    }
                }
            }
        }

        for (NodeID_ i = artificial_num_nodes; i < num_nodes; i++)
        {
            new_ids[i] = i;
        }
        slice_index.clear();

        t.Stop();
        PrintTime("Random Map Time", t.Seconds());
    }

    void GenerateRandomMapping_v2(const CSRGraph<NodeID_, DestID_, invert> &g,
                                  pvector<NodeID_> &new_ids)
    {
        Timer t;
        t.Start();
        std::srand(0); // so that the random graph generated is the same
        // everytime

        // Step I: create a random permutation - SLOW implementation
        pvector<NodeID_> claimedVtxs(g.num_nodes(), 0);

        #pragma omp parallel for
        for (NodeID_ v = 0; v < g.num_nodes(); ++v)
        {
            while (true)
            {
                NodeID_ randID = std::rand() % g.num_nodes();
                if (claimedVtxs[randID] != 1)
                {
                    if (compare_and_swap(claimedVtxs[randID], 0, 1) == true)
                    {
                        new_ids[v] = randID;
                        break;
                    }
                    else
                        continue;
                }
            }
        }

        #pragma omp parallel for
        for (NodeID_ v = 0; v < g.num_nodes(); ++v)
            assert(new_ids[v] != -1);

        t.Stop();
        PrintTime("Random Map Time", t.Seconds());
    }

    void GenerateHubSortDBGMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                   pvector<NodeID_> &new_ids, bool useOutdeg)
    {

        typedef std::pair<int64_t, NodeID_> degree_nodeid_t;

        Timer t;
        t.Start();

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges();

        int64_t avgDegree = num_edges / num_nodes;
        size_t hubCount{0};

        const int num_threads = omp_get_max_threads();
        pvector<degree_nodeid_t> local_degree_id_pairs[num_threads];
        int64_t slice = num_nodes / num_threads;
        int64_t start[num_threads];
        int64_t end[num_threads];
        int64_t hub_count[num_threads];
        int64_t non_hub_count[num_threads];
        int64_t new_index[num_threads];
        for (int t = 0; t < num_threads; t++)
        {
            start[t] = t * slice;
            end[t] = (t + 1) * slice;
            hub_count[t] = 0;
        }
        end[num_threads - 1] = num_nodes;

        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int64_t t = 0; t < num_threads; t++)
        {
            for (int64_t v = start[t]; v < end[t]; ++v)
            {
                if (useOutdeg)
                {
                    int64_t out_degree_v = g.out_degree(v);
                    if (out_degree_v > avgDegree)
                    {
                        local_degree_id_pairs[t].push_back(std::make_pair(out_degree_v, v));
                    }
                }
                else
                {
                    int64_t in_degree_v = g.in_degree(v);
                    if (in_degree_v > avgDegree)
                    {
                        local_degree_id_pairs[t].push_back(std::make_pair(in_degree_v, v));
                    }
                }
            }
        }
        for (int t = 0; t < num_threads; t++)
        {
            hub_count[t] = local_degree_id_pairs[t].size();
            hubCount += hub_count[t];
            non_hub_count[t] = end[t] - start[t] - hub_count[t];
        }
        new_index[0] = hubCount;
        for (int t = 1; t < num_threads; t++)
        {
            new_index[t] = new_index[t - 1] + non_hub_count[t - 1];
        }
        pvector<degree_nodeid_t> degree_id_pairs(hubCount);

        size_t k = 0;
        for (int i = 0; i < num_threads; i++)
        {
            for (size_t j = 0; j < local_degree_id_pairs[i].size(); j++)
            {
                degree_id_pairs[k++] = local_degree_id_pairs[i][j];
            }
            local_degree_id_pairs[i].clear();
        }
        assert(degree_id_pairs.size() == hubCount);
        assert(k == hubCount);

        __gnu_parallel::stable_sort(degree_id_pairs.begin(), degree_id_pairs.end(),
                                    std::greater<degree_nodeid_t>());

        #pragma omp parallel for
        for (size_t n = 0; n < hubCount; ++n)
        {
            new_ids[degree_id_pairs[n].second] = n;
        }
        pvector<degree_nodeid_t>().swap(degree_id_pairs);

        #pragma omp parallel for schedule(static) num_threads(num_threads)
        for (int t = 0; t < num_threads; t++)
        {
            for (int64_t v = start[t]; v < end[t]; ++v)
            {
                if (new_ids[v] == (NodeID_)UINT_E_MAX)
                {
                    new_ids[v] = new_index[t]++;
                }
            }
        }

        t.Stop();
        PrintTime("HubSortDBG Map Time", t.Seconds());
    }

    void GenerateHubClusterDBGMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                      pvector<NodeID_> &new_ids, bool useOutdeg)
    {
        Timer t;
        t.Start();

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges();

        uint32_t avg_vertex = num_edges / num_nodes;

        const int num_buckets = 2;
        avg_vertex = avg_vertex;
        uint32_t bucket_threshold[] = {avg_vertex, static_cast<uint32_t>(-1)};

        vector<uint32_t> bucket_vertices[num_buckets];
        const int num_threads = omp_get_max_threads();
        vector<uint32_t> local_buckets[num_threads][num_buckets];

        if (useOutdeg)
        {
            // This loop relies on a static scheduling
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < num_nodes; i++)
            {
                for (unsigned int j = 0; j < num_buckets; j++)
                {
                    const int64_t &count = g.out_degree(i);
                    if (count <= bucket_threshold[j])
                    {
                        local_buckets[omp_get_thread_num()][j].push_back(i);
                        break;
                    }
                }
            }
        }
        else
        {
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < num_nodes; i++)
            {
                for (unsigned int j = 0; j < num_buckets; j++)
                {
                    const int64_t &count = g.in_degree(i);
                    if (count <= bucket_threshold[j])
                    {
                        local_buckets[omp_get_thread_num()][j].push_back(i);
                        break;
                    }
                }
            }
        }

        int temp_k = 0;
        uint32_t start_k[num_threads][num_buckets];
        for (int32_t j = num_buckets - 1; j >= 0; j--)
        {
            for (int t = 0; t < num_threads; t++)
            {
                start_k[t][j] = temp_k;
                temp_k += local_buckets[t][j].size();
            }
        }

        #pragma omp parallel for schedule(static)
        for (int t = 0; t < num_threads; t++)
        {
            for (int32_t j = num_buckets - 1; j >= 0; j--)
            {
                const vector<uint32_t> &current_bucket = local_buckets[t][j];
                int k = start_k[t][j];
                const size_t &size = current_bucket.size();
                for (uint32_t i = 0; i < size; i++)
                {
                    new_ids[current_bucket[i]] = k++;
                }
            }
        }

        for (int i = 0; i < num_threads; i++)
        {
            for (unsigned int j = 0; j < num_buckets; j++)
            {
                local_buckets[i][j].clear();
            }
        }

        t.Stop();
        PrintTime("HubClusterDBG Map Time", t.Seconds());
    }

    void GenerateHubSortMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                pvector<NodeID_> &new_ids, bool useOutdeg)
    {

        typedef std::pair<int64_t, NodeID_> degree_nodeid_t;

        Timer t;
        t.Start();

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges();

        pvector<degree_nodeid_t> degree_id_pairs(num_nodes);
        int64_t avgDegree = num_edges / num_nodes;
        size_t hubCount{0};

        /* STEP I - collect degrees of all vertices */
        #pragma omp parallel for reduction(+ : hubCount)
        for (int64_t v = 0; v < num_nodes; ++v)
        {
            if (useOutdeg)
            {
                int64_t out_degree_v = g.out_degree(v);
                degree_id_pairs[v] = std::make_pair(out_degree_v, v);
                if (out_degree_v > avgDegree)
                {
                    ++hubCount;
                }
            }
            else
            {
                int64_t in_degree_v = g.in_degree(v);
                degree_id_pairs[v] = std::make_pair(in_degree_v, v);
                if (in_degree_v > avgDegree)
                {
                    ++hubCount;
                }
            }
        }

        /* Step II - sort the degrees in parallel */
        __gnu_parallel::stable_sort(degree_id_pairs.begin(), degree_id_pairs.end(),
                                    std::greater<degree_nodeid_t>());

        /* Step III - make a remap based on the sorted degree list [Only for hubs]
         */
        #pragma omp parallel for
        for (size_t n = 0; n < hubCount; ++n)
        {
            new_ids[degree_id_pairs[n].second] = n;
        }
        // clearing space from degree pairs
        pvector<degree_nodeid_t>().swap(degree_id_pairs);

        /* Step IV - assigning a remap for (easy) non hub vertices */
        auto numHubs = hubCount;
        SlidingQueue<int64_t> queue(numHubs);
        #pragma omp parallel
        {
            QueueBuffer<int64_t> lqueue(queue, numHubs / omp_get_max_threads());
            #pragma omp for
            for (int64_t n = numHubs; n < num_nodes; ++n)
            {
                if (new_ids[n] == (NodeID_)UINT_E_MAX)
                {
                    // This steps preserves the ordering of the original graph (as much as
                    // possible)
                    new_ids[n] = n;
                }
                else
                {
                    int64_t remappedTo = new_ids[n];
                    if (new_ids[remappedTo] == (NodeID_)UINT_E_MAX)
                    {
                        // safe to swap Ids because the original vertex is a non-hub
                        new_ids[remappedTo] = n;
                    }
                    else
                    {
                        // Cannot swap ids because original vertex was a hub (swapping
                        // would disturb sorted ordering of hubs - not allowed)
                        lqueue.push_back(n);
                    }
                }
            }
            lqueue.flush();
        }
        queue.slide_window(); // the queue keeps a list of vertices where a simple
        // swap of locations is not possible
        /* Step V - assigning remaps for remaining non hubs */
        int64_t unassignedCtr{0};
        auto q_iter = queue.begin();
        #pragma omp parallel for
        for (size_t n = 0; n < numHubs; ++n)
        {
            if (new_ids[n] == (NodeID_)UINT_E_MAX)
            {
                int64_t u = *(q_iter + __sync_fetch_and_add(&unassignedCtr, 1));
                new_ids[n] = u;
            }
        }

        t.Stop();
        PrintTime("HubSort Map Time", t.Seconds());
    }

    void GenerateSortMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                             pvector<NodeID_> &new_ids, bool useOutdeg,
                             bool lesser = false)
    {

        typedef std::pair<int64_t, NodeID_> degree_nodeid_t;

        Timer t;
        t.Start();

        int64_t num_nodes = g.num_nodes();
        // int64_t num_edges = g.num_edges_directed();

        pvector<degree_nodeid_t> degree_id_pairs(num_nodes);

        if (useOutdeg)
        {
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v)
            {
                int64_t out_degree_v = g.out_degree(v);
                degree_id_pairs[v] = std::make_pair(out_degree_v, v);
            }
        }
        else
        {
            #pragma omp parallel for
            for (int64_t v = 0; v < num_nodes; ++v)
            {
                int64_t in_degree_v = g.in_degree(v);
                degree_id_pairs[v] = std::make_pair(in_degree_v, v);
            }
        }

        if (lesser)
        {
            __gnu_parallel::stable_sort(degree_id_pairs.begin(),
                                        degree_id_pairs.end(),
                                        std::less<degree_nodeid_t>());
        }
        else
        {
            __gnu_parallel::stable_sort(degree_id_pairs.begin(),
                                        degree_id_pairs.end(),
                                        std::greater<degree_nodeid_t>());
        }

        #pragma omp parallel for
        for (int64_t n = 0; n < num_nodes; ++n)
        {
            new_ids[degree_id_pairs[n].second] = n;
        }

        pvector<degree_nodeid_t>().swap(degree_id_pairs);

        t.Stop();
        PrintTime("Sort Map Time", t.Seconds());
    }

    void GenerateDBGMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                            pvector<NodeID_> &new_ids, bool useOutdeg)
    {
        Timer t;
        t.Start();

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges();

        uint32_t avg_vertex = num_edges / num_nodes;
        const uint32_t &av = avg_vertex;

        // uint32_t bucket_threshold[] = {
        //   av / 2,   av,       av * 2,   av * 4,
        //   av * 8,   av * 16,  av * 32,  av * 64,
        //   av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)};

        uint32_t bucket_threshold[] = {av / 2, av, av * 2, av * 4, av * 8, av * 16, av * 32, av * 64, av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)};
        int num_buckets = 8;
        if ( num_buckets > 11 )
        {
            // if you really want to increase the bucket count, add more thresholds to the bucket_threshold above.
            std::cout << "Unsupported bucket size: " << num_buckets << std::endl;
            assert(0);
        }
        bucket_threshold[num_buckets - 1] = static_cast<uint32_t>(-1);

        vector<uint32_t> bucket_vertices[num_buckets];
        const int num_threads = omp_get_max_threads();
        vector<uint32_t> local_buckets[num_threads][num_buckets];

        if (useOutdeg)
        {
            // This loop relies on a static scheduling
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < num_nodes; i++)
            {
                for (int j = 0; j < num_buckets; j++)
                {
                    const int64_t &count = g.out_degree(i);
                    if (count <= bucket_threshold[j])
                    {
                        local_buckets[omp_get_thread_num()][j].push_back(i);
                        break;
                    }
                }
            }
        }
        else
        {
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < num_nodes; i++)
            {
                for (int j = 0; j < num_buckets; j++)
                {
                    const int64_t &count = g.in_degree(i);
                    if (count <= bucket_threshold[j])
                    {
                        local_buckets[omp_get_thread_num()][j].push_back(i);
                        break;
                    }
                }
            }
        }

        int temp_k = 0;
        uint32_t start_k[num_threads][num_buckets];
        for (int32_t j = num_buckets - 1; j >= 0; j--)
        {
            for (int t = 0; t < num_threads; t++)
            {
                start_k[t][j] = temp_k;
                temp_k += local_buckets[t][j].size();
            }
        }

        #pragma omp parallel for schedule(static)
        for (int t = 0; t < num_threads; t++)
        {
            for (int j = num_buckets - 1; j >= 0; j--)
            {
                const vector<uint32_t> &current_bucket = local_buckets[t][j];
                int k = start_k[t][j];
                const size_t &size = current_bucket.size();
                for (uint32_t i = 0; i < size; i++)
                {
                    new_ids[current_bucket[i]] = k++;
                }
            }
        }

        for (int i = 0; i < num_threads; i++)
        {
            for (int j = 0; j < num_buckets; j++)
            {
                local_buckets[i][j].clear();
            }
        }

        t.Stop();
        PrintTime("DBG Map Time", t.Seconds());
    }

    void GenerateHubClusterMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                   pvector<NodeID_> &new_ids, bool useOutdeg)
    {

        typedef std::pair<int64_t, NodeID_> degree_nodeid_t;

        Timer t;
        t.Start();

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges();

        pvector<degree_nodeid_t> degree_id_pairs(num_nodes);
        int64_t avgDegree = num_edges / num_nodes;
        // size_t hubCount {0};

        const int PADDING = 64 / sizeof(uintE);
        int64_t *localOffsets = new int64_t[omp_get_max_threads() * PADDING]();
        int64_t partitionSz = num_nodes / omp_get_max_threads();

        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int startID = partitionSz * tid;
            int stopID = partitionSz * (tid + 1);
            if (tid == omp_get_max_threads() - 1)
            {
                stopID = num_nodes;
            }
            for (int n = startID; n < stopID; ++n)
            {
                if (useOutdeg)
                {
                    int64_t out_degree_n = g.out_degree(n);
                    if (out_degree_n > avgDegree)
                    {
                        ++localOffsets[tid * PADDING];
                        new_ids[n] = 1;
                    }
                }
                else
                {
                    int64_t in_degree_n = g.in_degree(n);
                    if (in_degree_n > avgDegree)
                    {
                        ++localOffsets[tid * PADDING];
                        new_ids[n] = 1;
                    }
                }
            }
        }
        int64_t sum{0};
        for (int tid = 0; tid < omp_get_max_threads(); ++tid)
        {
            auto origCount = localOffsets[tid * PADDING];
            localOffsets[tid * PADDING] = sum;
            sum += origCount;
        }

        /* Step II - assign a remap for the hub vertices first */
        #pragma omp parallel
        {
            int64_t localCtr{0};
            int tid = omp_get_thread_num();
            int64_t startID = partitionSz * tid;
            int64_t stopID = partitionSz * (tid + 1);
            if (tid == omp_get_max_threads() - 1)
            {
                stopID = num_nodes;
            }
            for (int64_t n = startID; n < stopID; ++n)
            {
                if (new_ids[n] != (NodeID_)UINT_E_MAX)
                {
                    new_ids[n] = (NodeID_)localOffsets[tid * PADDING] + (NodeID_)localCtr;
                    ++localCtr;
                }
            }
        }
        delete[] localOffsets;

        /* Step III - assigning a remap for (easy) non hub vertices */
        auto numHubs = sum;
        SlidingQueue<int64_t> queue(numHubs);
        #pragma omp parallel
        {
            // assert(omp_get_max_threads() == 56);
            QueueBuffer<int64_t> lqueue(queue, numHubs / omp_get_max_threads());
            #pragma omp for
            for (int64_t n = numHubs; n < num_nodes; ++n)
            {
                if (new_ids[n] == (NodeID_)UINT_E_MAX)
                {
                    // This steps preserves the ordering of the original graph (as much as
                    // possible)
                    new_ids[n] = (NodeID_)n;
                }
                else
                {
                    int64_t remappedTo = new_ids[n];
                    if (new_ids[remappedTo] == (NodeID_)UINT_E_MAX)
                    {
                        // safe to swap Ids because the original vertex is a non-hub
                        new_ids[remappedTo] = (NodeID_)n;
                    }
                    else
                    {
                        // Cannot swap ids because original vertex was a hub (swapping
                        // would disturb sorted ordering of hubs - not allowed)
                        lqueue.push_back(n);
                    }
                }
            }
            lqueue.flush();
        }
        queue.slide_window(); // the queue keeps a list of vertices where a simple
        // swap of locations is not possible

        /* Step IV - assigning remaps for remaining non hubs */
        int64_t unassignedCtr{0};
        auto q_iter = queue.begin();
        #pragma omp parallel for
        for (int64_t n = 0; n < numHubs; ++n)
        {
            if (new_ids[n] == (NodeID_)UINT_E_MAX)
            {
                int64_t u = *(q_iter + __sync_fetch_and_add(&unassignedCtr, 1));
                new_ids[n] = (NodeID_)u;
            }
        }

        t.Stop();
        PrintTime("HubCluster Map Time", t.Seconds());
    }

    void GenerateCOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                               pvector<NodeID_> &new_ids)
    {
        Timer t;
        t.Start();

        auto num_nodes = g.num_nodes();
        auto num_edges = g.num_edges();
        unsigned average_degree = num_edges / num_nodes;
        params::partition_size = 1024;
        params::num_partitions = (num_nodes - 1) / params::partition_size + 1;
        unsigned num_partitions = params::num_partitions;
        // unsigned max_threads = omp_get_max_threads();
        std::vector<unsigned> segment_large;
        segment_large.reserve(num_nodes);
        std::vector<unsigned> segment_small;
        segment_small.reserve(num_nodes / 2);

        for (unsigned i = 0; i < num_nodes; i++)
            if (g.out_degree(i) > 1 * average_degree)
                segment_large.push_back(i);
            else
                segment_small.push_back(i);

        unsigned num_large_per_seg =
            ceil((float)segment_large.size() / num_partitions);
        params::overflow_ceil = num_large_per_seg;

        unsigned num_small_per_seg = params::partition_size - num_large_per_seg;

        // std::cout << "partition size: " << params::partition_size
        //           << " num of large: " << num_large_per_seg
        //           << " num of small: " << num_small_per_seg << '\n';
        unsigned last_cls = num_partitions - 1;

        while ((num_large_per_seg * last_cls > segment_large.size()) ||
                (num_small_per_seg * last_cls > segment_small.size()))
        {
            last_cls -= 1;
        }

        #pragma omp parallel for schedule(static)
        for (unsigned i = 0; i < last_cls; i++)
        {
            unsigned index = i * params::partition_size;
            for (unsigned j = 0; j < num_large_per_seg; j++)
            {
                new_ids[segment_large[i * num_large_per_seg + j]] = index++;
            }
            for (unsigned j = 0; j < num_small_per_seg; j++)
                new_ids[segment_small[i * num_small_per_seg + j]] = index++;
        }

        auto last_large = num_large_per_seg * last_cls;
        auto last_small = num_small_per_seg * last_cls;
        unsigned index = last_cls * params::partition_size;

        #pragma omp parallel for
        for (unsigned i = last_large; i < segment_large.size(); i++)
        {
            unsigned local_index = __sync_fetch_and_add(&index, 1);
            new_ids[segment_large[i]] = local_index;
        }

        #pragma omp parallel for
        for (unsigned i = last_small; i < segment_small.size(); i++)
        {
            unsigned local_index = __sync_fetch_and_add(&index, 1);
            new_ids[segment_small[i]] = local_index;
        }
        t.Stop();
        PrintTime("COrder Map Time", t.Seconds());
    }

    void GenerateCOrderMapping_v2(const CSRGraph<NodeID_, DestID_, invert> &g,
                                  pvector<NodeID_> &new_ids)
    {
        Timer t;
        t.Start();

        auto num_nodes = g.num_nodes();
        auto num_edges = g.num_edges();

        params::partition_size = 1024;
        params::num_partitions = (num_nodes - 1) / params::partition_size + 1;
        unsigned num_partitions = params::num_partitions;

        uint32_t average_degree = num_edges / num_nodes;

        const int max_threads = omp_get_max_threads();

        Vector2d<unsigned> large_segment(max_threads);
        Vector2d<unsigned> small_segment(max_threads);

        #pragma omp parallel for schedule(static, 1024) num_threads(max_threads)
        for (unsigned i = 0; i < num_nodes; i++)
        {
            if (g.out_degree(i) > average_degree)
            {
                large_segment[omp_get_thread_num()].push_back(i);
            }
            else
            {
                small_segment[omp_get_thread_num()].push_back(i);
            }
        }

        std::vector<unsigned> large_offset(max_threads + 1, 0);
        std::vector<unsigned> small_offset(max_threads + 1, 0);

        large_offset[1] = large_segment[0].size();
        small_offset[1] = small_segment[0].size();
        for (int i = 0; i < max_threads; i++)
        {
            large_offset[i + 1] = large_offset[i] + large_segment[i].size();
            small_offset[i + 1] = small_offset[i] + small_segment[i].size();
        }

        unsigned total_large = large_offset[max_threads];
        unsigned total_small = small_offset[max_threads];

        unsigned cluster_size = params::partition_size;
        unsigned num_clusters = num_partitions;
        unsigned num_large_per_seg = ceil((float)total_large / num_clusters);
        unsigned num_small_per_seg = cluster_size - num_large_per_seg;

        // Parallelize constructing partitions based on the classified hot/cold
        // vertices

        #pragma omp parallel for schedule(static) num_threads(max_threads)
        for (unsigned i = 0; i < num_clusters; i++)
        {

            // Debug output for current cluster's state across all threads

            unsigned index = i * cluster_size;
            unsigned num_large =
                (i != num_clusters - 1) ? (i + 1) * num_large_per_seg : total_large;
            unsigned large_start_t = 0;
            unsigned large_end_t = 0;
            unsigned large_start_v = 0;
            unsigned large_end_v = 0;
            unsigned large_per_seg = (i != num_clusters - 1)
                                     ? num_large_per_seg
                                     : total_large - i * num_large_per_seg;

            unsigned num_small =
                (i != num_clusters - 1) ? (i + 1) * num_small_per_seg : total_small;
            unsigned small_start_t = 0;
            unsigned small_end_t = 0;
            unsigned small_start_v = 0;
            unsigned small_end_v = 0;
            unsigned small_per_seg = (i != num_clusters - 1)
                                     ? num_small_per_seg
                                     : total_small - i * num_small_per_seg;

            // HOT find the starting segment and starting vertex
            for (int t = 0; t < max_threads; t++)
            {
                if (large_offset[t + 1] > num_large - large_per_seg)
                {
                    large_start_t = t;
                    large_start_v = num_large - large_per_seg - large_offset[t];
                    break;
                }
            }
            // HOT find the ending segment and ending vertex
            for (int t = large_start_t; t < max_threads; t++)
            {
                if (large_offset[t + 1] >= num_large)
                {
                    large_end_t = t;
                    large_end_v = num_large - large_offset[t] - 1;
                    break;
                }
            }

            // COLD find the starting segment and starting vertex
            for (int t = 0; t < max_threads; t++)
            {
                if (small_offset[t + 1] > num_small - small_per_seg)
                {
                    small_start_t = t;
                    small_start_v = num_small - small_per_seg - small_offset[t];
                    break;
                }
            }
            // COLD find the ending segment and ending vertex
            for (int t = small_start_t; t < max_threads; t++)
            {
                if (small_offset[t + 1] >= num_small)
                {
                    small_end_t = t;
                    small_end_v = num_small - small_offset[t] - 1;
                    break;
                }
            }

            if (large_start_t == large_end_t)
            {
                if (!large_segment[large_start_t].empty())
                {
                    for (unsigned j = large_start_v; j <= large_end_v; j++)
                    {
                        assert(large_start_t < large_segment.size());
                        assert(j < large_segment[large_start_t].size());
                        new_ids[large_segment[large_start_t][j]] = index++;
                    }
                }
            }
            else
            {
                for (unsigned t = large_start_t; t < large_end_t; t++)
                {
                    if (t != large_start_t)
                        large_start_v = 0;
                    if (!large_segment[t].empty())
                    {
                        for (unsigned j = large_start_v; j < large_segment[t].size(); j++)
                        {
                            new_ids[large_segment[t][j]] = index++;
                        }
                    }
                }
                if (!large_segment[large_end_t].empty())
                {
                    for (unsigned j = 0; j <= large_end_v; j++)
                    {
                        new_ids[large_segment[large_end_t][j]] = index++;
                    }
                }
            }

            if (small_start_t == small_end_t)
            {
                if (!small_segment[small_start_t].empty())
                {
                    for (unsigned j = small_start_v; j <= small_end_v; j++)
                    {
                        assert(small_start_t < small_segment.size());
                        assert(j < small_segment[small_start_t].size());
                        new_ids[small_segment[small_start_t][j]] = index++;
                    }
                }
            }
            else
            {
                for (unsigned t = small_start_t; t < small_end_t; t++)
                {
                    if (t != small_start_t)
                        small_start_v = 0;
                    if (!small_segment[t].empty())
                    {
                        for (unsigned j = small_start_v; j < small_segment[t].size(); j++)
                        {
                            new_ids[small_segment[t][j]] = index++;
                        }
                    }
                }
                if (!small_segment[small_end_t].empty())
                {
                    for (unsigned j = 0; j <= small_end_v; j++)
                    {
                        new_ids[small_segment[small_end_t][j]] = index++;
                    }
                }
            }
        }
        t.Stop();
        PrintTime("COrder Map Time", t.Seconds());
    }

    // @inproceedings{popt-hpca21,
    //   title={P-OPT: Practical Optimal Cache Replacement for Graph Analytics},
    //   author={Balaji, Vignesh and Crago, Neal and Jaleel, Aamer and Lucia,
    //   Brandon}, booktitle={2021 IEEE International Symposium on
    //   High-Performance Computer Architecture (HPCA)}, pages={668--681},
    //   year={2021},
    //   organization={IEEE}
    // }

    /*
       CSR-segmenting as proposed in the Cagra paper

       Partitions the graphs and produces a sub-graph within a specified range of
       vertices.

       The following implementation assumes use in pull implementation and that
       only the partitioned CSC is required
     */
    static CSRGraph<NodeID_, DestID_, invert>
    graphSlicer(const CSRGraph<NodeID_, DestID_, invert> &g, NodeID_ startID,
                NodeID_ stopID, bool outDegree = false,
                bool modifyBothDestlists = false)
    {
        /* create a partition of a graph in the range [startID, stopID) */
        Timer t;
        t.Start();

        // NOTE: that pull implementation should specify outDegree == false
        //       and push implementations should use outDegree == true

        if (g.directed() == true)
        {
            /* Step I : For the requested range [startID, stopID), construct the
             * reduced degree per vertex */
            pvector<NodeID_> degrees(
                g.num_nodes()); // note that stopID is not included in the range
            #pragma omp parallel for schedule(dynamic, 1024)
            for (NodeID_ n = 0; n < g.num_nodes(); ++n)
            {
                if (outDegree == true)
                {
                    NodeID_ newDegree(0);
                    for (NodeID_ m : g.out_neigh(n))
                    {
                        if (m >= startID && m < stopID)
                        {
                            ++newDegree;
                        }
                    }
                    degrees[n] = newDegree;
                }
                else
                {
                    NodeID_ newDegree(0);
                    for (NodeID_ m : g.in_neigh(n))
                    {
                        if (m >= startID && m < stopID)
                        {
                            ++newDegree;
                        }
                    }
                    degrees[n] = newDegree;
                }
            }

            /* Step II : Construct a trimmed offset list */
            pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
            DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
            DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
            if (outDegree == true)
            {
                #pragma omp parallel for schedule(dynamic, 1024)
                for (NodeID_ u = 0; u < g.num_nodes(); ++u)
                {
                    for (NodeID_ v : g.out_neigh(u))
                    {
                        if (v >= startID && v < stopID)
                        {
                            neighs[offsets[u]++] = v;
                        }
                    }
                }
            }
            else
            {
                #pragma omp parallel for schedule(dynamic, 1024)
                for (NodeID_ u = 0; u < g.num_nodes(); ++u)
                {
                    for (NodeID_ v : g.in_neigh(u))
                    {
                        if (v >= startID && v < stopID)
                        {
                            neighs[offsets[u]++] = v;
                        }
                    }
                }
            }

            /* Step III : Populate the inv dest lists (for push-pull implementations)
             */
            DestID_ *inv_neighs(nullptr);
            DestID_ **inv_index(nullptr);
            if (modifyBothDestlists == true)
            {
                // allocate space
                pvector<NodeID_> inv_degrees(g.num_nodes());
                #pragma omp parallel for
                for (NodeID_ u = 0; u < g.num_nodes(); ++u)
                {
                    if (outDegree == true)
                    {
                        inv_degrees[u] = g.in_degree(u);
                    }
                    else
                    {
                        inv_degrees[u] = g.out_degree(u);
                    }
                }
                pvector<SGOffset> inv_offsets = ParallelPrefixSum(inv_degrees);
                inv_neighs = new DestID_[inv_offsets[g.num_nodes()]];
                inv_index =
                    CSRGraph<NodeID_, DestID_>::GenIndex(inv_offsets, inv_neighs);

                // populate the inv dest list
                #pragma omp parallel for schedule(dynamic, 1024)
                for (NodeID_ u = 0; u < g.num_nodes(); ++u)
                {
                    if (outDegree == true)
                    {
                        for (NodeID_ v : g.in_neigh(u))
                        {
                            inv_neighs[inv_offsets[u]++] = v;
                        }
                    }
                    else
                    {
                        for (NodeID_ v : g.out_neigh(u))
                        {
                            inv_neighs[inv_offsets[u]++] = v;
                        }
                    }
                }
            }

            /* Step IV : return the appropriate graph */
            if (outDegree == true)
            {
                t.Stop();
                PrintTime("Slice-time", t.Seconds());
                return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs,
                        inv_index, inv_neighs);
            }
            else
            {
                t.Stop();
                PrintTime("Slice-time", t.Seconds());
                return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), inv_index,
                        inv_neighs, index, neighs);
            }
        }
        else
        {
            /* Step I : For the requested range [startID, stopID), construct the
             * reduced degree per vertex */
            pvector<NodeID_> degrees(
                g.num_nodes()); // note that stopID is not included in the range
            #pragma omp parallel for schedule(dynamic, 1024)
            for (NodeID_ n = 0; n < g.num_nodes(); ++n)
            {
                NodeID_ newDegree(0);
                for (NodeID_ m : g.out_neigh(n))
                {
                    if (m >= startID && m < stopID)
                    {
                        ++newDegree; // if neighbor is in current partition
                    }
                }
                degrees[n] = newDegree;
            }

            /* Step II : Construct a trimmed offset list */
            pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
            DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
            DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
            #pragma omp parallel for schedule(dynamic, 1024)
            for (NodeID_ u = 0; u < g.num_nodes(); ++u)
            {
                for (NodeID_ v : g.out_neigh(u))
                {
                    if (v >= startID && v < stopID)
                    {
                        neighs[offsets[u]++] = v;
                    }
                }
            }

            /* Step III : return the appropriate graph */
            t.Stop();
            PrintTime("Slice-time", t.Seconds());
            return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
        }
    }

    static CSRGraph<NodeID_, DestID_, invert>
    quantizeGraph(const CSRGraph<NodeID_, DestID_, invert> &g, NodeID_ numTiles)
    {

        NodeID_ tileSz = g.num_nodes() / numTiles;
        if (numTiles > g.num_nodes())
            tileSz = 1;
        else if (g.num_nodes() % numTiles != 0)
            tileSz += 1;

        pvector<NodeID_> degrees(g.num_nodes(), 0);
        #pragma omp parallel for
        for (NodeID_ n = 0; n < g.num_nodes(); ++n)
        {
            std::set<NodeID_> uniqNghs;
            for (NodeID_ ngh : g.out_neigh(n))
            {
                uniqNghs.insert(ngh / tileSz);
            }
            degrees[n] = uniqNghs.size();
            assert(degrees[n] <= numTiles);
        }

        pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
        DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
        DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
        #pragma omp parallel for schedule(dynamic, 1024)
        for (NodeID_ u = 0; u < g.num_nodes(); u++)
        {
            std::set<NodeID_> uniqNghs;
            for (NodeID_ ngh : g.out_neigh(u))
            {
                uniqNghs.insert(ngh / tileSz);
            }

            auto it = uniqNghs.begin();
            for (NodeID_ i = 0; i < static_cast<NodeID_>(uniqNghs.size()); ++i)
            {
                neighs[offsets[u]++] = *it;
                it++;
            }
        }
        return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
    }

    // Proper degree sorting
    static CSRGraph<NodeID_, DestID_, invert>
    DegSort(const CSRGraph<NodeID_, DestID_, invert> &g, bool outDegree,
            pvector<NodeID_> &new_ids, bool createOnlyDegList,
            bool createBothCSRs)
    {
        Timer t;
        t.Start();

        typedef std::pair<int64_t, NodeID_> degree_node_p;
        pvector<degree_node_p> degree_id_pairs(g.num_nodes());
        if (g.directed() == true)
        {
            /* Step I: Create a list of degrees */
            #pragma omp parallel for
            for (NodeID_ n = 0; n < g.num_nodes(); n++)
            {
                if (outDegree == true)
                {
                    degree_id_pairs[n] = std::make_pair(g.out_degree(n), n);
                }
                else
                {
                    degree_id_pairs[n] = std::make_pair(g.in_degree(n), n);
                }
            }

            /* Step II: Sort based on degree order */
            __gnu_parallel::sort(
                degree_id_pairs.begin(), degree_id_pairs.end(),
                std::greater<degree_node_p>()); // TODO:Use parallel sort

            /* Step III: assigned remap for the hub vertices */
            #pragma omp parallel for
            for (NodeID_ n = 0; n < g.num_nodes(); n++)
            {
                new_ids[degree_id_pairs[n].second] = n;
            }

            /* Step VI: generate degree to build a new graph */
            pvector<NodeID_> degrees(g.num_nodes());
            pvector<NodeID_> inv_degrees(g.num_nodes());
            if (outDegree == true)
            {
                #pragma omp parallel for
                for (NodeID_ n = 0; n < g.num_nodes(); n++)
                {
                    degrees[new_ids[n]] = g.out_degree(n);
                    inv_degrees[new_ids[n]] = g.in_degree(n);
                }
            }
            else
            {
                #pragma omp parallel for
                for (NodeID_ n = 0; n < g.num_nodes(); n++)
                {
                    degrees[new_ids[n]] = g.in_degree(n);
                    inv_degrees[new_ids[n]] = g.out_degree(n);
                }
            }

            /* Graph building phase */
            pvector<SGOffset> offsets = ParallelPrefixSum(inv_degrees);
            DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
            DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
            #pragma omp parallel for schedule(dynamic, 1024)
            for (NodeID_ u = 0; u < g.num_nodes(); u++)
            {
                if (outDegree == true)
                {
                    for (NodeID_ v : g.in_neigh(u))
                        neighs[offsets[new_ids[u]]++] = new_ids[v];
                }
                else
                {
                    for (NodeID_ v : g.out_neigh(u))
                        neighs[offsets[new_ids[u]]++] = new_ids[v];
                }
                std::sort(index[new_ids[u]],
                          index[new_ids[u] + 1]); // sort neighbors of each vertex
            }
            DestID_ *inv_neighs(nullptr);
            DestID_ **inv_index(nullptr);
            if (createOnlyDegList == true || createBothCSRs == true)
            {
                // making the inverse list (in-degrees in this case)
                pvector<SGOffset> inv_offsets = ParallelPrefixSum(degrees);
                inv_neighs = new DestID_[inv_offsets[g.num_nodes()]];
                inv_index =
                    CSRGraph<NodeID_, DestID_>::GenIndex(inv_offsets, inv_neighs);
                if (createBothCSRs == true)
                {
                    #pragma omp parallel for schedule(dynamic, 1024)
                    for (NodeID_ u = 0; u < g.num_nodes(); u++)
                    {
                        if (outDegree == true)
                        {
                            for (NodeID_ v : g.out_neigh(u))
                                inv_neighs[inv_offsets[new_ids[u]]++] = new_ids[v];
                        }
                        else
                        {
                            for (NodeID_ v : g.in_neigh(u))
                                inv_neighs[inv_offsets[new_ids[u]]++] = new_ids[v];
                        }
                        std::sort(
                            inv_index[new_ids[u]],
                            inv_index[new_ids[u] + 1]); // sort neighbors of each vertex
                    }
                }
            }
            t.Stop();
            PrintTime("Sort Map Time", t.Seconds());
            if (outDegree == true)
            {
                return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), inv_index,
                        inv_neighs, index, neighs);
            }
            else
            {
                return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs,
                        inv_index, inv_neighs);
            }
        }
        else
        {
            /* Undirected graphs - no need to make separate lists for in and out degree */
            /* Step I: Create a list of degrees */
            #pragma omp parallel for
            for (NodeID_ n = 0; n < g.num_nodes(); n++)
            {
                degree_id_pairs[n] = std::make_pair(g.out_degree(n), n);
            }

            /* Step II: Sort based on degree order */
            __gnu_parallel::sort(
                degree_id_pairs.begin(), degree_id_pairs.end(),
                std::greater<degree_node_p>()); // TODO:Use parallel sort

            /* Step III: assigned remap for the hub vertices */
            #pragma omp parallel for
            for (NodeID_ n = 0; n < g.num_nodes(); n++)
            {
                new_ids[degree_id_pairs[n].second] = n;
            }

            /* Step VI: generate degree to build a new graph */
            pvector<NodeID_> degrees(g.num_nodes());
            #pragma omp parallel for
            for (NodeID_ n = 0; n < g.num_nodes(); n++)
            {
                degrees[new_ids[n]] = g.out_degree(n);
            }

            /* Graph building phase */
            pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
            DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
            DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
            #pragma omp parallel for schedule(dynamic, 1024)
            for (NodeID_ u = 0; u < g.num_nodes(); u++)
            {
                for (NodeID_ v : g.out_neigh(u))
                    neighs[offsets[new_ids[u]]++] = new_ids[v];
                std::sort(index[new_ids[u]], index[new_ids[u] + 1]);
            }
            t.Stop();
            PrintTime("Sort Map Time", t.Seconds());
            return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
        }
    }

    static CSRGraph<NodeID_, DestID_, invert>
    RandOrder(const CSRGraph<NodeID_, DestID_, invert> &g,
              pvector<NodeID_> &new_ids, bool createOnlyDegList,
              bool createBothCSRs)
    {
        Timer t;
        t.Start();
        std::srand(0); // so that the random graph generated is the same everytime
        bool outDegree = true;

        if (g.directed() == true)
        {
            // Step I: create a random permutation - SLOW implementation
            pvector<NodeID_> claimedVtxs(g.num_nodes(), 0);

            // #pragma omp parallel for
            for (NodeID_ v = 0; v < g.num_nodes(); ++v)
            {
                while (true)
                {
                    NodeID_ randID = std::rand() % g.num_nodes();
                    if (claimedVtxs[randID] != 1)
                    {
                        if (compare_and_swap(claimedVtxs[randID], 0, 1) == true)
                        {
                            new_ids[v] = randID;
                            break;
                        }
                        else
                            continue;
                    }
                }
            }

            #pragma omp parallel for
            for (NodeID_ v = 0; v < g.num_nodes(); ++v)
                assert(new_ids[v] != -1);

            /* Step VI: generate degree to build a new graph */
            pvector<NodeID_> degrees(g.num_nodes());
            pvector<NodeID_> inv_degrees(g.num_nodes());
            if (outDegree == true)
            {
                #pragma omp parallel for
                for (NodeID_ n = 0; n < g.num_nodes(); n++)
                {
                    degrees[new_ids[n]] = g.out_degree(n);
                    inv_degrees[new_ids[n]] = g.in_degree(n);
                }
            }
            else
            {
                #pragma omp parallel for
                for (NodeID_ n = 0; n < g.num_nodes(); n++)
                {
                    degrees[new_ids[n]] = g.in_degree(n);
                    inv_degrees[new_ids[n]] = g.out_degree(n);
                }
            }

            /* Graph building phase */
            pvector<SGOffset> offsets = ParallelPrefixSum(inv_degrees);
            DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
            DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
            #pragma omp parallel for schedule(dynamic, 1024)
            for (NodeID_ u = 0; u < g.num_nodes(); u++)
            {
                if (outDegree == true)
                {
                    for (NodeID_ v : g.in_neigh(u))
                        neighs[offsets[new_ids[u]]++] = new_ids[v];
                }
                else
                {
                    for (NodeID_ v : g.out_neigh(u))
                        neighs[offsets[new_ids[u]]++] = new_ids[v];
                }
                std::sort(index[new_ids[u]],
                          index[new_ids[u] + 1]); // sort neighbors of each vertex
            }
            DestID_ *inv_neighs(nullptr);
            DestID_ **inv_index(nullptr);
            if (createOnlyDegList == true || createBothCSRs == true)
            {
                // making the inverse list (in-degrees in this case)
                pvector<SGOffset> inv_offsets = ParallelPrefixSum(degrees);
                inv_neighs = new DestID_[inv_offsets[g.num_nodes()]];
                inv_index =
                    CSRGraph<NodeID_, DestID_>::GenIndex(inv_offsets, inv_neighs);
                if (createBothCSRs == true)
                {
                    #pragma omp parallel for schedule(dynamic, 1024)
                    for (NodeID_ u = 0; u < g.num_nodes(); u++)
                    {
                        if (outDegree == true)
                        {
                            for (NodeID_ v : g.out_neigh(u))
                                inv_neighs[inv_offsets[new_ids[u]]++] = new_ids[v];
                        }
                        else
                        {
                            for (NodeID_ v : g.in_neigh(u))
                                inv_neighs[inv_offsets[new_ids[u]]++] = new_ids[v];
                        }
                        std::sort(
                            inv_index[new_ids[u]],
                            inv_index[new_ids[u] + 1]); // sort neighbors of each vertex
                    }
                }
            }
            t.Stop();
            PrintTime("Random Map Time", t.Seconds());
            if (outDegree == true)
            {
                return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), inv_index,
                        inv_neighs, index, neighs);
            }
            else
            {
                return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs,
                        inv_index, inv_neighs);
            }
        }
        else
        {
            /* Undirected graphs - no need to make separate lists for in and out
             * degree */
            // Step I: create a random permutation - SLOW implementation
            pvector<NodeID_> claimedVtxs(g.num_nodes(), 0);

            // #pragma omp parallel for
            for (NodeID_ v = 0; v < g.num_nodes(); ++v)
            {
                while (true)
                {
                    NodeID_ randID = std::rand() % g.num_nodes();
                    if (claimedVtxs[randID] != 1)
                    {
                        if (compare_and_swap(claimedVtxs[randID], 0, 1) == true)
                        {
                            new_ids[v] = randID;
                            break;
                        }
                        else
                            continue;
                    }
                }
            }

            #pragma omp parallel for
            for (NodeID_ v = 0; v < g.num_nodes(); ++v)
                assert(new_ids[v] != -1);

            /* Step VI: generate degree to build a new graph */
            pvector<NodeID_> degrees(g.num_nodes());
            #pragma omp parallel for
            for (NodeID_ n = 0; n < g.num_nodes(); n++)
            {
                degrees[new_ids[n]] = g.out_degree(n);
            }

            /* Graph building phase */
            pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
            DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
            DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
            #pragma omp parallel for schedule(dynamic, 1024)
            for (NodeID_ u = 0; u < g.num_nodes(); u++)
            {
                for (NodeID_ v : g.out_neigh(u))
                    neighs[offsets[new_ids[u]]++] = new_ids[v];
                std::sort(index[new_ids[u]], index[new_ids[u] + 1]);
            }
            t.Stop();
            PrintTime("RandOrder Time", t.Seconds());
            return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
        }
    }

    /*
       Return a compressed transpose matrix (Rereference Matrix)
     */
    static void makeOffsetMatrix(const CSRGraph<NodeID_, DestID_, invert> &g,
                                 pvector<uint8_t> &offsetMatrix,
                                 int numVtxPerLine, int numEpochs,
                                 bool traverseCSR = true)
    {
        if (g.directed() == false)
            traverseCSR = true;

        Timer tm;

        /* Step I: Collect quantized edges & Compact vertices into "super vertices"
         */
        tm.Start();
        NodeID_ numCacheLines = (g.num_nodes() + numVtxPerLine - 1) / numVtxPerLine;
        NodeID_ epochSz = (g.num_nodes() + numEpochs - 1) / numEpochs;
        pvector<NodeID_> lastRef(numCacheLines * numEpochs, -1);
        NodeID_ chunkSz = 64 / numVtxPerLine;
        if (chunkSz == 0)
            chunkSz = 1;

        #pragma omp parallel for schedule(dynamic, chunkSz)
        for (NodeID_ c = 0; c < numCacheLines; ++c)
        {
            NodeID_ startVtx = c * numVtxPerLine;
            NodeID_ endVtx = (c + 1) * numVtxPerLine;
            if (c == numCacheLines - 1)
                endVtx = g.num_nodes();

            for (NodeID_ v = startVtx; v < endVtx; ++v)
            {
                if (traverseCSR == true)
                {
                    for (NodeID_ ngh : g.out_neigh(v))
                    {
                        NodeID_ nghEpoch = ngh / epochSz;
                        lastRef[(c * numEpochs) + nghEpoch] =
                            std::max(ngh, lastRef[(c * numEpochs) + nghEpoch]);
                    }
                }
                else
                {
                    for (NodeID_ ngh : g.in_neigh(v))
                    {
                        NodeID_ nghEpoch = ngh / epochSz;
                        lastRef[(c * numEpochs) + nghEpoch] =
                            std::max(ngh, lastRef[(c * numEpochs) + nghEpoch]);
                    }
                }
            }
        }
        tm.Stop();
        std::cout << "[CSR-HYBRID-PREPROCESSING] Time to quantize nghs and compact "
                  "vertices = "
                  << tm.Seconds() << std::endl;
        assert(numEpochs == 256);

        /* Step II: Converting adjacency matrix into offsets */
        tm.Start();
        uint8_t maxReref = 127; // because MSB is reserved for identifying between
        // reref val (1) & switch point (0)
        NodeID_ subEpochSz =
            (epochSz + 127) /
            128; // Using remaining 7 bits to identify intra-epoch information
        pvector<uint8_t> compressedOffsets(numCacheLines * numEpochs);
        uint8_t mask = 1;
        uint8_t orMask = mask << 7;
        uint8_t andMask = ~(orMask);
        assert(orMask == 128 && andMask == 127);
        #pragma omp parallel for schedule(static)
        for (NodeID_ c = 0; c < numCacheLines; ++c)
        {
            {
                // first set values for the last epoch
                NodeID_ e = numEpochs - 1;
                if (lastRef[(c * numEpochs) + e] != -1)
                {
                    compressedOffsets[(c * numEpochs) + e] = maxReref;
                    compressedOffsets[(c * numEpochs) + e] &= andMask;
                }
                else
                {
                    compressedOffsets[(c * numEpochs) + e] = maxReref;
                    compressedOffsets[(c * numEpochs) + e] |= orMask;
                }
            }

            // Now back track and set values for all epochs
            for (NodeID_ e = numEpochs - 2; e >= 0; --e)
            {
                if (lastRef[(c * numEpochs) + e] != -1)
                {
                    // There was a ref this epoch - store the quantized val of the lastRef
                    NodeID_ subEpochDist = lastRef[(c * numEpochs) + e] - (e * epochSz);
                    assert(subEpochDist >= 0);
                    NodeID_ lastRefQ = (subEpochDist / subEpochSz);
                    assert(lastRefQ <= maxReref);
                    compressedOffsets[(c * numEpochs) + e] =
                        static_cast<uint8_t>(lastRefQ);
                    compressedOffsets[(c * numEpochs) + e] &= andMask;
                }
                else
                {
                    if ((compressedOffsets[(c * numEpochs) + e + 1] & orMask) != 0)
                    {
                        // No access next epoch as well - add inter-epoch distance
                        uint8_t nextRef =
                            compressedOffsets[(c * numEpochs) + e + 1] & andMask;
                        if (nextRef == maxReref)
                            compressedOffsets[(c * numEpochs) + e] = maxReref;
                        else
                            compressedOffsets[(c * numEpochs) + e] = nextRef + 1;
                    }
                    else
                    {
                        // There is an access next epoch - so inter-epoch distance is set to
                        // next epoch
                        compressedOffsets[(c * numEpochs) + e] = 1;
                    }
                    compressedOffsets[(c * numEpochs) + e] |= orMask;
                }
            }
        }
        tm.Stop();
        std::cout
                << "[CSR-HYBRID-PREPROCESSING] Time to convert to offsets matrix = "
                << tm.Seconds() << std::endl;

        /* Step III: Transpose edgePresent*/
        tm.Start();
        #pragma omp parallel for schedule(static)
        for (NodeID_ c = 0; c < numCacheLines; ++c)
        {
            for (NodeID_ e = 0; e < numEpochs; ++e)
            {
                offsetMatrix[(e * numCacheLines) + c] =
                    compressedOffsets[(c * numEpochs) + e];
            }
        }
        tm.Stop();
        std::cout
                << "[CSR-HYBRID-PREPROCESSING] Time to transpose offsets matrix =  "
                << tm.Seconds() << std::endl;
    }

    //
    // A demo program of reordering using Rabbit Order.
    //
    // Author: ARAI Junya <arai.junya@lab.ntt.co.jp> <araijn@gmail.com>
    //

#ifdef RABBIT_ENABLE
    typedef std::vector<std::vector<std::pair<rabbit_order::vint, float>>>
    adjacency_list;

    rabbit_order::vint
    count_unused_id(const rabbit_order::vint n,
                    const std::vector<edge_list::edge> &edges)
    {
        std::vector<char> appears(n);
        for (size_t i = 0; i < edges.size(); ++i)
        {
            appears[std::get<0>(edges[i])] = true;
            appears[std::get<1>(edges[i])] = true;
        }
        return static_cast<rabbit_order::vint>(boost::count(appears, false));
    }

    template <typename RandomAccessRange>
    adjacency_list make_adj_list(const rabbit_order::vint n,
                                 const RandomAccessRange &es)
    {
        using std::get;

        // Symmetrize the edge list and remove self-loops simultaneously
        std::vector<edge_list::edge> ss(boost::size(es) * 2);
        #pragma omp parallel for
        for (size_t i = 0; i < boost::size(es); ++i)
        {
            auto &e = es[i];
            if (get<0>(e) != get<1>(e))
            {
                ss[i * 2] = std::make_tuple(get<0>(e), get<1>(e), get<2>(e));
                ss[i * 2 + 1] = std::make_tuple(get<1>(e), get<0>(e), get<2>(e));
            }
            else
            {
                // Insert zero-weight edges instead of loops; they are ignored in making
                // an adjacency list
                ss[i * 2] = std::make_tuple(0, 0, 0.0f);
                ss[i * 2 + 1] = std::make_tuple(0, 0, 0.0f);
            }
        }

        // Sort the edges
        __gnu_parallel::sort(ss.begin(), ss.end());

        // Convert to an adjacency list
        adjacency_list adj(n);
        #pragma omp parallel
        {
            // Advance iterators to a boundary of a source vertex
            const auto adv = [](auto it, const auto first, const auto last)
            {
                while (first != it && it != last && get<0>(*(it - 1)) == get<0>(*it))
                    ++it;
                return it;
            };

            // Compute an iterator range assigned to this thread
            const int p = omp_get_max_threads();
            const size_t t = static_cast<size_t>(omp_get_thread_num());
            const size_t ifirst = ss.size() / p * (t) + std::min(t, ss.size() % p);
            const size_t ilast =
                ss.size() / p * (t + 1) + std::min(t + 1, ss.size() % p);
            auto it = adv(ss.begin() + ifirst, ss.begin(), ss.end());
            const auto last = adv(ss.begin() + ilast, ss.begin(), ss.end());

            // Reduce edges and store them in std::vector
            while (it != last)
            {
                const rabbit_order::vint s = get<0>(*it);

                // Obtain an upper bound of degree and reserve memory
                const auto maxdeg =
                    std::find_if(it, last, [s](auto & x)
                {
                    return get<0>(x) != s;
                }) -
                it;
                adj[s].reserve(maxdeg);

                while (it != last && get<0>(*it) == s)
                {
                    const rabbit_order::vint t = get<1>(*it);
                    float w = 0.0;
                    while (it != last && get<0>(*it) == s && get<1>(*it) == t)
                        w += get<2>(*it++);
                    if (w > 0.0)
                        adj[s].push_back({t, w});
                }

                // The actual degree can be smaller than the upper bound
                adj[s].shrink_to_fit();
            }
        }

        return adj;
    }

    adjacency_list read_graph(const std::string &graphpath)
    {
        const auto edges = edge_list::read(graphpath);

        // The number of vertices = max vertex ID + 1 (assuming IDs start from zero)
        const auto n = boost::accumulate(
                           edges, static_cast<rabbit_order::vint>(0),
                           [](rabbit_order::vint s, auto & e)
        {
            return std::max(s, std::max(std::get<0>(e), std::get<1>(e)) + 1);
        });

        if (const size_t c = count_unused_id(n, edges))
        {
            // std::cerr << "WARNING: " << c << "/" << n << " vertex IDs are unused"
            // << " (zero-degree vertices or noncontiguous IDs?)\n";
        }

        return make_adj_list(n, edges);
    }

    adjacency_list
    readRabbitOrderGraphCSR(const CSRGraph<NodeID_, DestID_, invert> &g)
    {

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges_directed();

        std::vector<edge_list::edge> edges(num_edges);
        edges.resize(num_edges);

        // Parallel loop to construct the edge list
        #pragma omp parallel for
        for (NodeID_ i = 0; i < num_nodes; ++i)
        {
            NodeID_ out_start = g.out_offset(i);

            NodeID_ j = 0;
            for (DestID_ neighbor : g.out_neigh(i))
            {
                if (g.is_weighted())
                {
                    NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                    WeightT_ weight =
                        static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;

                    edge_list::edge edge(i, dest, weight);
                    edges[out_start + j] = edge;
                }
                else
                {
                    edge_list::edge edge(i, neighbor, 1.0f);
                    edges[out_start + j] = edge;
                }
                ++j;
            }
        }

        // The number of vertices = max vertex ID + 1 (assuming IDs start from zero)
        const auto n = boost::accumulate(
                           edges, static_cast<rabbit_order::vint>(0),
                           [](rabbit_order::vint s, auto & e)
        {
            return std::max(s, std::max(std::get<0>(e), std::get<1>(e)) + 1);
        });

        // if (const size_t c = count_unused_id(n, edges)) {
        //   std::cerr << "WARNING: " << c << "/" << n << " vertex IDs are unused"
        //             << " (zero-degree vertices or noncontiguous IDs?)\n";
        // }

        return make_adj_list(n, edges);
    }

    adjacency_list
    readRabbitOrderAdjacencylist(const std::vector<edge_list::edge> &edges)
    {

        // The number of vertices = max vertex ID + 1 (assuming IDs start from zero)
        const auto n = boost::accumulate(
                           edges, static_cast<rabbit_order::vint>(0),
                           [](rabbit_order::vint s, auto & e)
        {
            return std::max(s, std::max(std::get<0>(e), std::get<1>(e)) + 1);
        });

        // if (const size_t c = count_unused_id(n, edges)) {
        //   std::cerr << "WARNING: " << c << "/" << n << " vertex IDs are unused"
        //             << " (zero-degree vertices or noncontiguous IDs?)\n";
        // }

        return make_adj_list(n, edges);
    }
#endif

    void GenerateRabbitOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                    pvector<NodeID_> &new_ids)
    {
#ifdef RABBIT_ENABLE
        using boost::adaptors::transformed;

        // std::cerr << "Number of threads: " << omp_get_num_threads() << std::endl;
        // omp_set_num_threads(omp_get_max_threads());

        // std::cerr << "Number of threads: " << omp_get_max_threads() << std::endl;
        auto adj = readRabbitOrderGraphCSR(g);
        // const auto m =
        //     boost::accumulate(adj | transformed([](auto &es) { return es.size();
        //     }),
        //                       static_cast<size_t>(0));
        // std::cerr << "Number of vertices: " << adj.size() << std::endl;
        // std::cerr << "Number of edges: " << m << std::endl;

        // if (commode)
        //   detect_community(std::move(adj));
        // else
        reorder_internal(std::move(adj), new_ids);
#else
        GenerateOriginalMapping(g, new_ids);
#endif
    }

    double GenerateRabbitModularityEdgelist(EdgeList &edgesList,
                                            bool is_weighted)
    {

#ifdef RABBIT_ENABLE
        using boost::adaptors::transformed;

        std::vector<edge_list::edge> edges(edgesList.size());

        // Parallel for loop
        #pragma omp parallel for
        for (size_t i = 0; i < edges.size(); ++i)
        {
            if (is_weighted)
            {
                rabbit_order::vint src = edgesList[i].u;
                rabbit_order::vint dest =
                    static_cast<NodeWeight<NodeID_, WeightT_>>(edgesList[i].v).v;
                float weight =
                    static_cast<NodeWeight<NodeID_, WeightT_>>(edgesList[i].v).w;
                edges[i] = std::make_tuple(src, dest, weight);
            }
            else
            {
                rabbit_order::vint src = edgesList[i].u;
                rabbit_order::vint dest = edgesList[i].v;
                edges[i] = std::make_tuple(src, dest, 1.0f);
            }
        }

        auto adj = readRabbitOrderAdjacencylist(edges);
        auto _adj = adj; // copy `adj` because it is used for computing modularity
        //--------------------------------------------
        auto g = rabbit_order::aggregate(std::move(_adj));
        const auto c = std::make_unique<rabbit_order::vint[]>(g.n());
        #pragma omp parallel for
        for (rabbit_order::vint v = 0; v < g.n(); ++v)
            c[v] = rabbit_order::trace_com(v, &g);

        const double q = compute_modularity(adj, c.get());
        return q;
#else
        return 1.0f;
#endif
    }

#ifdef RABBIT_ENABLE
    void
    GenerateRabbitOrderMappingEdgelist(const std::vector<edge_list::edge> &edges,
                                       pvector<NodeID_> &new_ids)
    {

        using boost::adaptors::transformed;

        // std::cerr << "Number of threads: " << omp_get_num_threads() << std::endl;
        // omp_set_num_threads(omp_get_max_threads());

        // std::cerr << "Number of threads: " << omp_get_max_threads() << std::endl;
        auto adj = readRabbitOrderAdjacencylist(edges);
        // const auto m =
        //     boost::accumulate(adj | transformed([](auto &es) { return es.size();
        //     }),
        //                       static_cast<size_t>(0));
        // std::cerr << "Number of vertices: " << adj.size() << std::endl;
        // std::cerr << "Number of edges: " << m << std::endl;

        // if (commode)
        //   detect_community(std::move(adj));
        // else
        reorder_internal_single(std::move(adj), new_ids);
    }

    template <typename InputIt>
    typename std::iterator_traits<InputIt>::difference_type
    count_uniq(const InputIt f, const InputIt l)
    {
        std::vector<typename std::iterator_traits<InputIt>::value_type> ys(f, l);
        return boost::size(boost::unique(boost::sort(ys)));
    }

    double compute_modularity(const adjacency_list &adj,
                              const rabbit_order::vint *const coms)
    {
        const rabbit_order::vint n = static_cast<rabbit_order::vint>(adj.size());
        const auto ncom = count_uniq(coms, coms + n);
        double m2 = 0.0; // total weight of the (bidirectional) edges

        std::unordered_map<rabbit_order::vint, double[2]> degs(
            ncom); // ID -> {all, loop}
        degs.reserve(ncom);

        #pragma omp parallel reduction(+ : m2)
        {
            std::unordered_map<rabbit_order::vint, double[2]> mydegs(ncom);
            mydegs.reserve(ncom);

            #pragma omp for
            for (rabbit_order::vint v = 0; v < n; ++v)
            {
                const rabbit_order::vint c = coms[v];
                auto *const d = &mydegs[c];
                for (const auto e : adj[v])
                {
                    m2 += e.second;
                    (*d)[0] += e.second;
                    if (coms[e.first] == c)
                        (*d)[1] += e.second;
                }
            }

            #pragma omp critical
            {
                for (auto &kv : mydegs)
                {
                    auto *const d = &degs[kv.first];
                    (*d)[0] += kv.second[0];
                    (*d)[1] += kv.second[1];
                }
            }
        }
        assert(static_cast<intmax_t>(degs.size()) == ncom);

        double q = 0.0;
        for (auto &kv : degs)
        {
            const double all = kv.second[0];
            const double loop = kv.second[1];
            q += loop / m2 - (all / m2) * (all / m2);
        }

        return q;
    }

    void detect_community(adjacency_list adj)
    {
        auto _adj = adj; // copy `adj` because it is used for computing modularity

        // std::cerr << "Detecting communities...\n";
        const double tstart = rabbit_order::now_sec();
        //--------------------------------------------
        auto g = rabbit_order::aggregate(std::move(_adj));
        const auto c = std::make_unique<rabbit_order::vint[]>(g.n());
        #pragma omp parallel for
        for (rabbit_order::vint v = 0; v < g.n(); ++v)
            c[v] = rabbit_order::trace_com(v, &g);
        //--------------------------------------------
        // std::cerr << "Community detection Time"
        //           << rabbit_order::now_sec() - tstart << std::endl;
        PrintTime("Community Time", rabbit_order::now_sec() - tstart);

        // Print the result
        // std::copy(&c[0], &c[g.n()],
        //           std::ostream_iterator<rabbit_order::vint>(std::cout, "\n"));

        // std::cerr << "Computing modularity of the result...\n";
        const double q = compute_modularity(adj, c.get());
        std::cerr << "Modularity: " << q << std::endl;
    }

    void reorder(adjacency_list adj)
    {
        // std::cerr << "Generating a permutation...\n";
        const double tstart = rabbit_order::now_sec();
        //--------------------------------------------
        const auto g = rabbit_order::aggregate(std::move(adj));
        const auto p = rabbit_order::compute_perm(g);
        //--------------------------------------------
        // std::cerr << "Permutation generation Time: "
        //           << rabbit_order::now_sec() - tstart << std::endl;
        PrintTime("Permutation generation Time", rabbit_order::now_sec() - tstart);
        // Print the result
        std::copy(&p[0], &p[g.n()],
                  std::ostream_iterator<rabbit_order::vint>(std::cout, "\n"));
    }

    void reorder_internal(adjacency_list adj, pvector<NodeID_> &new_ids)
    {
        // std::cerr << "Generating a permutation...\n";
        auto _adj = adj; // copy `adj` because it is used for computing modularity
        const double tstart = rabbit_order::now_sec();
        //--------------------------------------------
        auto g = rabbit_order::aggregate(std::move(_adj));
        const auto p = rabbit_order::compute_perm(g);
        const double tend = rabbit_order::now_sec();
        //--------------------------------------------
        const auto c = std::make_unique<rabbit_order::vint[]>(g.n());
        #pragma omp parallel for
        for (rabbit_order::vint v = 0; v < g.n(); ++v)
            c[v] = rabbit_order::trace_com(v, &g);

        const double q = compute_modularity(adj, c.get());

        // Print the result
        // std::copy(&c[0], &c[g.n()],
        //           std::ostream_iterator<rabbit_order::vint>(std::cout, "\n"));

        PrintTime("Modularity", q);
        //--------------------------------------------
        // std::cerr << "Permutation generation Time: "
        //           << rabbit_order::now_sec() - tstart << std::endl;
        PrintTime("RabbitOrder Map Time", tend - tstart);
        // Ensure new_ids is large enough to hold all new IDs

        if (new_ids.size() < g.n())
            new_ids.resize(g.n());

        #pragma omp parallel for
        for (size_t i = 0; i < g.n(); ++i)
        {
            new_ids[i] = (NodeID_)p[i];
        }
    }

    void reorder_internal_single(adjacency_list adj, pvector<NodeID_> &new_ids)
    {
        // std::cerr << "Generating a permutation...\n";
        auto _adj = adj; // copy `adj` because it is used for computing modularity
        const double tstart = rabbit_order::now_sec();
        //--------------------------------------------
        auto g = rabbit_order::aggregate(std::move(_adj));
        const auto p = rabbit_order::compute_perm(g);
        const double tend = rabbit_order::now_sec();
        //--------------------------------------------
        const auto c = std::make_unique<rabbit_order::vint[]>(g.n());
        #pragma omp parallel for
        for (rabbit_order::vint v = 0; v < g.n(); ++v)
            c[v] = rabbit_order::trace_com(v, &g);

        // const double q = compute_modularity(adj, c.get());

        // Print the result
        // std::copy(&c[0], &c[g.n()],
        //           std::ostream_iterator<rabbit_order::vint>(std::cout, "\n"));

        // PrintTime("Modularity", q);
        //--------------------------------------------
        // std::cerr << "Permutation generation Time: "
        //           << rabbit_order::now_sec() - tstart << std::endl;
        PrintTime("Sub-RabbitOrder Map Time", tend - tstart);
        // Ensure new_ids is large enough to hold all new IDs

        if (new_ids.size() < g.n())
            new_ids.resize(g.n());

        #pragma omp parallel for
        for (size_t i = 0; i < g.n(); ++i)
        {
            new_ids[i] = (NodeID_)p[i];
        }
    }
#endif
    /*
       MIT License

       Copyright (c) 2016, Hao Wei.

       Permission is hereby granted, free of charge, to any person obtaining a
       copy of this software and associated documentation files (the "Software"),
       to deal in the Software without restriction, including without limitation
       the rights to use, copy, modify, merge, publish, distribute, sublicense,
       and/or sell copies of the Software, and to permit persons to whom the
       Software is furnished to do so, subject to the following conditions:

       The above copyright notice and this permission notice shall be included in
       all copies or substantial portions of the Software.

       THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
       IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
       FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
       AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
       LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
       FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
       DEALINGS IN THE SOFTWARE.
     */

    void GenerateGOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                               pvector<NodeID_> &new_ids)
    {

        int window = 5;

        // int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges();

        std::vector<std::pair<int, int>> edges;
        edges.reserve(num_edges * 2);

        for (NodeID_ i = 0; i < g.num_nodes(); i++)
        {
            for (DestID_ j : g.out_neigh(i))
            {
                if (g.is_weighted())
                    edges.push_back({i, static_cast<NodeWeight<NodeID_, WeightT_>>(j).v});
                else
                    edges.push_back({i, j});
            }
        }

        if (g.directed())
        {
            if (num_edges < g.num_edges_directed())
            {
                for (NodeID_ i = 0; i < g.num_nodes(); i++)
                {
                    for (DestID_ j : g.in_neigh(i))
                    {
                        if (g.is_weighted())
                            edges.push_back(
                        {i, static_cast<NodeWeight<NodeID_, WeightT_>>(j).v});
                        else
                            edges.push_back({i, j});
                    }
                }
            }
        }

        edges.shrink_to_fit();

        Gorder::GoGraph go;
        vector<int> order;
        Timer tm;
        std::string name;
        name = GorderUtil::extractFilename(cli_.filename().c_str());
        go.setFilename(name);

        tm.Start();
        go.readGraphEdgelist(edges, g.num_nodes());
        edges.clear();
        // go.readGraph(cli_.filename().c_str());
        go.Transform();
        tm.Stop();
        PrintTime("GOrder graph", tm.Seconds());

        tm.Start();
        go.GorderGreedy(order, window);
        tm.Stop();
        PrintTime("GOrder Map Time", tm.Seconds());

        if (new_ids.size() < (size_t)go.vsize)
            new_ids.resize(go.vsize);

        #pragma omp parallel for
        for (int i = 0; i < go.vsize; i++)
        {
            int u = order[go.order_l1[i]];
            new_ids[i] = (NodeID_)u;
        }
    }

    void GenerateRCMOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                 pvector<NodeID_> &new_ids)
    {

        // int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges();

        std::vector<std::pair<int, int>> edges;
        edges.reserve(num_edges * 2);

        for (NodeID_ i = 0; i < g.num_nodes(); i++)
        {
            for (DestID_ j : g.out_neigh(i))
            {
                if (g.is_weighted())
                    edges.push_back({i, static_cast<NodeWeight<NodeID_, WeightT_>>(j).v});
                else
                    edges.push_back({i, j});
            }
        }

        if (g.directed())
        {
            if (num_edges < g.num_edges_directed())
            {
                for (NodeID_ i = 0; i < g.num_nodes(); i++)
                {
                    for (DestID_ j : g.in_neigh(i))
                    {
                        if (g.is_weighted())
                            edges.push_back(
                        {i, static_cast<NodeWeight<NodeID_, WeightT_>>(j).v});
                        else
                            edges.push_back({i, j});
                    }
                }
            }
        }

        edges.shrink_to_fit();

        Gorder::GoGraph go;
        vector<int> order;
        Timer tm;
        std::string name;
        name = GorderUtil::extractFilename(cli_.filename().c_str());
        go.setFilename(name);

        tm.Start();
        go.readGraphEdgelist(edges, g.num_nodes());
        edges.clear();
        // go.readGraph(cli_.filename().c_str());
        go.Transform();
        tm.Stop();
        PrintTime("RCMOrder graph", tm.Seconds());

        tm.Start();
        go.RCMOrder(order);
        tm.Stop();
        PrintTime("RCMOrder Map Time", tm.Seconds());

        if (new_ids.size() < (size_t)go.vsize)
            new_ids.resize(go.vsize);

        #pragma omp parallel for
        for (int i = 0; i < go.vsize; i++)
        {
            int u = order[go.order_l1[i]];
            new_ids[i] = (NodeID_)u;
        }
    }

    // HELPERS
    // -------

    template <class G, class R>
    inline double getModularity(const G &x, const R &a, double M)
    {
        auto fc = [&](auto u)
        {
            return a.membership[u];
        };
        return modularityByOmp(x, fc, M, 1.0);
    }

    template <class K, class W>
    inline float refinementTime(const LouvainResult<K, W> &a)
    {
        return 0;
    }
    template <class K, class W>
    inline float refinementTime(const LeidenResult<K, W> &a)
    {
        return a.refinementTime;
    }

    // PERFORM EXPERIMENT
    // ------------------

    template <class G>
    void runExperiment(G &x, double resolution = 0.75, int maxIterations = 10,
                       int maxPasses = 10)
    {
        // using K = typename G::key_type;
        // using V = typename G::edge_value_type;
        Timer tm;
        std::random_device dev;
        std::default_random_engine rnd(dev());
        int repeat = REPEAT_METHOD;
        double M = edgeWeightOmp(x) / 2;
        // Follow a specific result logging format, which can be easily parsed
        // later.
        // auto flog = [&](const auto &ans, const char *technique) {
        //   printf("{%09.1fms, %09.1fms mark, %09.1fms init, %09.1fms "
        //          "firstpass,%09.1fms locmove, %09.1fms refine, %09.1fms aggr,
        //          %.3e " "aff,%04d iters, %03d passes, %01.9f modularity, "
        //          "%zu/%zudisconnected} %s\n",
        //          ans.time, ans.markingTime, ans.initializationTime,
        //          ans.firstPassTime, ans.localMoveTime, refinementTime(ans),
        //          ans.aggregationTime, double(ans.affectedVertices),
        //          ans.iterations, ans.passes, getModularity(x, ans, M),
        //          countValue(communitiesDisconnectedOmp(x, ans.membership),
        //          char(1)), communities(x, ans.membership).size(), technique);
        // };
        // Get community memberships on original graph (static).
        {
            // auto a0 = louvainStaticOmp(x, {repeat});
            // flog(a0, "louvainStaticOmp");
        }
        {

            tm.Start();

            // auto b0 = leidenStaticOmp<false, false>(rnd, x, {repeat});
            // flog(b0, "leidenStaticOmpGreedy");
            // auto b1 = leidenStaticOmp<false,  true>(rnd, x, {repeat});
            // flog(b1, "leidenStaticOmpGreedyOrg");
            // auto c0 = leidenStaticOmp<false, false>(
            //   rnd, x, {repeat, 0.5, 1e-12, 0.8, 1.0, 100, 100});
            // flog(c0, "leidenStaticOmpGreedyMedium");
            auto c1 = leidenStaticOmp<false, false>(
                          rnd, x,
            {repeat, resolution, 1e-12, 0.8, 1.0, maxIterations, maxPasses});
            tm.Stop();
            PrintTime("Modularity", getModularity(x, c1, M));
            PrintTime("LeidenOrder Map Time", tm.Seconds());

            // flog(c1, "leidenStaticOmpGreedyMediumOrg");
            // auto d0 = leidenStaticOmp<false, false>(rnd, x, {repeat, 1.0,
            // 1e-12, 1.0, 1.0, 100, 100}); flog(d0, "leidenStaticOmpGreedyHeavy");
            // auto d1 = leidenStaticOmp<false,  true>(rnd, x, {repeat, 1.0,
            // 1e-12, 1.0, 1.0, 100, 100}); flog(d1, "leidenStaticOmpGreedyHeavyOrg");
        }
        {
            // auto b2 = leidenStaticOmp<true, false>(rnd, x, {repeat});
            // flog(b2, "leidenStaticOmpRandom");
            // auto b3 = leidenStaticOmp<true,  true>(rnd, x, {repeat});
            // flog(b3, "leidenStaticOmpRandomOrg");
            // auto c2 = leidenStaticOmp<true, false>(rnd, x, {repeat, 1.0, 1e-12,
            // 0.8, 1.0, 100, 100}); flog(c2, "leidenStaticOmpRandomMedium"); auto c3
            // = leidenStaticOmp<true,  true>(rnd, x, {repeat, 1.0, 1e-12, 0.8, 1.0,
            // 100, 100}); flog(c3, "leidenStaticOmpRandomMediumOrg"); auto d2 =
            // leidenStaticOmp<true, false>(rnd, x, {repeat, 1.0, 1e-12, 1.0, 1.0,
            // 100, 100}); flog(d2, "leidenStaticOmpRandomHeavy"); auto d3 =
            // leidenStaticOmp<true,  true>(rnd, x, {repeat, 1.0, 1e-12, 1.0, 1.0,
            // 100, 100}); flog(d3, "leidenStaticOmpRandomHeavyOrg");
        }
    }

    using K = uint32_t;

    void sort_by_vector_element(
        std::vector<std::vector<K>> &communityVectorTuplePerPass,
        size_t element_index)
    {
        __gnu_parallel::stable_sort(
            communityVectorTuplePerPass.begin(), communityVectorTuplePerPass.end(),
            [&](const std::vector<K> &a, const std::vector<K> &b)
        {
            return a[element_index] < b[element_index];
        });
    }

    void GenerateLeidenMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                               pvector<NodeID_> &new_ids,
                               std::vector<std::string> reordering_options)
    {

        Timer tm;

        // std::cout << "Options: ";
        // for (const auto& param : reordering_options) {
        //   std::cout << param << " ";
        // }
        // std::cout << std::endl;

        using V = TYPE;
        install_sigsegv();

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges();

        std::vector<std::tuple<size_t, size_t, double>> edges;
        edges.reserve(num_edges * 2);

        for (NodeID_ i = 0; i < g.num_nodes(); i++)
        {
            for (DestID_ j : g.out_neigh(i))
            {
                if (g.is_weighted())
                    edges.push_back({i, static_cast<NodeWeight<NodeID_, WeightT_>>(j).v,
                                     static_cast<NodeWeight<NodeID_, WeightT_>>(j).w});
                else
                    edges.push_back({i, j, 1.0f});
            }
        }

        if (g.directed())
        {
            if (num_edges < g.num_edges_directed())
            {
                for (NodeID_ i = 0; i < g.num_nodes(); i++)
                {
                    for (DestID_ j : g.in_neigh(i))
                    {
                        if (g.is_weighted())
                            edges.push_back(
                        {
                            i, static_cast<NodeWeight<NodeID_, WeightT_>>(j).v,
                                    static_cast<NodeWeight<NodeID_, WeightT_>>(j).w});
                        else
                            edges.push_back({i, j, 1.0f});
                    }
                }
            }
        }

        edges.shrink_to_fit();

        tm.Start();
        bool symmetric = false;
        bool weighted = g.is_weighted();
        DiGraph<K, None, V> x;
        readVecOmpW(x, edges, num_nodes, symmetric,
                    weighted); // LOG(""); println(x);
        edges.clear();
        if (!symmetric)
        {
            x = symmetricizeOmp(x);
        } //; LOG(""); print(x); printf(" (->symmetricize)\n"); }
        tm.Stop();
        PrintTime("DiGraph graph", tm.Seconds());

        double resolution = 0.8;
        int maxIterations = 30;
        /** Maximum number of passes [10]. */
        int maxPasses = 30;

        if (!reordering_options.empty())
        {
            resolution = std::stod(reordering_options[0]);
            resolution = (resolution > 1) ? 0.8 : resolution;
        }
        if (reordering_options.size() > 1)
        {
            maxIterations = std::stoi(reordering_options[1]);
        }
        if (reordering_options.size() > 2)
        {
            maxPasses = std::stoi(reordering_options[2]);
        }

        runExperiment(x, resolution, maxIterations, maxPasses);

        size_t num_nodesx;
        size_t num_passes;
        num_nodesx = x.span();
        num_passes = x.communityMappingPerPass.size() + 3;

        std::vector<std::vector<K>> communityVectorTuplePerPass(
                                     num_nodesx, std::vector<K>(num_passes, 0));
        // // Initialize each inner vector
        tm.Start();
        #pragma omp parallel for
        for (size_t i = 0; i < num_nodesx; ++i)
        {
            communityVectorTuplePerPass[i][0] = UINT_E_MAX;
            communityVectorTuplePerPass[i][1] = i;
            communityVectorTuplePerPass[i][2] = x.degree(i);
        }

        for (size_t i = 0; i < num_passes - 3; ++i)
        {
            #pragma omp parallel for
            for (size_t j = 0; j < num_nodesx; ++j)
            {
                communityVectorTuplePerPass[j][3 + i] = x.communityMappingPerPass[i][j];
            }
        }

        // sort_by_vector_element(communityVectorTuplePerPass, 2);
        sort_by_vector_element(communityVectorTuplePerPass, num_passes - 1);

        // auto running_v_id = 0;
        // auto running_v_hops = 0;
        // int64_t avgDegree = g.num_edges() / g.num_nodes() + 1;

        // for (int64_t i = 0; i < num_nodes; i++)
        // {
        //     if (communityVectorTuplePerPass[i][0] == UINT_E_MAX)
        //     {
        //         auto current_com_id = communityVectorTuplePerPass[i][num_passes - 1];
        //         communityVectorTuplePerPass[i][0] = running_v_id;
        //         running_v_id++;
        //         auto current_v_id = communityVectorTuplePerPass[i][1];
        //         for (int64_t j = (i + 1); j < num_nodes; j++)
        //         {
        //             auto next_com_id = communityVectorTuplePerPass[j][num_passes - 1];
        //             if (current_com_id != next_com_id ||
        //                     (running_v_hops % avgDegree) == 0)
        //             {
        //                 running_v_hops = 0;
        //                 break;
        //             }
        //             auto set_v_id = communityVectorTuplePerPass[j][1];
        //             if (communityVectorTuplePerPass[j][0] == UINT_E_MAX)
        //             {
        //                 if (g.out_neigh(current_v_id).contains(set_v_id))
        //                 {
        //                     communityVectorTuplePerPass[j][0] = running_v_id;
        //                     running_v_id++;
        //                     running_v_hops++;
        //                 }
        //             }
        //         }
        //     }
        // }

        // sort_by_vector_element(communityVectorTuplePerPass, 0);

        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; i++)
        {
            new_ids[communityVectorTuplePerPass[i][1]] = (NodeID_)i;
        }

        tm.Stop();
        PrintTime("GenID Time", tm.Seconds());
        PrintTime("Num Passes", x.communityMappingPerPass.size());
        PrintTime("Resolution", resolution);
    }

    void GenerateGraphBrewMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                  pvector<NodeID_> &new_ids, bool useOutdeg,
                                  std::vector<std::string> reordering_options, int numLevels = 1,
                                  bool recursion = false)
    {

        Timer tm;

        // std::cout << "Options: ";
        // for (const auto& param : reordering_options) {
        //   std::cout << param << " ";
        // }
        // std::cout << std::endl;

        using V = TYPE;

        install_sigsegv();

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges_directed();
        size_t num_nodesx = num_nodes;
        size_t num_passes = 0;
        double resolution = 0.8;
        int maxIterations = 30;
        /** Maximum number of passes [10]. */
        int maxPasses = 30;
        // Define the frequency threshold
        size_t frequency_threshold = 10; // Set your frequency threshold here

        const char *reorderingAlgo_arg =
            "8"; // This can be any string representing a valid integer
        ReorderingAlgo algo = getReorderingAlgo(reorderingAlgo_arg);


        if (!reordering_options.empty())
        {
            frequency_threshold = std::stoi(reordering_options[0]);
        }
        if (reordering_options.size() > 1)
        {
            algo = getReorderingAlgo(reordering_options[1].c_str());
        }
        if (reordering_options.size() > 2)
        {
            resolution = std::stod(reordering_options[2]);
            resolution = (resolution > 1) ? 0.75 : resolution;
        }
        if (reordering_options.size() > 3)
        {
            maxIterations = std::stoi(reordering_options[3]);
        }
        if (reordering_options.size() > 4)
        {
            maxPasses = std::stoi(reordering_options[4]);
        }

        if(recursion && numLevels > 0)
            numLevels -= 1;

        vector<vector<K>> communityMappingPerPass;
        std::vector<size_t> comm_ids(num_nodes, 0);
        {
            std::vector<std::tuple<size_t, size_t, double>> edges(num_edges);
            edges.reserve(num_edges);
            // Parallel loop to construct the edge list
            #pragma omp parallel for
            for (NodeID_ i = 0; i < num_nodes; ++i)
            {
                NodeID_ out_start = g.out_offset(i);

                NodeID_ j = 0;
                for (DestID_ neighbor : g.out_neigh(i))
                {
                    if (g.is_weighted())
                    {
                        NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                        WeightT_ weight =
                            static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;

                        std::tuple<size_t, size_t, double> edge =
                            std::make_tuple(i, dest, weight);
                        edges[out_start + j] = edge;
                    }
                    else
                    {
                        std::tuple<size_t, size_t, double> edge =
                            std::make_tuple(i, neighbor, 1.0f);
                        edges[out_start + j] = edge;
                    }
                    ++j;
                }
            }

            tm.Start();
            bool symmetric = false;
            bool weighted = g.is_weighted();
            DiGraph<K, None, V> x;
            readVecOmpW(x, edges, num_nodes, symmetric,
                        weighted); // LOG(""); println(x);
            edges.clear();
            x = symmetricizeOmp(x);
            num_nodesx = x.span();

            tm.Stop();
            PrintTime("DiGraph graph", tm.Seconds());
            runExperiment(x, resolution, maxIterations, maxPasses);
            x.passAndClear(communityMappingPerPass);
            num_passes = communityMappingPerPass.size() + 2;
        }

        size_t num_comm;

        std::vector<std::vector<K>> communityVectorTuplePerPass(
                                     num_nodesx, std::vector<K>(num_passes, 0));
        // // Initialize each inner vector
        tm.Start();
        #pragma omp parallel for
        for (size_t i = 0; i < num_nodesx; ++i)
        {
            communityVectorTuplePerPass[i][0] = i;
            communityVectorTuplePerPass[i][1] = g.out_degree(i);
        }

        for (size_t i = 0; i < num_passes - 2; ++i)
        {
            #pragma omp parallel for
            for (size_t j = 0; j < num_nodesx; ++j)
            {
                communityVectorTuplePerPass[j][2 + i] = communityMappingPerPass[i][j];
            }
        }

        // sort_by_vector_element(communityVectorTuplePerPass, 2);
        sort_by_vector_element(communityVectorTuplePerPass, num_passes - 1);

        #pragma omp parallel for
        for (size_t i = 0; i < num_nodesx; ++i)
        {
            comm_ids[i] = communityVectorTuplePerPass[i][num_passes - 1];
        }

        std::cout << std::endl;
        for (size_t i = 0; i < num_nodesx; ++i)
        {
            std::cout <<  i << "->" << comm_ids[i] << std::endl;
        }
        std::cout << std::endl;

        communityMappingPerPass.clear();
        communityVectorTuplePerPass.clear();

        num_comm = *(__gnu_parallel::max_element(comm_ids.begin(), comm_ids.end()));

        tm.Stop();
        // Create the frequency array
        std::vector<size_t> frequency_array_pre(num_comm + 1,
                                                0); // +1 to include num_comm index
        std::vector<size_t> frequency_array(num_comm + 1,
                                            0); // +1 to include num_comm index

        // Fill the frequency array
        #pragma omp parallel for
        for (size_t i = 0; i < comm_ids.size(); ++i)
        {
            #pragma omp atomic
            ++frequency_array_pre[comm_ids[i]];
        }

        // Find the community ID with the minimum frequency
        size_t min_freq_comm_id =
            __gnu_parallel::min_element(frequency_array_pre.begin(),
                                        frequency_array_pre.end()) -
            frequency_array_pre.begin();

        // Reassign nodes with frequency below the threshold

        if (!frequency_array_pre.empty())
        {
            std::vector<size_t> sorted_freq_array = frequency_array_pre;
            std::sort(sorted_freq_array.begin(), sorted_freq_array.end(), std::greater<size_t>());

            if (frequency_threshold > 0 && frequency_threshold <= sorted_freq_array.size())
            {
                frequency_threshold = sorted_freq_array[frequency_threshold - 1];
            }
            else
            {
                frequency_threshold = sorted_freq_array.back();
            }
        }

        #pragma omp parallel for
        for (size_t i = 0; i < comm_ids.size(); ++i)
        {
            if (frequency_array_pre[comm_ids[i]] < frequency_threshold)
            {
                comm_ids[i] = min_freq_comm_id;
            }
        }

        // Fill the frequency array
        #pragma omp parallel for
        for (size_t i = 0; i < comm_ids.size(); ++i)
        {
            #pragma omp atomic
            ++frequency_array[comm_ids[i]];
        }

        // Create a vector of pairs (frequency, community ID) for sorting
        std::vector<std::pair<size_t, size_t>> freq_comm_pairs;
        for (size_t i = 0; i < frequency_array.size(); ++i)
        {
            if (frequency_array[i] >= 1)
                freq_comm_pairs.emplace_back(frequency_array[i], i);
        }

        // Sort the pairs by frequency in descending order
        __gnu_parallel::sort(freq_comm_pairs.begin(), freq_comm_pairs.end(),
                             std::greater<>());

        // Get the top ten community IDs
        size_t top_comms_size = freq_comm_pairs.size();
        std::vector<size_t> top_communities(top_comms_size, 0);

        for (size_t i = 0; i < top_communities.size(); ++i)
        {
            top_communities[i] = freq_comm_pairs[i].second;
        }

        // Print the top ten frequencies
        std::cout << "Top community frequencies:\n";
        for (size_t i = 0; i < top_communities.size(); ++i)
        {
            std::cout << "Community ID " << top_communities[i] << " has frequency "
                      << freq_comm_pairs[i].first << ".\n";
        }

        // Create a set of top communities for quick lookup
        std::unordered_set<size_t> top_communities_set(top_communities.begin(),
                top_communities.end());

        // Create thread-private edge lists for each community
        std::vector<std::unordered_map<size_t, EdgeList>> thread_edge_lists(
                    omp_get_max_threads());

        // Loop through the original graph and add edges to the appropriate community
        // edge list
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto &local_edge_list = thread_edge_lists[tid];

            #pragma omp for nowait
            for (NodeID_ i = 0; i < num_nodes; ++i)
            {
                size_t src_comm_id = comm_ids[i];
                if (top_communities_set.find(src_comm_id) !=
                        top_communities_set.end())
                {
                    for (DestID_ neighbor : g.out_neigh(i))
                    {
                        if (g.is_weighted())
                        {
                            NodeID_ dest =
                                static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                            WeightT_ weight =
                                static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;
                            // size_t dst_comm_id = comm_ids[neighbor];
                            // if (src_comm_id == dst_comm_id)
                            // {
                            Edge e = Edge(i, NodeWeight<NodeID_, WeightT_>(dest, weight));
                            local_edge_list[src_comm_id].push_back(e);
                            // }
                        }
                        else
                        {
                            // size_t dst_comm_id = comm_ids[neighbor];
                            // if (src_comm_id == dst_comm_id)
                            // {
                            Edge e = Edge(i, neighbor);
                            local_edge_list[src_comm_id].push_back(e);
                            // }
                        }
                    }
                }
            }
        }

        // Merge thread-private edge lists into the final community edge lists
        std::unordered_map<size_t, EdgeList> community_edge_lists;

        for (const auto &local_edge_list : thread_edge_lists)
        {
            for (const auto &entry : local_edge_list)
            {
                auto &comm_edge_list = community_edge_lists[entry.first];
                comm_edge_list.insert(comm_edge_list.end(), entry.second.begin(),
                                      entry.second.end());
            }
        }

        // Print the segmented edge lists (optional)
        for (const auto &[comm_id, edge_list] : community_edge_lists)
        {
            std::cout << "Community ID " << comm_id
                      << " edge list: " << edge_list.size() << "\n";
            // for (const auto &edge : edge_list)
            // {
            //     std::cout << "(" << std::get<0>(edge) << ", " << std::get<1>(edge)
            //     << ", " << std::get<2>(edge) << ")\n";
            // }
        }

        // Create a vector to store the new_ids vectors for each community
        std::unordered_map<size_t, std::vector<std::pair<size_t, NodeID_>>>
        community_id_mappings;

        // Call GenerateRabbitOrderMappingEdglist for each top community edge list
        // and store the new_ids result void GenerateMappingLocalEdgelist(const
        // EdgeList & el,
        //                                   pvector<NodeID_> &new_ids,
        //                                   ReorderingAlgo reordering_algo, bool
        //                                   useOutdeg, std::vector<std::string>
        //                                   reordering_options)
        // #pragma omp parallel for
        // enum ReorderingAlgo
        // {
        //     ORIGINAL = 0,
        //     Random = 1,
        //     Sort = 2,
        //     HubSort = 3,
        //     HubCluster = 4,
        //     DBG = 5,
        //     HubSortDBG = 6,
        //     HubClusterDBG = 7,
        //     RabbitOrder = 8,
        //     GOrder = 9,
        //     COrder = 10,
        //     RCMOrder = 11,
        //     LeidenOrder = 12,
        //     GraphBrewOrder = 13,
        //     MAP = 14,
        // };
        for (size_t idx = 0; idx < top_communities.size(); ++idx)
        {
            size_t comm_id = top_communities[idx];
            auto &edge_list = community_edge_lists[comm_id];
            pvector<NodeID_> new_ids_sub(num_nodes, -1);
            // GenerateRabbitOrderMappingEdgelist(edge_list, new_ids_sub);

            // double modularity =
            //     GenerateRabbitModularityEdgelist(edge_list, g.is_weighted());

            // PrintTime("Sub-Modularity", modularity);
            std::cout << "Community ID " << comm_id
                      << " edge list: " << edge_list.size() << "\n";

            // if (numLevels && top_communities.size() > 1 && edge_list.size() > 1) {
            //   algo = ReorderingAlgo::GraphBrewOrder;
            //   reordering_options[0] = std::to_string(static_cast<int>(frequency_threshold * 1.5));
            // } else {
            //   algo = ReorderingAlgo::RabbitOrder;
            // }

            GenerateMappingLocalEdgelist(edge_list, new_ids_sub, algo, true,
                                         reordering_options, numLevels, true);
            // Add id pairs to the corresponding community list
            #pragma omp parallel for
            for (size_t i = 0; i < new_ids_sub.size(); ++i)
            {
                size_t src_comm_id = comm_ids[i];
                if (new_ids_sub[i] != -1 && comm_id == src_comm_id)
                {
                    #pragma omp critical
                    community_id_mappings[src_comm_id].emplace_back(i, new_ids_sub[i]);
                }
            }
        }

        // Make the mapping consecutive within each community list
        for (auto &entry : community_id_mappings)
        {
            auto &id_pairs = entry.second;
            __gnu_parallel::sort(id_pairs.begin(), id_pairs.end(),
                                 [](const auto & a, const auto & b)
            {
                return static_cast<unsigned>(a.second) <
                       static_cast<unsigned>(b.second);
            });

            #pragma omp parallel for
            for (size_t i = 0; i < id_pairs.size(); ++i)
            {
                id_pairs[i].second = i;
            }
        }

        // Calculate the total size of all pairs
        size_t total_size = 0;
        for (const auto &entry : community_id_mappings)
        {
            total_size += entry.second.size();
        }

        // Create all_pairs with the total size
        std::vector<std::pair<size_t, NodeID_>> all_pairs(total_size);

        // Assign the pairs to all_pairs and update the community_id_mappings with
        // new indices
        size_t index = 0;
        for (auto &entry : community_id_mappings)
        {
            #pragma omp parallel for
            for (size_t i = 0; i < entry.second.size(); ++i)
            {
                entry.second[i].second = index + i;
                all_pairs[index + i] = entry.second[i];
            }
            index += entry.second.size();
        }

        // Sort all pairs by new IDs to make the mapping consecutive across all
        // lists
        __gnu_parallel::sort(all_pairs.begin(), all_pairs.end(),
                             [](const auto & a, const auto & b)
        {
            return static_cast<unsigned>(a.first) <
                   static_cast<unsigned>(b.first);
        });

        #pragma omp parallel for
        for (size_t i = 0; i < all_pairs.size(); i++)
        {
            new_ids[all_pairs[i].first] = (NodeID_)all_pairs[i].second;
        }

        std::cout << std::endl;
        for (size_t i = 0; i < new_ids.size(); ++i)
        {
            std::cout <<  i << "->" << new_ids[i] << std::endl;
        }
        std::cout << std::endl;

        PrintTime("GenID Time", tm.Seconds());
        PrintTime("Num Passes", num_passes);
        PrintTime("Num Comm", num_comm);
        PrintTime("Resolution", resolution);
    }

    void GenerateLeidenFullMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                   pvector<NodeID_> &new_ids)
    {

        Timer tm;
        Timer tm2;

        using V = TYPE;
        install_sigsegv();

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges();

        std::vector<std::tuple<size_t, size_t, double>> edges;
        edges.reserve(num_edges * 2);

        for (NodeID_ i = 0; i < g.num_nodes(); i++)
        {
            for (DestID_ j : g.out_neigh(i))
            {
                if (g.is_weighted())
                    edges.push_back({i, static_cast<NodeWeight<NodeID_, WeightT_>>(j).v,
                                     static_cast<NodeWeight<NodeID_, WeightT_>>(j).w});
                else
                    edges.push_back({i, j, 1.0f});
            }
        }

        if (g.directed())
        {
            if (num_edges < g.num_edges_directed())
            {
                for (NodeID_ i = 0; i < g.num_nodes(); i++)
                {
                    for (DestID_ j : g.in_neigh(i))
                    {
                        if (g.is_weighted())
                            edges.push_back(
                        {
                            i, static_cast<NodeWeight<NodeID_, WeightT_>>(j).v,
                                    static_cast<NodeWeight<NodeID_, WeightT_>>(j).w});
                        else
                            edges.push_back({i, j, 1.0f});
                    }
                }
            }
        }

        edges.shrink_to_fit();

        tm.Start();
        bool symmetric = false;
        bool weighted = g.is_weighted();
        DiGraph<K, None, V> x;
        readVecOmpW(x, edges, num_nodes, symmetric,
                    weighted); // LOG(""); println(x);
        edges.clear();
        if (!symmetric)
        {
            x = symmetricizeOmp(x);
        } //; LOG(""); print(x); printf(" (->symmetricize)\n"); }
        tm.Stop();
        PrintTime("DiGraph graph", tm.Seconds());

        tm.Start();
        runExperiment(x);
        tm.Stop();

        size_t num_nodesx;
        size_t num_passes;
        num_nodesx = x.span();
        num_passes = x.communityMappingPerPass.size() + 2;

        std::vector<std::vector<K>> communityVectorTuplePerPass(
                                     num_nodesx, std::vector<K>(num_passes, 0));
        // // Initialize each inner vector
        // tm.Start();
        #pragma omp parallel for
        for (size_t i = 0; i < num_nodesx; ++i)
        {
            communityVectorTuplePerPass[i][0] = i;
            communityVectorTuplePerPass[i][1] = x.degree(i);
        }

        for (size_t i = 0; i < num_passes - 2; ++i)
        {
            #pragma omp parallel for
            for (size_t j = 0; j < num_nodesx; ++j)
            {
                communityVectorTuplePerPass[j][2 + i] = x.communityMappingPerPass[i][j];
            }
        }

        // sort_by_vector_element(communityVectorTuplePerPass,num_passes-1);

        for (size_t i = 1; i < num_passes; ++i)
        {
            sort_by_vector_element(communityVectorTuplePerPass, i);
        }

        pvector<NodeID_> interim_ids(num_nodes, -1);

        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; i++)
        {
            interim_ids[communityVectorTuplePerPass[i][0]] = (NodeID_)i;
        }

        tm2.Start();
        GenerateDBGMappingInterim(g, new_ids, interim_ids, true);
        tm2.Stop();

        // tm.Stop();
        PrintTime("LeidenFullOrder Map Time", tm.Seconds() + tm2.Seconds());
    }

    void GenerateDBGMappingInterim(const CSRGraph<NodeID_, DestID_, invert> &g,
                                   pvector<NodeID_> &new_ids,
                                   pvector<NodeID_> &interim_ids,
                                   bool useOutdeg)
    {
        // Timer t;
        // t.Start();

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges();

        pvector<NodeID_> interim_ids_inv(num_nodes, -1);
        pvector<NodeID_> new_ids_interim(num_nodes, -1);

        #pragma omp parallel for
        for (NodeID_ n = 0; n < num_nodes_; n++)
        {
            // assert(interim_ids_inv[interim_ids[n]] == -1);
            interim_ids_inv[interim_ids[n]] = n;
            // std::cout << "Node " << n << " | interim_ids: " << interim_ids[n] << "
            // | interim_ids_inv: " << interim_ids_inv[interim_ids[n]] << std::endl;
        }

        uint32_t avg_vertex = num_edges / num_nodes;
        const uint32_t &av = avg_vertex;

        uint32_t bucket_threshold[] =
        {
            av / 2,   av,       av * 2,   av * 4,
            av * 8,   av * 16,  av * 32,  av * 64,
            av * 128, av * 256, av * 512, static_cast<uint32_t>(-1)
        };
        int num_buckets = 11;
        if (num_buckets > 11)
        {
            // if you really want to increase the bucket count, add more thresholds to
            // the bucket_threshold above.
            std::cout << "Unsupported bucket size: " << num_buckets << std::endl;
            assert(0);
        }
        bucket_threshold[num_buckets - 1] = static_cast<uint32_t>(-1);

        vector<uint32_t> bucket_vertices[num_buckets];
        const int num_threads = omp_get_max_threads();
        vector<uint32_t> local_buckets[num_threads][num_buckets];

        if (useOutdeg)
        {
            // This loop relies on a static scheduling
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < num_nodes; i++)
            {
                for (int j = 0; j < num_buckets; j++)
                {
                    const int64_t &count = g.out_degree(interim_ids_inv[i]);
                    if (count <= bucket_threshold[j])
                    {
                        local_buckets[omp_get_thread_num()][j].push_back(i);
                        break;
                    }
                }
            }
        }
        else
        {
            #pragma omp parallel for schedule(static)
            for (int64_t i = 0; i < num_nodes; i++)
            {
                for (int j = 0; j < num_buckets; j++)
                {
                    const int64_t &count = g.in_degree(interim_ids_inv[i]);
                    if (count <= bucket_threshold[j])
                    {
                        local_buckets[omp_get_thread_num()][j].push_back(i);
                        break;
                    }
                }
            }
        }

        int temp_k = 0;
        uint32_t start_k[num_threads][num_buckets];
        for (int32_t j = num_buckets - 1; j >= 0; j--)
        {
            for (int t = 0; t < num_threads; t++)
            {
                start_k[t][j] = temp_k;
                temp_k += local_buckets[t][j].size();
            }
        }

        #pragma omp parallel for schedule(static)
        for (int t = 0; t < num_threads; t++)
        {
            for (int j = num_buckets - 1; j >= 0; j--)
            {
                const vector<uint32_t> &current_bucket = local_buckets[t][j];
                int k = start_k[t][j];
                const size_t &size = current_bucket.size();
                for (uint32_t i = 0; i < size; i++)
                {
                    new_ids_interim[current_bucket[i]] = k++;
                }
            }
        }

        for (int i = 0; i < num_threads; i++)
        {
            for (int j = 0; j < num_buckets; j++)
            {
                local_buckets[i][j].clear();
            }
        }

        #pragma omp parallel for
        for (NodeID_ n = 0; n < num_nodes_; n++)
        {
            new_ids[n] = new_ids_interim[interim_ids[n]];
            // std::cout << "Node " << n << " | new_ids: " << new_ids[n] << " |
            // new_ids_interim: " << new_ids_interim[interim_ids[n]] << std::endl;
        }

        // t.Stop();
        // PrintTime("DBG Map Time", t.Seconds());
    }
};

#endif // BUILDER_H_
