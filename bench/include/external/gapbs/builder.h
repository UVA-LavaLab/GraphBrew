// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef BUILDER_H_
#define BUILDER_H_

/**
 * @file builder.h
 * @brief Graph construction and reordering framework for GAP Benchmark Suite
 * 
 * This file provides the BuilderBase class which handles:
 * - Graph construction from edge lists (file or synthetic)
 * - Graph reordering using various algorithms
 * - CSR format manipulation and transformations
 * 
 * ARCHITECTURE:
 * - Core graph operations: MakeGraph(), MakeGraphFromEL(), SquishCSR()
 * - Reordering dispatch: GenerateMapping() - main entry point
 * - Algorithm delegates: Each Generate*Mapping() calls reorder/*.h implementations
 * 
 * REORDERING ALGORITHMS (see reorder/*.h for implementations):
 * - Basic (0-2): Original, Random, Sort
 * - Hub (3-7): HubSort, HubCluster, DBG variants
 * - RabbitOrder (8): Community-aware hierarchical clustering
 * - Classic (9-11): GOrder, COrder, RCMOrder
 * - GraphBrew (12): Multi-level community-based reordering
 * - Adaptive (14): ML-based per-community algorithm selection
 * - Leiden (15-17): Community detection variants
 * 
 * Author: Scott Beamer (original), GraphBrew Team (extensions)
 */

// ============================================================================
// STANDARD LIBRARY INCLUDES
// ============================================================================

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <mutex>
#include <numeric>
#include <parallel/algorithm>
#include <parallel/numeric>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ============================================================================
// LOCAL INCLUDES
// ============================================================================

#include "command_line.h"
#include "generator.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "reader.h"
#include "sliding_queue.h"
#include "timer.h"
#include "util.h"

// Unified reorder header - provides all algorithm implementations
#include <reorder/reorder.h>

// Cagra/GraphIT partitioning helpers (P-OPT)
#include <partition/cagra/popt.h>

// Graph partitioning (TRUST algorithm)
#include <partition/trust.h>

// ============================================================================
// EXTERNAL LIBRARY INCLUDES
// ============================================================================

#ifdef RABBIT_ENABLE
#include <rabbit/edge_list.hpp>
#include <rabbit/rabbit_order.hpp>
#include <boost/range/adaptor/transformed.hpp>
#include <boost/range/algorithm/count.hpp>
using namespace edge_list;
#endif

// GOrder includes (MIT License - Hao Wei, 2016)
#include <gorder/GoGraph.h>
#include <gorder/GoUtil.h>

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
#include <corder/vec2d.h>

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

// ============================================================================
// UNIFIED LEIDEN DEFAULTS - For fair comparison across all Leiden algorithms
// ============================================================================
#ifndef LEIDEN_DEFAULT_ITERATIONS
/** Number of iterations per Leiden pass - controls local move refinement */
#define LEIDEN_DEFAULT_ITERATIONS 20
#endif
#ifndef LEIDEN_DEFAULT_PASSES
/** Number of Leiden passes - controls coarsening depth */
#define LEIDEN_DEFAULT_PASSES 10
#endif

#include <leiden/main.hxx>

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

    NodeID_ FindMinNodeID(const EdgeList &el)
    {
        NodeID_ min_seen = FindMaxNodeID(el);
        #pragma omp parallel for reduction(min : min_seen)
        for (auto it = el.begin(); it < el.end(); it++)
        {
            Edge e = *it;
            min_seen = __gnu_parallel::min(min_seen, e.u);
            min_seen = __gnu_parallel::min(min_seen, (NodeID_)e.v);
        }
        return min_seen;
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
            if(!cli_.keep_self())
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

    pvector<NodeID_> CountLocalDegrees(const EdgeList &el, bool transpose, int64_t num_nodes_local = -1)
    {
        pvector<NodeID_> degrees(num_nodes_local, 0);
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
        int64_t num_nodes_local = FindMaxNodeID(el) + 1;
        pvector<NodeID_> degrees = CountLocalDegrees(el, transpose, num_nodes_local);
        pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
        *neighs = new DestID_[offsets[num_nodes_local]];
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

    CSRGraph<NodeID_, DestID_, invert> MakeLocalGraphFromEL(EdgeList &el, bool verbose = false)
    {
        DestID_ **index = nullptr;
        // **inv_index = nullptr;
        DestID_ *neighs = nullptr;
        // *inv_neighs = nullptr;
        Timer t;
        t.Start();

        int64_t num_nodes_local = FindMaxNodeID(el) + 1;

        MakeLocalCSR(el, false, &index, &neighs);
        // MakeLocalCSR(el, true, &inv_index, &inv_neighs);
        // CSRGraph<NodeID_, DestID_, invert> g = CSRGraph<NodeID_, DestID_, invert>(
        //         num_nodes_, index, neighs, inv_index, inv_neighs);
        CSRGraph<NodeID_, DestID_, invert> g = CSRGraph<NodeID_, DestID_,
                                           invert>(num_nodes_local, index, neighs);

        g = SquishGraph(g);
        // SquishCSR(g, false, &index, &neighs);
        // SquishCSR(g, true, &inv_index, &inv_neighs);
        t.Stop();
        if (verbose) {
            PrintTime("Local Build Time", t.Seconds());
        }
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
    MakeUniDirectELFromGraph(const CSRGraph<NodeID_, DestID_, invert> &g, NodeID_ min_seen = 0)
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
                if (g.is_weighted())
                {
                    NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v - min_seen;
                    WeightT_ weight =
                        static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;
                    el[out_start + j] = Edge(i - min_seen, NodeWeight<NodeID_, WeightT_>(dest, weight));
                    el[in_start + j] =
                        Edge(dest, NodeWeight<NodeID_, WeightT_>(i - min_seen, weight));
                }
                else
                {
                    el[out_start + j] = Edge(i - min_seen, neighbor - min_seen);
                    el[in_start + j] = Edge(neighbor - min_seen, i - min_seen);
                }

                ++j;
            }
        }

        // PrintEdgeList(el);

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
     * TRUST partitioning method based on the input type.
     *
     * For TRUST partitioning, uses TrustPartitioner from partition/trust.h
     * For Cagra partitioning, uses MakeCagraPartitionedGraph from cache/popt.h
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
            partitions = ::MakeCagraPartitionedGraph<NodeID_, DestID_, invert>(
                g, p_n, p_m, cli_.use_out_degree());
            break;
        case 1: // <1:TRUST>
            {
                // Use standalone TrustPartitioner from partition/trust.h
                TrustPartitioner<NodeID_, DestID_, WeightT_, invert> trustPartitioner(cli_.logging_en());
                partitions = trustPartitioner.MakeTrustPartitionedGraph(g, p_n, p_m);
            }
            break;
        default:
            partitions = ::MakeCagraPartitionedGraph<NodeID_, DestID_, invert>(
                g, p_n, p_m, cli_.use_out_degree());
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
        
        // Compute and print global graph topology features ONCE before any reordering
        // This provides clustering_coeff, avg_path_length, diameter, community_count
        // for the Python training scripts to use
        ComputeAndPrintGlobalTopologyFeatures(g_final);
        
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

    /**
     * Convert ReorderingAlgo to string.
     * Delegates to the global ::ReorderingAlgoStr in reorder/reorder.h.
     */
    const std::string ReorderingAlgoStr(ReorderingAlgo type)
    {
        return ::ReorderingAlgoStr(type);
    }
    
    /**
     * Apply a basic reordering algorithm to a subgraph.
     * 
     * This is a helper for per-community reordering that handles the common
     * basic algorithms. It does NOT handle complex algorithms like Leiden,
     * GraphBrew, or AdaptiveOrder (use GenerateMappingLocalEdgelist for those).
     * 
     * @param sub_g The subgraph to reorder
     * @param sub_new_ids Output mapping (must be pre-sized to sub_g.num_nodes())
     * @param algo Algorithm to apply
     * @param useOutdeg Whether to use out-degree (true) or in-degree (false)
     */
    void ApplyBasicReordering(CSRGraph<NodeID_, DestID_, invert>& sub_g,
                              pvector<NodeID_>& sub_new_ids,
                              ReorderingAlgo algo,
                              bool useOutdeg) {
        // Delegate to standalone function in reorder_types.h
        ::ApplyBasicReorderingStandalone<NodeID_, DestID_, WeightT_, invert>(
            sub_g, sub_new_ids, algo, useOutdeg, cli_.filename());
    }
    
    /**
     * Reorder a community's nodes and assign global IDs.
     * 
     * This helper encapsulates the common pattern:
     * 1. Build global-to-local / local-to-global mappings
     * 2. Create edge list for the induced subgraph
     * 3. Build CSR graph from edge list
     * 4. Apply reordering algorithm
     * 5. Map local reordered IDs back to global IDs
     * 6. Assign sequential global IDs to the reordered nodes
     * 
     * Delegates to ::ReorderCommunitySubgraphStandalone in reorder/reorder.h
     * 
     * @param g The full graph
     * @param nodes Nodes in this community
     * @param node_set Set version for O(1) membership lookup
     * @param algo Algorithm to apply
     * @param useOutdeg Use out-degree (true) or in-degree (false)
     * @param new_ids Output global mapping (modified in place)
     * @param current_id Starting global ID (updated after assignment)
     */
    void ReorderCommunitySubgraph(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        const std::vector<NodeID_>& nodes,
        const std::unordered_set<NodeID_>& node_set,
        ReorderingAlgo algo,
        bool useOutdeg,
        pvector<NodeID_>& new_ids,
        NodeID_& current_id)
    {
        ::ReorderCommunitySubgraphStandalone<NodeID_, DestID_, WeightT_, invert>(
            g, nodes, node_set, algo, useOutdeg, new_ids, current_id);
    }

    void GenerateMapping(CSRGraph<NodeID_, DestID_, invert> &g,
                         pvector<NodeID_> &new_ids,
                         ReorderingAlgo reordering_algo, bool useOutdeg,
                         std::vector<std::string> reordering_options)
    {
        // Unified timing wrapper for all reordering algorithms
        Timer reorder_timer;
        reorder_timer.Start();
        
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
        {
            // RabbitOrder with variants: csr (default), boost
            // Format: -o 8:variant (e.g., -o 8:boost for original Boost-based)
            std::string variant = "csr";  // Default to CSR (faster, no external deps)
            if (!reordering_options.empty() && !reordering_options[0].empty()) {
                variant = reordering_options[0];
            }
            
            if (variant == "boost") {
                // Original Boost-based RabbitOrder needs preprocessing
                pvector<NodeID_> new_ids_local(g.num_nodes(), -1);
                pvector<NodeID_> new_ids_local_2(g.num_nodes(), -1);
                GenerateSortMappingRabbit(g, new_ids_local, true, true);
                CSRGraph<NodeID_, DestID_, invert> g_trans = RelabelByMapping(g, new_ids_local);
                GenerateRabbitOrderMapping(g_trans, new_ids_local_2);
                
                #pragma omp parallel for
                for (NodeID_ n = 0; n < g.num_nodes(); n++)
                {
                    new_ids[n] = new_ids_local_2[new_ids_local[n]];
                }
            } else {
                // Native CSR implementation with degree preprocessing (like boost)
                // Pre-sort nodes by degree for better community detection convergence
                pvector<NodeID_> new_ids_local(g.num_nodes(), -1);
                pvector<NodeID_> new_ids_local_2(g.num_nodes(), -1);
                GenerateSortMappingRabbit(g, new_ids_local, true, true);
                CSRGraph<NodeID_, DestID_, invert> g_trans = RelabelByMapping(g, new_ids_local);
                GenerateRabbitOrderCSRMapping(g_trans, new_ids_local_2);
                
                #pragma omp parallel for
                for (NodeID_ n = 0; n < g.num_nodes(); n++)
                {
                    new_ids[n] = new_ids_local_2[new_ids_local[n]];
                }
            }
        }
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
            // GVE-Leiden library (baseline reference) - Format: 15:resolution
            GenerateLeidenMapping(g, new_ids, reordering_options);
            break;
        case LeidenDendrogram:
            // Leiden Dendrogram - Format: 16:resolution:variant
            // Variants: dfs, dfshub, dfssize, bfs, hybrid (default: hybrid)
            // ⚠️ DEPRECATED: Use LeidenCSR (17) variants instead
            GenerateLeidenDendrogramMappingUnified(g, new_ids, reordering_options);
            break;
        case LeidenCSR:
            // Fast Leiden on CSR - Format: 17:variant:resolution:iterations:passes
            // Variants: gve (default), gveopt, gverabbit, dfs, bfs, hubsort, fast, modularity
            GenerateLeidenCSRMappingUnified(g, new_ids, reordering_options);
            break;
        case GraphBrewOrder:
            // Extended GraphBrewOrder - Format: 12:cluster_variant:final_algo:resolution:levels
            // Cluster variants: leiden (default), gve, gveopt, rabbit, hubcluster
            // Also supports old format: 12:freq_threshold:final_algo:resolution
            GenerateGraphBrewMappingUnified(g, new_ids, useOutdeg, reordering_options);
            break;
        case AdaptiveOrder:
            GenerateAdaptiveMapping(g, new_ids, useOutdeg, reordering_options);
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
        
        // Print unified reorder time for easy parsing
        reorder_timer.Stop();
        std::cout << "=== Reorder Summary ===" << std::endl;
        PrintLabel("Algorithm", ReorderingAlgoStr(reordering_algo));
        PrintTime("Reorder Time", reorder_timer.Seconds());

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
    GenerateMappingLocalEdgelist(const CSRGraph<NodeID_, DestID_, invert> &g_org, EdgeList &el, pvector<NodeID_> &new_ids,
                                 ReorderingAlgo reordering_algo, bool useOutdeg,
                                 std::vector<std::string> reordering_options, int numLevels = 1, bool recursion = false)
    {
        // CRITICAL: Disable nested parallelism to avoid thread explosion
        // For small subgraphs (<100K edges), nested parallelism causes massive
        // overhead (30K+ thread creates) making it 50x slower than sequential
        const size_t MIN_EDGES_FOR_PARALLEL = 100000;
        const bool is_small_subgraph = el.size() < MIN_EDGES_FOR_PARALLEL;
        
        // Save current settings
        int prev_nested = omp_get_nested();
        int prev_max_levels = omp_get_max_active_levels();
        int prev_num_threads = omp_get_max_threads();
        
        if (is_small_subgraph) {
            // For small subgraphs: run sequentially to avoid thread overhead
            omp_set_nested(0);
            omp_set_max_active_levels(1);
            omp_set_num_threads(1);
        } else {
            // For large subgraphs: enable limited parallelism
            omp_set_nested(1);
            omp_set_max_active_levels(2);  // Limit nesting depth
        }
        
        CSRGraph<NodeID_, DestID_, invert> g = MakeLocalGraphFromEL(el);
        g.copy_org_ids(g_org.get_org_ids());

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
        {
            // RabbitOrder with variants: csr (default), boost
            std::string variant = "csr";  // Default to CSR
            if (!reordering_options.empty() && !reordering_options[0].empty()) {
                variant = reordering_options[0];
            }
            
            if (variant == "boost") {
                // Original Boost-based RabbitOrder needs preprocessing
                pvector<NodeID_> new_ids_local(g_org.num_nodes(), -1);
                pvector<NodeID_> new_ids_local_2(g_org.num_nodes(), -1);
                GenerateSortMappingRabbit(g, new_ids_local, true, true);
                g = RelabelByMapping(g, new_ids_local);
                GenerateRabbitOrderMapping(g, new_ids_local_2);

                // Only parallelize final merge for large graphs
                // CRITICAL: Only process nodes that were in the subgraph (have valid mapping)
                if (is_small_subgraph) {
                    for (NodeID_ n = 0; n < g_org.num_nodes(); n++)
                    {
                        if (new_ids_local[n] != (NodeID_)-1 && new_ids_local[n] < g_org.num_nodes()) {
                            new_ids[n] = new_ids_local_2[new_ids_local[n]];
                        }
                    }
                } else {
                    #pragma omp parallel for
                    for (NodeID_ n = 0; n < g_org.num_nodes(); n++)
                    {
                        if (new_ids_local[n] != (NodeID_)-1 && new_ids_local[n] < g_org.num_nodes()) {
                            new_ids[n] = new_ids_local_2[new_ids_local[n]];
                        }
                    }
                }
            } else {
                // Native CSR implementation with degree preprocessing (like boost)
                // Pre-sort nodes by degree for better community detection convergence
                pvector<NodeID_> new_ids_local(g.num_nodes(), -1);
                pvector<NodeID_> new_ids_local_2(g.num_nodes(), -1);
                GenerateSortMappingRabbit(g, new_ids_local, true, true);
                CSRGraph<NodeID_, DestID_, invert> g_trans = RelabelByMapping(g, new_ids_local);
                GenerateRabbitOrderCSRMapping(g_trans, new_ids_local_2);
                
                // Combine mappings - handle subgraph case
                if (is_small_subgraph) {
                    for (NodeID_ n = 0; n < g.num_nodes(); n++)
                    {
                        new_ids[n] = new_ids_local_2[new_ids_local[n]];
                    }
                } else {
                    #pragma omp parallel for
                    for (NodeID_ n = 0; n < g.num_nodes(); n++)
                    {
                        new_ids[n] = new_ids_local_2[new_ids_local[n]];
                    }
                }
            }
        }
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
            // GVE-Leiden library (baseline reference) - Format: 15:resolution
            GenerateLeidenMapping(g, new_ids, reordering_options);
            break;
        case LeidenDendrogram:
            // Leiden Dendrogram - Format: 16:resolution:variant
            // ⚠️ DEPRECATED: Use LeidenCSR (17) variants instead
            GenerateLeidenDendrogramMappingUnified(g, new_ids, reordering_options);
            break;
        case LeidenCSR:
            // Fast Leiden on CSR - Format: 17:variant:resolution:iterations:passes
            GenerateLeidenCSRMappingUnified(g, new_ids, reordering_options);
            break;
        case GraphBrewOrder:
            GenerateGraphBrewMapping(g, new_ids, useOutdeg, reordering_options, numLevels, recursion);
            break;
        case AdaptiveOrder:
            GenerateAdaptiveMapping(g, new_ids, useOutdeg, reordering_options);
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
        
        // Restore OpenMP settings
        omp_set_nested(prev_nested);
        omp_set_max_active_levels(prev_max_levels);
        omp_set_num_threads(prev_num_threads);
        
#ifdef _DEBUG
        VerifyMapping(g, new_ids);
        // exit(-1);
#endif
    }

    /**
     * Convert string argument to ReorderingAlgo.
     * Delegates to the global ::getReorderingAlgo in reorder/reorder.h.
     */
    ReorderingAlgo getReorderingAlgo(const char *arg)
    {
        return ::getReorderingAlgo(arg);
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

    /**
     * @brief Identity mapping - keeps original vertex IDs
     * Delegates to ::GenerateOriginalMapping in reorder/reorder_basic.h
     */
    void GenerateOriginalMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                 pvector<NodeID_> &new_ids) {
        ::GenerateOriginalMapping<NodeID_, DestID_, invert>(g, new_ids);
    }

    /**
     * @brief Random permutation of vertex IDs
     * Delegates to ::GenerateRandomMapping in reorder/reorder_basic.h
     */
    void GenerateRandomMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                               pvector<NodeID_> &new_ids) {
        ::GenerateRandomMapping<NodeID_, DestID_, invert>(g, new_ids);
    }

    /**
     * @brief Random permutation using atomic compare-and-swap (legacy v2)
     * Delegates to ::GenerateRandomMapping_v2 in reorder/reorder_basic.h
     */
    void GenerateRandomMapping_v2(const CSRGraph<NodeID_, DestID_, invert> &g,
                                  pvector<NodeID_> &new_ids) {
        ::GenerateRandomMapping_v2<NodeID_, DestID_, invert>(g, new_ids);
    }

    /**
     * @brief HubSort within DBG buckets
     * Delegates to ::GenerateHubSortDBGMapping in reorder/reorder_hub.h
     */
    void GenerateHubSortDBGMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                   pvector<NodeID_> &new_ids, bool useOutdeg) {
        ::GenerateHubSortDBGMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
    }

    /**
     * @brief HubCluster within DBG buckets (2-bucket version)
     * Delegates to ::GenerateHubClusterDBGMapping in reorder/reorder_hub.h
     */
    void GenerateHubClusterDBGMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                      pvector<NodeID_> &new_ids, bool useOutdeg) {
        ::GenerateHubClusterDBGMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
    }

    /**
     * @brief Sort vertices by degree, placing hubs first
     * Delegates to ::GenerateHubSortMapping in reorder/reorder_hub.h
     */
    void GenerateHubSortMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                pvector<NodeID_> &new_ids, bool useOutdeg) {
        ::GenerateHubSortMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
    }

    /**
     * @brief Sort all vertices by degree
     * Delegates to ::GenerateSortMapping in reorder/reorder_basic.h
     */
    void GenerateSortMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                             pvector<NodeID_> &new_ids, bool useOutdeg,
                             bool lesser = false) {
        ::GenerateSortMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg, lesser);
    }

    /**
     * Sort by (out-degree, in-degree) descending with RabbitOrder-style handling
     * Delegates to ::GenerateSortMappingRabbit in reorder/reorder_basic.h
     */
    void GenerateSortMappingRabbit(const CSRGraph<NodeID_, DestID_, invert> &g,
                                   pvector<NodeID_> &new_ids, bool useOutdeg,
                                   bool lesser = false) {
        ::GenerateSortMappingRabbit<NodeID_, DestID_, invert>(g, new_ids, useOutdeg, lesser);
    }

    /**
     * @brief Degree-Based Grouping into logarithmic buckets
     * Delegates to ::GenerateDBGMapping in reorder/reorder_hub.h
     */
    void GenerateDBGMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                            pvector<NodeID_> &new_ids, bool useOutdeg) {
        ::GenerateDBGMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
    }

    /**
     * @brief Cluster hub vertices with partition-aware locality
     * Delegates to ::GenerateHubClusterMapping in reorder/reorder_hub.h
     */
    void GenerateHubClusterMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                   pvector<NodeID_> &new_ids, bool useOutdeg) {
        ::GenerateHubClusterMapping<NodeID_, DestID_, invert>(g, new_ids, useOutdeg);
    }

    /**
     * @brief Cache-aware workload balancing ordering
     * Delegates to ::GenerateCOrderMapping in reorder/reorder_classic.h
     */
    void GenerateCOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                               pvector<NodeID_> &new_ids) {
        ::GenerateCOrderMapping<NodeID_, DestID_, invert>(g, new_ids);
    }

    /**
     * COrder v2 - Optimized parallel version using Vector2d
     * Delegates to ::GenerateCOrderMapping_v2 in reorder/reorder_classic.h
     */
    void GenerateCOrderMapping_v2(const CSRGraph<NodeID_, DestID_, invert> &g,
                                  pvector<NodeID_> &new_ids) {
        ::GenerateCOrderMapping_v2<NodeID_, DestID_, invert>(g, new_ids);
    }


    /**
     * @brief RabbitOrder - Community-aware reordering using hierarchical clustering
     * Delegates to ::GenerateRabbitOrderMapping in reorder/reorder_rabbit.h
     */
    void GenerateRabbitOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                    pvector<NodeID_> &new_ids) {
        ::GenerateRabbitOrderMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids);
    }

    /**
     * @brief Compute RabbitOrder modularity from edge list
     * Delegates to ::GenerateRabbitModularityEdgelist in reorder/reorder_rabbit.h
     */
    double GenerateRabbitModularityEdgelist(EdgeList &edgesList, bool is_weighted) {
        std::vector<edge_list::edge> edges(edgesList.size());
        #pragma omp parallel for
        for (size_t i = 0; i < edges.size(); ++i) {
            if (is_weighted) {
                edges[i] = std::make_tuple(
                    static_cast<rabbit_order::vint>(edgesList[i].u),
                    static_cast<NodeWeight<NodeID_, WeightT_>>(edgesList[i].v).v,
                    static_cast<NodeWeight<NodeID_, WeightT_>>(edgesList[i].v).w);
            } else {
                edges[i] = std::make_tuple(
                    static_cast<rabbit_order::vint>(edgesList[i].u),
                    static_cast<rabbit_order::vint>(edgesList[i].v),
                    1.0f);
            }
        }
        return ::GenerateRabbitModularityEdgelist<NodeID_>(edges);
    }

#ifdef RABBIT_ENABLE
    /**
     * @brief RabbitOrder from edge list
     * Delegates to ::GenerateRabbitOrderMappingEdgelist in reorder/reorder_rabbit.h
     */
    void GenerateRabbitOrderMappingEdgelist(const std::vector<edge_list::edge> &edges,
                                            pvector<NodeID_> &new_ids) {
        ::GenerateRabbitOrderMappingEdgelist<NodeID_>(edges, new_ids);
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

    /**
     * @brief GOrder - Graph Ordering using dynamic programming and windowing
     * Delegates to ::GenerateGOrderMapping in reorder/reorder_classic.h
     */
    void GenerateGOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                               pvector<NodeID_> &new_ids) {
        ::GenerateGOrderMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids, cli_.filename());
    }

    /**
     * @brief Reverse Cuthill-McKee ordering for bandwidth reduction
     * Delegates to ::GenerateRCMOrderMapping in reorder/reorder_classic.h
     */
    void GenerateRCMOrderMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                 pvector<NodeID_> &new_ids) {
        ::GenerateRCMOrderMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids, cli_.filename());
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

    //==========================================================================
    // GVE-LEIDEN: True Leiden Algorithm on CSR Graphs
    // Implementation following ACM paper: "Fast Leiden Algorithm for Community
    // Detection in Shared Memory Setting" (DOI: 10.1145/3673038.3673146)
    //
    // IMPORTANT: This implementation handles BOTH symmetric and non-symmetric
    // CSR graphs by scanning both out_neigh and in_neigh to get all edges.
    //
    // Key differences from Louvain:
    // 1. Refinement phase: Only isolated vertices can move
    // 2. Community bounds: Refined communities constrained within local-moving results
    // 3. Well-connected communities guaranteed
    //
    // Note: GVELeidenResult, GVEDendroResult, GVEAtomicDendroResult structures,
    // atomicMergeToDendro, mergeToDendro, and initDendrogram are now in reorder/reorder_types.h
    //==========================================================================
    
    /**
     * Traverse dendrogram using DFS, assigning new IDs.
     * Orders vertices so that community members are contiguous.
     * Hub-first: Higher weight children are visited first.
     * Delegates to ::traverseDendrogramDFS in reorder/reorder_types.h
     */
    template <typename K = uint32_t>
    void traverseDendrogramDFS(
        const GVEDendroResult<K>& dendro,
        pvector<NodeID_>& new_ids,
        bool hub_first = true) {
        
        // Convert pvector to std::vector for the standalone function
        std::vector<NodeID_> temp_ids(new_ids.size());
        ::traverseDendrogramDFS<K, NodeID_>(dendro, temp_ids, hub_first);
        
        // Copy back to pvector
        #pragma omp parallel for
        for (size_t i = 0; i < temp_ids.size(); ++i) {
            new_ids[i] = temp_ids[i];
        }
    }
    
    /**
     * Fast parallel modularity computation for any community assignment.
     * 
     * Modularity Q = (1/2m) * Σ[A_ij - k_i*k_j/(2m)] * δ(c_i, c_j)
     * 
     * Delegates to ::computeModularityCSR in reorder/reorder_types.h
     */
    template <typename K>
    double computeModularityCSR(
        const CSRGraph<NodeID_, DestID_, true>& g,
        const std::vector<K>& community,
        double resolution = 1.0) {
        return ::computeModularityCSR<K, NodeID_, DestID_>(g, community, resolution);
    }
    
    /**
     * Scan all edges connected to vertex u (both out-edges and in-edges).
     * Delegates to ::scanVertexEdges in reorder/reorder_types.h
     */
    template <typename K, typename W>
    inline W scanVertexEdges(
        NodeID_ u,
        const K* vcom,
        std::unordered_map<K, W>& hash,
        K d,
        const CSRGraph<NodeID_, DestID_, true>& g,
        bool graph_is_symmetric) {
        return ::scanVertexEdges<K, W, NodeID_, DestID_>(u, vcom, hash, d, g, graph_is_symmetric);
    }
    
    /**
     * Compute vertex total weight (degree sum for unweighted graphs).
     * Delegates to ::computeVertexTotalWeightCSR in reorder/reorder_types.h
     */
    template <typename W>
    inline W computeVertexTotalWeight(
        NodeID_ u,
        const CSRGraph<NodeID_, DestID_, true>& g,
        bool graph_is_symmetric) {
        return ::computeVertexTotalWeightCSR<W, NodeID_, DestID_>(u, g, graph_is_symmetric);
    }
    
    /**
     * Mark all neighbors of vertex u as affected.
     * Delegates to ::markNeighborsAffected in reorder/reorder_types.h
     */
    inline void markNeighborsAffected(
        NodeID_ u,
        std::vector<char>& vaff,
        const CSRGraph<NodeID_, DestID_, true>& g,
        bool graph_is_symmetric) {
        ::markNeighborsAffected<NodeID_, DestID_>(u, vaff, g, graph_is_symmetric);
    }
    
    /**
     * Delta modularity calculation for community move.
     * Delegates to graphbrew::leiden::gveDeltaModularity in reorder/reorder_leiden.h
     */
    template <typename W>
    inline W gveDeltaModularity(W ki_to_c, W ki_to_d, W ki, W sigma_c, W sigma_d, double M, double R) {
        return graphbrew::leiden::gveDeltaModularity<W>(
            ki_to_c, ki_to_d, ki, sigma_c, sigma_d, static_cast<W>(M), static_cast<W>(R));
    }
    
    /**
     * GVE-Leiden Local-Moving Phase (Algorithm 2)
     * Delegates to ::gveLeidenLocalMoveCSR in reorder/reorder_leiden.h
     */
    template <typename K = uint32_t, typename W = double>
    int gveLeidenLocalMove(
        std::vector<K>& vcom,
        std::vector<W>& ctot,
        std::vector<char>& vaff,
        const std::vector<W>& vtot,
        const int64_t num_nodes,
        const CSRGraph<NodeID_, DestID_, true>& g,
        bool graph_is_symmetric,
        double M, double R, int L, double tolerance) {
        return ::gveLeidenLocalMoveCSR<K, W, NodeID_, DestID_>(
            vcom, ctot, vaff, vtot, num_nodes, g, graph_is_symmetric, M, R, L, tolerance);
    }
    
    /**
     * GVE-Leiden Refinement Phase (Algorithm 3)
     * Delegates to ::gveLeidenRefineCSR in reorder/reorder_leiden.h
     */
    template <typename K = uint32_t, typename W = double>
    int gveLeidenRefine(
        std::vector<K>& vcom,
        std::vector<W>& ctot,
        std::vector<char>& vaff,
        const std::vector<K>& vcob,
        const std::vector<W>& vtot,
        const int64_t num_nodes,
        const CSRGraph<NodeID_, DestID_, true>& g,
        bool graph_is_symmetric,
        double M, double R) {
        return ::gveLeidenRefineCSR<K, W, NodeID_, DestID_>(
            vcom, ctot, vaff, vcob, vtot, num_nodes, g, graph_is_symmetric, M, R);
    }
    
    /**
     * Compute community-to-community edge weights for virtual aggregation.
     * Delegates to ::computeCommunityGraphCSR in reorder/reorder_leiden.h
     */
    template <typename K, typename W>
    void computeCommunityGraph(
        std::unordered_map<K, std::unordered_map<K, W>>& comm_graph,
        std::unordered_map<K, W>& comm_weight,
        const std::vector<K>& vcom,
        const std::vector<W>& vtot,
        const int64_t num_nodes,
        const CSRGraph<NodeID_, DestID_, true>& g,
        bool graph_is_symmetric) {
        ::computeCommunityGraphCSR<K, W, NodeID_, DestID_>(
            comm_graph, comm_weight, vcom, vtot, num_nodes, g, graph_is_symmetric);
    }
    
    /**
     * Local-moving on the community graph (virtual aggregation).
     * Delegates to ::communityLocalMove in reorder/reorder_leiden.h
     */
    template <typename K, typename W>
    std::unordered_map<K, K> communityLocalMove(
        const std::unordered_map<K, std::unordered_map<K, W>>& comm_graph,
        const std::unordered_map<K, W>& comm_weight,
        double M, double R, int max_iterations, double tolerance) {
        return ::communityLocalMove<K, W>(comm_graph, comm_weight, M, R, max_iterations, tolerance);
    }
    
    /**
     * Main GVE-Leiden Algorithm (Algorithm 1) with REFINEMENT and AGGREGATION
     * 
     * Following the paper: "Fast Leiden Algorithm for Community Detection in Shared Memory Setting"
     * ACM DOI: 10.1145/3673038.3673146
     * 
     * Key differences from Louvain (used by RabbitOrder):
     * 1. REFINEMENT phase ensures well-connected communities (only isolated vertices move)
     * 2. Proper aggregation with community bounds
     * 
     * This should produce HIGHER modularity than RabbitOrder/Louvain.
     * 
     * Delegates to ::GVELeidenCSR in reorder/reorder_leiden.h
     */
    template <typename K = uint32_t>
    GVELeidenResult<K> GVELeidenCSR(
        const CSRGraph<NodeID_, DestID_, true>& g,
        double resolution = 1.0,
        double tolerance = 1e-2,
        double aggregation_tolerance = 0.8,
        double tolerance_drop = 10.0,
        int max_iterations = 20,
        int max_passes = 10) {
        
        return ::GVELeidenCSR<K, double, NodeID_, DestID_>(
            g, resolution, tolerance, aggregation_tolerance, 
            tolerance_drop, max_iterations, max_passes);
    }

    //==========================================================================
    // OPTIMIZED GVE-LEIDEN: Cache-Optimized Leiden Algorithm
    //==========================================================================
    
    /**
     * Optimized local move scan using flat array instead of hash map.
     * Uses prefetching for better cache performance.
     * Delegates to external implementation in reorder_leiden.h
     */
    template <typename K = uint32_t, typename W = double>
    inline W gveOptScanVertex(
        NodeID_ u,
        const K* __restrict__ vcom,
        W* __restrict__ comm_weights,
        K* __restrict__ touched_comms,
        int& num_touched,
        K d,
        const CSRGraph<NodeID_, DestID_, true>& g,
        bool graph_is_symmetric) {
        return ::gveOptScanVertex<K, W, NodeID_, DestID_, WeightT_>(
            u, vcom, comm_weights, touched_comms, num_touched, d, g, graph_is_symmetric);
    }
    
    /**
     * Optimized GVE-Leiden Local-Moving Phase
     * Delegates to external implementation in reorder_leiden.h
     */
    template <typename K = uint32_t, typename W = double>
    int gveOptLocalMove(
        std::vector<K>& vcom,
        std::vector<W>& ctot,
        std::vector<char>& vaff,
        const std::vector<W>& vtot,
        const int64_t num_nodes,
        const CSRGraph<NodeID_, DestID_, true>& g,
        bool graph_is_symmetric,
        double M, double R, int L, double tolerance) {
        return ::gveOptLocalMove<K, W, NodeID_, DestID_, WeightT_>(
            vcom, ctot, vaff, vtot, num_nodes, g, graph_is_symmetric, M, R, L, tolerance);
    }
    
    /**
     * Optimized GVE-Leiden Refinement Phase
     * Delegates to external implementation in reorder_leiden.h
     */
    template <typename K = uint32_t, typename W = double>
    int gveOptRefine(
        std::vector<K>& vcom,
        std::vector<W>& ctot,
        std::vector<char>& vaff,
        const std::vector<K>& vcob,
        const std::vector<W>& vtot,
        const int64_t num_nodes,
        const CSRGraph<NodeID_, DestID_, true>& g,
        bool graph_is_symmetric,
        double M, double R) {
        return ::gveOptRefine<K, W, NodeID_, DestID_, WeightT_>(
            vcom, ctot, vaff, vcob, vtot, num_nodes, g, graph_is_symmetric, M, R);
    }
    
    /**
     * GVELeidenOpt - Optimized GVE-Leiden with cache optimizations
     * 
     * Key optimizations:
     * - Flat arrays instead of hash maps for community scanning
     * - Prefetching for community lookups
     * - Guided scheduling for better load balancing
     * - Optimized super-graph construction with sorted edge merging
     * 
     * Delegates to ::GVELeidenOptCSR in reorder/reorder_leiden.h
     */
    template <typename K = uint32_t>
    GVELeidenResult<K> GVELeidenOpt(
        const CSRGraph<NodeID_, DestID_, true>& g,
        double resolution = 1.0,
        double tolerance = 1e-2,
        double aggregation_tolerance = 0.8,
        double tolerance_drop = 10.0,
        int max_iterations = 20,
        int max_passes = 10) {
        
        return ::GVELeidenOptCSR<K, double, NodeID_, DestID_>(
            g, resolution, tolerance, aggregation_tolerance,
            tolerance_drop, max_iterations, max_passes);
    }

    //==========================================================================
    // GVE-LEIDEN WITH INCREMENTAL DENDROGRAM (RabbitOrder-inspired)
    //
    // These variants build the dendrogram DURING community detection instead
    // of as a post-processing step. This is inspired by RabbitOrder's efficient
    // tree building using child/sibling pointers.
    //
    // Key optimization: Instead of storing community_per_pass and rebuilding
    // the hierarchy, we track parent-child relationships as vertices merge.
    //==========================================================================
    
    /**
     * Build dendrogram from community assignments AFTER local-moving completes.
     * Delegates to ::buildDendrogramFromCommunities in reorder/reorder_types.h
     */
    template <typename K, typename W>
    void buildDendrogramFromCommunities(
        GVEDendroResult<K>& dendro,
        const std::vector<K>& vcom,
        const std::vector<W>& vtot,
        int64_t num_nodes) {
        ::buildDendrogramFromCommunities<K, W>(dendro, vcom, vtot, num_nodes);
    }
    
    /**
     * GVELeidenDendo - GVE-Leiden with INCREMENTAL atomic dendrogram building
     * 
     * This is a clone of GVELeidenCSR that builds the dendrogram incrementally
     * using RabbitOrder-style atomic CAS instead of post-processing.
     * 
     * Key optimization: Instead of storing community_per_pass and rebuilding
     * the tree in post-processing, we build parent-child relationships as
     * vertices merge during detection using lock-free atomic operations.
     * 
     * Delegates to ::GVELeidenDendoCSR in reorder/reorder_leiden.h
     */
    template <typename K = uint32_t>
    GVEDendroResult<K> GVELeidenDendo(
        const CSRGraph<NodeID_, DestID_, true>& g,
        double resolution = 1.0,
        double tolerance = 1e-2,
        double aggregation_tolerance = 0.8,
        double tolerance_drop = 10.0,
        int max_iterations = 20,
        int max_passes = 10) {
        
        return ::GVELeidenDendoCSR<K, double, NodeID_, DestID_>(
            g, resolution, tolerance, aggregation_tolerance,
            tolerance_drop, max_iterations, max_passes);
    }
    
    /**
     * GVELeidenOptDendo - Optimized GVE-Leiden with incremental dendrogram
     * 
     * Clone of GVELeidenOpt with atomic dendrogram building.
     * Uses optimized flat-array scanning plus lock-free tree construction.
     * 
     * Delegates to ::GVELeidenOptDendoCSR in reorder/reorder_leiden.h
     */
    template <typename K = uint32_t>
    GVEDendroResult<K> GVELeidenOptDendo(
        const CSRGraph<NodeID_, DestID_, true>& g,
        double resolution = 1.0,
        double tolerance = 1e-2,
        double aggregation_tolerance = 0.8,
        double tolerance_drop = 10.0,
        int max_iterations = 20,
        int max_passes = 10) {
        
        return ::GVELeidenOptDendoCSR<K, double, NodeID_, DestID_>(
            g, resolution, tolerance, aggregation_tolerance,
            tolerance_drop, max_iterations, max_passes);
    }
    
    /**
     * Generate mapping using GVELeidenDendo algorithm
     * Delegates to ::GenerateGVELeidenDendoMapping in reorder/reorder_leiden.h
     */
    void GenerateGVELeidenDendoMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        std::vector<std::string> reordering_options) {
        ::GenerateGVELeidenDendoMapping<K, NodeID_, DestID_>(g, new_ids, reordering_options);
    }
    
    /**
     * Generate mapping using GVELeidenOptDendo algorithm
     * Delegates to ::GenerateGVELeidenOptDendoMapping in reorder/reorder_leiden.h
     */
    void GenerateGVELeidenOptDendoMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        std::vector<std::string> reordering_options) {
        ::GenerateGVELeidenOptDendoMapping<K, NodeID_, DestID_>(g, new_ids, reordering_options);
    }

    //==========================================================================
    // FAST LEIDEN-CSR: Direct CSR Community Detection (No DiGraph Conversion)
    //==========================================================================
    
    /**
     * FastLeidenCSR - Union-Find based community detection on CSR graphs
     * 
     * Uses modularity-guided community merging with Union-Find for efficiency.
     * Delegates to ::FastLeidenCSR in reorder/reorder_leiden.h
     *
     * Returns: vector of community assignments per pass (finest to coarsest)
     */
    template <typename K = uint32_t>
    std::vector<std::vector<K>> FastLeidenCSR(
        const CSRGraph<NodeID_, DestID_, true>& g,
        double resolution = 1.0,
        int max_iterations = 10,
        int max_passes = 10)
    {
        return ::FastLeidenCSR<K, NodeID_, DestID_>(g, resolution, max_iterations, max_passes);
    }
    
    // Keep old function name for compatibility
    template <typename K = uint32_t>
    std::vector<std::vector<K>> FastLabelPropagationCSR(
        const CSRGraph<NodeID_, DestID_, true>& g,
        double resolution = 1.0,
        int max_iterations = 10,
        int max_passes = 5)
    {
        return FastLeidenCSR<K>(g, resolution, max_iterations, max_passes);
    }
    
    /**
     * GenerateLeidenCSRMapping - Fast Leiden-style ordering directly on CSR
     * 
     * This avoids the expensive DiGraph conversion by using label propagation
     * directly on the CSR graph structure. Produces similar quality to 
     * LeidenOrder but much faster (~5-10x speedup on large graphs).
     * 
     * Algorithm:
     * 1. Fast label propagation for multi-level community detection on CSR
     * 2. Build dendrogram from hierarchical community structure
     * 3. Apply ordering based on flavor (DFS, BFS, HubSort)
     * 
     * Flavors (int):
     *   0: DFS (standard)
     *   1: BFS (level-first)
     *   2: HubSort (community + degree)
     */
    /**
     * GenerateLeidenCSRMapping - Fast Leiden-style ordering directly on CSR
     * 
     * Delegates to ::GenerateLeidenCSRMapping in reorder/reorder_leiden.h
     */
    void GenerateLeidenCSRMapping(const CSRGraph<NodeID_, DestID_, invert>& g,
                                   pvector<NodeID_>& new_ids,
                                   std::vector<std::string> reordering_options,
                                   int flavor = 2) {
        ::GenerateLeidenCSRMapping<K, NodeID_, DestID_>(g, new_ids, reordering_options, flavor);
    }

    //==========================================================================
    // LEIDENFAST: Parallel Community-Based Graph Reordering
    //==========================================================================
    
    /**
     * LeidenFast: Fast parallel reordering using Union-Find + Label Propagation
     * 
     * Improvements over initial version:
     * 1. Parallel Union-Find with atomic CAS (like RabbitOrder)
     * 2. Best-fit merging (scan ALL neighbors, not first-fit)
     * 3. Efficient hash-based counting for label propagation
     * 4. Multi-level aggregation for deeper hierarchy
     * 5. Modularity-weighted edge selection
     * 
     * Strategy:
     * Phase 1: Parallel Union-Find merging with modularity criterion
     * Phase 2: Label propagation refinement with modularity-weighted moves
     * Phase 3: Optional aggregation for hierarchical structure
     * Final: Order by community strength DESC, degree DESC within community
     */
    
    /**
     * Fast parallel community detection using Union-Find + Label Propagation
     * Delegates to ::FastModularityCommunityDetection in reorder/reorder_leiden.h
     */
    template<typename NodeID_T, typename DestID_T>
    void FastModularityCommunityDetection(
        const CSRGraph<NodeID_T, DestID_T, true>& g,
        std::vector<double>& vertex_strength,
        std::vector<int64_t>& final_community,
        double resolution = 1.0,
        int max_passes = 3)
    {
        ::FastModularityCommunityDetection<NodeID_T, DestID_T>(
            g, vertex_strength, final_community, resolution, max_passes);
    }
    
    /**
     * Build final ordering from communities
     * Delegates to ::BuildCommunityOrdering in reorder/reorder_leiden.h
     */
    template<typename NodeID_T, typename DestID_T>
    void BuildCommunityOrdering(
        const CSRGraph<NodeID_T, DestID_T, true>& g,
        const std::vector<double>& vertex_strength,
        const std::vector<int64_t>& community,
        std::vector<int64_t>& ordered_vertices)
    {
        ::BuildCommunityOrdering<NodeID_T, DestID_T>(
            g, vertex_strength, community, ordered_vertices);
    }
    
    /**
     * GenerateLeidenFastMapping - Main entry point for LeidenFast algorithm
     * 
     * Improved version with:
     * - Parallel Union-Find with atomic CAS
     * - Best-fit modularity merging (not first-fit)
     * - Hash-based label propagation (faster than sorted array)
     * - Proper convergence detection
     * 
     * Delegates to ::GenerateLeidenFastMapping in reorder/reorder_leiden.h
     */
    void GenerateLeidenFastMapping(const CSRGraph<NodeID_, DestID_, invert>& g,
                                   pvector<NodeID_>& new_ids,
                                   std::vector<std::string> reordering_options) {
        ::GenerateLeidenFastMapping<NodeID_, DestID_>(g, new_ids, reordering_options);
    }
    
    //==========================================================================
    // LEIDEN (True Implementation): Quality-focused community detection
    //==========================================================================
    
    /**
     * Leiden Algorithm - Highly optimized parallel local moving
     * 
     * Optimizations:
     * 1. Use flat array counting when labels are dense (first iterations)
     * 2. Process degree-1 vertices specially (just adopt neighbor's community)
     * 3. Minimize atomic operations
     * 4. Auto-tune resolution based on graph density
     */
    
    /**
     * Compute optimal resolution based on graph properties.
     * 
     * Heuristic for stable partitions for reordering; not a research-derived
     * optimum. Users should sweep γ for best community quality.
     * 
     * Logic:
     * - Continuous mapping: γ = clip(0.5 + 0.25*log10(avg_degree+1), 0.5, 1.2)
     * - CV guardrail: if degree variance is high (CV > 2), nudge toward 1.0
     *   because heavy-tailed (hubby) graphs can produce unstable mega-communities
     */
    template<typename NodeID_T, typename DestID_T>
    double LeidenAutoResolution(const CSRGraph<NodeID_T, DestID_T, true>& g) {
        return computeAutoResolution<NodeID_T, DestID_T>(g);
    }
    
    /**
     * Optimized parallel local moving - two-phase approach
     * Delegates to ::LeidenLocalMoveParallel in reorder/reorder_leiden.h
     */
    template<typename NodeID_T, typename DestID_T>
    int64_t LeidenLocalMoveParallel(
        const CSRGraph<NodeID_T, DestID_T, true>& g,
        std::vector<int64_t>& community,
        std::vector<double>& comm_weight,
        const std::vector<double>& vertex_weight,
        double total_weight,
        double resolution,
        int max_iterations)
    {
        return ::LeidenLocalMoveParallel<NodeID_T, DestID_T>(
            g, community, comm_weight, vertex_weight,
            total_weight, resolution, max_iterations);
    }
    
    /**
     * Main Leiden algorithm - focus on quality communities
     * Delegates to ::LeidenCommunityDetection in reorder/reorder_leiden.h
     */
    template<typename NodeID_T, typename DestID_T>
    void LeidenCommunityDetection(
        const CSRGraph<NodeID_T, DestID_T, true>& g,
        std::vector<int64_t>& final_community,
        double resolution = 1.0,
        int max_passes = 3,
        int max_iterations = 20)
    {
        ::LeidenCommunityDetection<NodeID_T, DestID_T>(
            g, final_community, resolution, max_passes, max_iterations);
    }
    
    /**
     * GenerateLeidenMapping2 - Quality-focused Leiden reordering
     * Delegates to ::GenerateLeidenMapping2 in reorder/reorder_leiden.h
     */
    void GenerateLeidenMapping2(const CSRGraph<NodeID_, DestID_, invert>& g,
                                pvector<NodeID_>& new_ids,
                                std::vector<std::string> reordering_options) {
        ::GenerateLeidenMapping2<NodeID_, DestID_>(g, new_ids, reordering_options);
    }
    
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

        using V = TYPE;
        install_sigsegv();

        // Use auto-resolution based on graph density
        double resolution = LeidenAutoResolution<NodeID_, DestID_>(g);
        // Unified defaults across all Leiden algorithms for fair comparison
        int maxIterations = LEIDEN_DEFAULT_ITERATIONS;
        int maxPasses = LEIDEN_DEFAULT_PASSES;

        if (!reordering_options.empty() && !reordering_options[0].empty())
        {
            const std::string& res_opt = reordering_options[0];
            // Handle special keywords like LeidenCSR does
            if (res_opt == "auto" || res_opt == "0" || res_opt.rfind("dynamic", 0) == 0) {
                // Keep auto-resolution
            } else {
                try {
                    double parsed = std::stod(res_opt);
                    if (parsed > 0 && parsed <= 3) {
                        resolution = parsed;
                    }
                } catch (...) {
                    // Parse error, keep auto-resolution
                }
            }
        }
        if (reordering_options.size() > 1 && !reordering_options[1].empty())
        {
            try { maxIterations = std::stoi(reordering_options[1]); } catch (...) {}
        }
        if (reordering_options.size() > 2 && !reordering_options[2].empty())
        {
            try { maxPasses = std::stoi(reordering_options[2]); } catch (...) {}
        }

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges_directed();

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

        tm.Stop();
        PrintTime("DiGraph graph", tm.Seconds());

        runExperiment(x, resolution, maxIterations, maxPasses);

        size_t num_nodesx;
        size_t num_passes;
        num_nodesx = x.span();
        num_passes = x.communityMappingPerPass.size() + 2;

        // Use flat array with stride for better cache locality (SoA pattern)
        // Layout: [all node IDs][all degrees][pass0 communities][pass1 communities]...
        const size_t stride = num_nodesx;
        std::vector<K> communityDataFlat(num_nodesx * num_passes);
        
        // Initialize node IDs and degrees
        tm.Start();
        #pragma omp parallel for
        for (size_t i = 0; i < num_nodesx; ++i)
        {
            communityDataFlat[i] = i;                        // column 0: node ID
            communityDataFlat[stride + i] = x.degree(i);     // column 1: degree
        }

        // Copy community mappings per pass
        for (size_t p = 0; p < num_passes - 2; ++p)
        {
            K* dest_col = &communityDataFlat[(2 + p) * stride];
            const auto& src = x.communityMappingPerPass[p];
            #pragma omp parallel for
            for (size_t j = 0; j < num_nodesx; ++j)
            {
                dest_col[j] = src[j];
            }
        }

        // Sort by last pass community - create index array for indirect sort
        std::vector<size_t> sort_indices(num_nodesx);
        #pragma omp parallel for
        for (size_t i = 0; i < num_nodesx; ++i) {
            sort_indices[i] = i;
        }
        
        /**
         * DENDROGRAM-BASED ORDERING (Optimization inspired by RabbitOrder)
         * 
         * Key insight: RabbitOrder outperforms LeidenOrder because it preserves
         * hierarchical locality through dendrogram DFS traversal.
         * 
         * Original LeidenOrder problem:
         *   - Only sorts by LAST pass community
         *   - Within a community, order is arbitrary
         *   - Loses fine-grained locality from earlier passes
         * 
         * Solution: Multi-level hierarchical sort
         *   - Sort by ALL passes in order: (pass_N, pass_N-1, ..., pass_0, degree)
         *   - This is equivalent to DFS traversal of the community dendrogram
         *   - Vertices in the same sub-sub-community become adjacent
         *   - Secondary sort by degree puts hubs together (cache-friendly)
         * 
         * This achieves RabbitOrder-like locality while using Leiden's
         * higher-quality community structure.
         */
        const size_t actual_passes = num_passes - 2;  // Exclude nodeID and degree columns
        
        // Sort by ALL passes (coarsest to finest) then by degree
        // This achieves dendrogram DFS-like ordering without building the tree
        __gnu_parallel::sort(sort_indices.begin(), sort_indices.end(),
            [&communityDataFlat, stride, actual_passes](size_t a, size_t b) {
                // Compare all passes from coarsest (last) to finest (first)
                for (size_t p = actual_passes; p > 0; --p) {
                    size_t pass_col = 2 + p - 1;  // Column index for this pass
                    K comm_a = communityDataFlat[pass_col * stride + a];
                    K comm_b = communityDataFlat[pass_col * stride + b];
                    if (comm_a != comm_b) {
                        return comm_a < comm_b;
                    }
                }
                // All passes equal - sort by degree (descending) for hub locality
                K deg_a = communityDataFlat[stride + a];  // degree column
                K deg_b = communityDataFlat[stride + b];
                return deg_a > deg_b;  // High degree first (hubs together)
            });

        // Assign new IDs based on sorted order
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; i++)
        {
            new_ids[communityDataFlat[sort_indices[i]]] = (NodeID_)i;
        }

        // Count unique communities in last pass
        size_t num_communities = 0;
        if (!x.communityMappingPerPass.empty()) {
            const auto& last_pass = x.communityMappingPerPass.back();
            std::set<K> unique_comms(last_pass.begin(), last_pass.end());
            num_communities = unique_comms.size();
        }

        tm.Stop();
        PrintTime("GenID Time", tm.Seconds());
        PrintTime("Num Passes", x.communityMappingPerPass.size());
        PrintTime("Num Communities", num_communities);
        PrintTime("Resolution", resolution);
    }

    //==========================================================================
    // LEIDEN DENDROGRAM-BASED ORDERING (RabbitOrder-style traversal)
    // NOTE: LeidenDendrogramNode, buildLeidenDendrogram, orderDendrogramDFSParallel
    // are now in reorder/reorder_types.h
    //==========================================================================

    /**
     * DFS ordering of dendrogram (sequential version)
     * Delegates to ::orderDendrogramDFS in reorder/reorder_types.h
     */
    void orderDendrogramDFS(
        const std::vector<LeidenDendrogramNode>& nodes,
        const std::vector<int64_t>& roots,
        pvector<NodeID_>& new_ids,
        bool hub_first,
        bool size_first) {
        ::orderDendrogramDFS<NodeID_>(nodes, roots, new_ids, hub_first, size_first);
    }

    /**
     * BFS ordering of dendrogram (by level)
     * Delegates to ::orderDendrogramBFS in reorder/reorder_types.h
     */
    void orderDendrogramBFS(
        const std::vector<LeidenDendrogramNode>& nodes,
        const std::vector<int64_t>& roots,
        pvector<NodeID_>& new_ids) {
        ::orderDendrogramBFS<NodeID_>(nodes, roots, new_ids);
    }

    /**
     * Hybrid ordering: sort by (community, degree descending)
     * Delegates to ::orderLeidenHybridHubDFS in reorder/reorder_types.h
     */
    template<typename K>
    void orderLeidenHybridHubDFS(
        const std::vector<std::vector<K>>& communityMappingPerPass,
        const std::vector<K>& degrees,
        pvector<NodeID_>& new_ids) {
        ::orderLeidenHybridHubDFS<K, NodeID_>(communityMappingPerPass, degrees, new_ids);
    }

    /**
     * Unified Leiden Dendrogram Mapping - Parses variant from options
     * Format: 16:resolution:variant
     * Variants: dfs, dfshub, dfssize, bfs, hybrid (default: hybrid)
     */
    void GenerateLeidenDendrogramMappingUnified(
        CSRGraph<NodeID_, DestID_, invert> &g,
        pvector<NodeID_> &new_ids,
        std::vector<std::string> reordering_options) {
        
        // Default values - use auto-resolution based on graph density
        double resolution = LeidenAutoResolution<NodeID_, DestID_>(g);
        std::string variant = "hybrid";
        
        // Parse options: variant, resolution (flexible order)
        // Format: -o 16:variant:resolution or -o 16:variant or -o 16
        // Resolution can be: "auto", "0", "dynamic", "dynamic:2.0", or numeric
        // e.g., -o 16:hybrid:0.7 or -o 16:dfs or -o 16:dfs:dynamic
        if (!reordering_options.empty() && !reordering_options[0].empty()) {
            std::string first_opt = reordering_options[0];
            // Check if first option is a variant name or a number (resolution)
            if (first_opt == "dfs" || first_opt == "dfshub" || first_opt == "dfssize" || 
                first_opt == "bfs" || first_opt == "hybrid") {
                variant = first_opt;
                // Check for optional resolution
                if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
                    const std::string& res_opt = reordering_options[1];
                    // Handle special keywords
                    if (res_opt == "auto" || res_opt == "0" || res_opt.rfind("dynamic", 0) == 0) {
                        // Keep auto-resolution
                    } else {
                        try {
                            double parsed = std::stod(res_opt);
                            if (parsed > 0 && parsed <= 3) {
                                resolution = parsed;
                            }
                        } catch (...) {
                            // Parse error, keep auto-resolution
                        }
                    }
                }
            } else {
                // Check if first option is a special keyword or numeric resolution
                if (first_opt == "auto" || first_opt == "0" || first_opt.rfind("dynamic", 0) == 0) {
                    // Keep auto-resolution
                } else {
                    try {
                        double parsed = std::stod(first_opt);
                        if (parsed > 0 && parsed <= 3) {
                            resolution = parsed;
                        }
                    } catch (...) {
                        // Parse error, keep auto-resolution
                    }
                }
                if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
                    variant = reordering_options[1];
                }
            }
        }
        
        // Map variant string to internal flavor
        // We'll use a simple approach: pass the original enum values internally
        // but keep the implementation unchanged
        std::vector<std::string> internal_options;
        internal_options.push_back(std::to_string(resolution));
        
        printf("LeidenDendrogram: resolution=%.2f, variant=%s\n", resolution, variant.c_str());
        
        if (variant == "dfs") {
            GenerateLeidenDendrogramMappingInternal(g, new_ids, internal_options, 0);
        } else if (variant == "dfshub") {
            GenerateLeidenDendrogramMappingInternal(g, new_ids, internal_options, 1);
        } else if (variant == "dfssize") {
            GenerateLeidenDendrogramMappingInternal(g, new_ids, internal_options, 2);
        } else if (variant == "bfs") {
            GenerateLeidenDendrogramMappingInternal(g, new_ids, internal_options, 3);
        } else { // hybrid (default)
            GenerateLeidenDendrogramMappingInternal(g, new_ids, internal_options, 4);
        }
    }
    
    /**
     * Unified Leiden CSR Mapping - Parses variant from options
     * Format: 17:variant:resolution:iterations:passes
     * Variants: gveopt2 (default), gve, gveopt, gverabbit, dfs, bfs, hubsort, fast, modularity
     */
    void GenerateLeidenCSRMappingUnified(
        const CSRGraph<NodeID_, DestID_, invert> &g,
        pvector<NodeID_> &new_ids,
        std::vector<std::string> reordering_options) {
        
        // Default values - use auto-resolution based on graph density
        double resolution = LeidenAutoResolution<NodeID_, DestID_>(g);
        // Unified defaults across all Leiden algorithms for fair comparison
        int max_iterations = LEIDEN_DEFAULT_ITERATIONS;
        int max_passes = LEIDEN_DEFAULT_PASSES;
        std::string variant = "gveopt2";  // Default to GVE-Leiden Opt2 (fastest + best quality)
        std::string resolution_mode = "auto";
        
        // Parse options: variant, resolution, max_iterations, max_passes
        // CLI format: -o 17:variant:resolution:max_iterations:max_passes
        // e.g., -o 17:hubsort:1.0:10:5
        // Resolution can be: "auto", "0", "dynamic", "dynamic:2.0", or numeric
        if (!reordering_options.empty() && !reordering_options[0].empty()) {
            variant = reordering_options[0];
        }
        if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
            const std::string& res_opt = reordering_options[1];
            resolution_mode = res_opt;
            
            // Handle special keywords
            if (res_opt == "auto" || res_opt == "0") {
                // Keep auto-resolution
            } else if (res_opt.rfind("dynamic", 0) == 0) {
                // Dynamic mode - extract initial value if provided (use underscore delimiter)
                size_t sep_pos = res_opt.find('_');
                if (sep_pos == std::string::npos) sep_pos = res_opt.find(':');
                if (sep_pos != std::string::npos && sep_pos + 1 < res_opt.size()) {
                    resolution = std::stod(res_opt.substr(sep_pos + 1));
                }
                // Keep auto-resolution as base, algorithms will handle dynamic
            } else {
                // Try numeric parsing
                try {
                    double parsed = std::stod(res_opt);
                    // Only override auto-resolution if value is in valid range (0, 3]
                    if (parsed > 0 && parsed <= 3) {
                        resolution = parsed;
                        resolution_mode = "fixed";
                    }
                    // If parsed <= 0 or > 3, keep auto-resolution
                } catch (...) {
                    // Parse error, keep auto-resolution
                }
            }
        }
        
        // For VIBE variants, skip integer parsing - VIBE has its own flexible parser
        // that handles options like "vibe:rabbit:bfs" where later args aren't integers
        bool isVibeVariant = (variant == "vibe" || variant.rfind("vibe", 0) == 0);
        
        if (!isVibeVariant) {
            // Only try to parse iterations/passes for non-VIBE variants
            if (reordering_options.size() > 2 && !reordering_options[2].empty()) {
                try { max_iterations = std::stoi(reordering_options[2]); } catch (...) {}
            }
            if (reordering_options.size() > 3 && !reordering_options[3].empty()) {
                try { max_passes = std::stoi(reordering_options[3]); } catch (...) {}
            }
        }
        
        printf("LeidenCSR: resolution=%.2f, max_passes=%d, max_iterations=%d, variant=%s\n", 
               resolution, max_passes, max_iterations, variant.c_str());
        
        // Prepare internal options: resolution (or mode string), max_iterations, max_passes
        std::vector<std::string> internal_options;
        // Pass through the original resolution string to let individual functions parse it
        if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
            internal_options.push_back(reordering_options[1]);
        } else {
            internal_options.push_back(std::to_string(resolution));
        }
        internal_options.push_back(std::to_string(max_iterations));
        internal_options.push_back(std::to_string(max_passes));
        
        if (variant == "dfs") {
            GenerateLeidenCSRMapping(g, new_ids, internal_options, 0);
        } else if (variant == "bfs") {
            GenerateLeidenCSRMapping(g, new_ids, internal_options, 1);
        } else if (variant == "hubsort") {
            GenerateLeidenCSRMapping(g, new_ids, internal_options, 2);
        } else if (variant == "fast") {
            // LeidenFast: Union-Find + Label Propagation
            GenerateLeidenFastMapping(g, new_ids, internal_options);
        } else if (variant == "modularity") {
            // True Leiden with modularity optimization
            // GenerateLeidenMapping2 expects: [resolution, max_passes, max_iterations]
            // Override to use 1 pass with 4 iterations for quality-focused detection
            std::vector<std::string> modularity_options;
            if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
                modularity_options.push_back(reordering_options[1]);  // resolution
            } else {
                modularity_options.push_back(std::to_string(resolution));
            }
            modularity_options.push_back("1");  // max_passes = 1
            modularity_options.push_back("4");  // max_iterations = 4
            GenerateLeidenMapping2(g, new_ids, modularity_options);
        } else if (variant == "gve") {
            // GVE-Leiden: True Leiden algorithm per ACM paper
            GenerateGVELeidenCSRMapping(g, new_ids, internal_options);
        } else if (variant == "gve2") {
            // GVE-Leiden2: Double-buffered super-graph (leiden.hxx style) - faster
            GenerateGVELeidenCSR2Mapping(g, new_ids, internal_options);
        } else if (variant == "gveopt") {
            // GVE-Leiden Optimized: Cache-optimized Leiden with prefetching
            GenerateGVELeidenOptMapping(g, new_ids, internal_options);
        } else if (variant == "gvedendo" || variant == "dendo") {
            // GVE-Leiden with incremental dendrogram (RabbitOrder-inspired)
            GenerateGVELeidenDendoMapping(g, new_ids, internal_options);
        } else if (variant == "gveoptdendo" || variant == "optdendo") {
            // GVE-Leiden Optimized with incremental dendrogram
            GenerateGVELeidenOptDendoMapping(g, new_ids, internal_options);
        } else if (variant == "gveoptsort" || variant == "optsort") {
            // GVE-Leiden Optimized with LeidenOrder-style multi-level sort (Strategy 1)
            GenerateGVELeidenOptSortMapping(g, new_ids, internal_options);
        } else if (variant == "gveopt2" || variant == "opt2") {
            // GVE-Leiden Opt2: CSR-based aggregation (faster than sort-based)
            GenerateGVELeidenOpt2Mapping(g, new_ids, internal_options);
        } else if (variant == "gveadaptive" || variant == "adaptive") {
            // GVE-Leiden Adaptive: Dynamic resolution adjustment each pass
            GenerateGVELeidenAdaptiveMapping(g, new_ids, internal_options);
        } else if (variant == "gvefast" || variant == "fast2") {
            // GVE-Leiden Fast: CSR buffer reuse (leiden.hxx style aggregation)
            GenerateGVELeidenFastMapping(g, new_ids, internal_options);
        } else if (variant == "gveturbo" || variant == "turbo") {
            // GVE-Leiden Turbo: Maximum speed (no refinement, early termination)
            GenerateGVELeidenTurboMapping(g, new_ids, internal_options);
        } else if (variant == "gverabbit" || variant == "rabbit") {
            // GVE-Rabbit: Hybrid RabbitOrder speed + Leiden quality
            GenerateGVERabbitMapping(g, new_ids, internal_options);
        } else if (variant == "faithful") {
            // Faithful 1:1 Leiden implementation (matches leiden.hxx exactly)
            GenerateFaithfulMapping(g, new_ids, internal_options);
        } else if (variant == "vibe" || variant.rfind("vibe:", 0) == 0 || variant.rfind("vibe", 0) == 0) {
            // VIBE: Fully modular implementation
            // Supports combinations like: vibe, vibe:dfs, vibe:rabbit:bfs, etc.
            // Pass ALL reordering_options (except variant) through VIBE config parser
            std::vector<std::string> vibe_options;
            
            // If variant has colon-separated parts after "vibe", split them first
            if (variant.length() > 4) {
                std::string rest = variant.substr(4); // Skip "vibe"
                if (!rest.empty() && rest[0] == ':') rest = rest.substr(1);
                std::stringstream ss(rest);
                std::string part;
                while (std::getline(ss, part, ':')) {
                    if (!part.empty()) vibe_options.push_back(part);
                }
            }
            
            // Add all options from reordering_options[1:] (skip the variant itself)
            // This captures things like "rabbit", "bfs", resolution, etc.
            for (size_t i = 1; i < reordering_options.size(); ++i) {
                if (!reordering_options[i].empty()) {
                    vibe_options.push_back(reordering_options[i]);
                }
            }
            
            // Let VIBE parser handle everything
            GenerateVibeMapping(g, new_ids, vibe_options);
        } else {
            // Default to GVE-Leiden (best quality)
            GenerateGVELeidenCSRMapping(g, new_ids, internal_options);
        }
    }
    
    /**
     * GenerateGVELeidenCSRMapping - True Leiden ordering using GVE-Leiden algorithm
     * Delegates to ::GenerateGVELeidenCSRMapping in reorder/reorder_leiden.h
     */
    void GenerateGVELeidenCSRMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        std::vector<std::string> reordering_options) {
        ::GenerateGVELeidenCSRMapping<K, NodeID_, DestID_>(g, new_ids, reordering_options);
    }

    /**
     * GenerateGVELeidenOptMapping - Optimized GVE-Leiden ordering
     * Delegates to ::GenerateGVELeidenOptMapping in reorder/reorder_leiden.h
     */
    void GenerateGVELeidenOptMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        std::vector<std::string> reordering_options) {
        ::GenerateGVELeidenOptMapping<K, NodeID_, DestID_>(g, new_ids, reordering_options);
    }

    /**
     * GenerateGVELeidenOpt2Mapping - GVE-Leiden with CSR-based aggregation
     * Delegates to ::GenerateGVELeidenOpt2Mapping in reorder/reorder_leiden.h
     */
    void GenerateGVELeidenOpt2Mapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        std::vector<std::string> reordering_options) {
        ::GenerateGVELeidenOpt2Mapping<K, NodeID_, DestID_>(g, new_ids, reordering_options);
    }

    /**
     * GenerateGVELeidenAdaptiveMapping - GVE-Leiden with dynamic resolution
     * Delegates to ::GenerateGVELeidenAdaptiveMapping in reorder/reorder_leiden.h
     */
    void GenerateGVELeidenAdaptiveMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        std::vector<std::string> reordering_options) {
        ::GenerateGVELeidenAdaptiveMapping<K, NodeID_, DestID_>(g, new_ids, reordering_options);
    }

    /**
     * GenerateGVELeidenOptSortMapping - Optimized GVE-Leiden with sort-based ordering
     * Delegates to ::GenerateGVELeidenOptSortMapping in reorder/reorder_leiden.h
     */
    void GenerateGVELeidenOptSortMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        std::vector<std::string> reordering_options) {
        ::GenerateGVELeidenOptSortMapping<K, NodeID_, DestID_>(g, new_ids, reordering_options);
    }

    /**
     * GenerateGVELeidenTurboMapping - Maximum speed GVE-Leiden variant
     * Delegates to ::GenerateGVELeidenTurboMapping in reorder/reorder_leiden.h
     */
    void GenerateGVELeidenTurboMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        std::vector<std::string> reordering_options) {
        ::GenerateGVELeidenTurboMapping<K, NodeID_, DestID_>(g, new_ids, reordering_options);
    }

    /**
     * GenerateGVELeidenFastMapping - CSR buffer reuse variant
     * Delegates to ::GenerateGVELeidenFastMapping in reorder/reorder_leiden.h
     */
    void GenerateGVELeidenFastMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        std::vector<std::string> reordering_options) {
        ::GenerateGVELeidenFastMapping<K, NodeID_, DestID_>(g, new_ids, reordering_options);
    }

    // ========================================================================
    // GVE-Rabbit Hybrid Algorithm
    // 
    // Combines RabbitOrder's fast incremental aggregation with Leiden's 
    // refinement phase for high-quality communities at near-RabbitOrder speed.
    //
    // Algorithm:
    // 1. FAST AGGREGATION (RabbitOrder-style):
    //    - Process vertices in degree order
    //    - For each vertex, find best neighbor maximizing ΔQ
    //    - Merge via lock-free union-find with CAS
    //    - Build dendrogram during merging
    //
    // 2. LEIDEN REFINEMENT (GVE-style):
    //    - For each community from step 1, run local refinement
    //    - Only isolated vertices (within their community bound) can move
    //    - This breaks poorly-connected clusters
    //
    // 3. ORDERING (RabbitOrder-style):
    //    - DFS traversal of dendrogram
    //    - Hub-first within each community
    // ========================================================================

    /**
     * GVE-Rabbit Hybrid Result Structure
     * NOTE: GVERabbitResult is in reorder/reorder_types.h
     * NOTE: GVERabbitCoreResult is in reorder/reorder_leiden.h (used by GVERabbitCore)
     */
    template <typename K = uint32_t>
    using GVERabbitResult = ::GVERabbitResult<K>;
    
    template <typename K = uint32_t>
    using GVERabbitCoreResult = ::GVERabbitCoreResult<K>;

    /**
     * GVE-Rabbit Core Algorithm
     * Delegates to ::GVERabbitCore in reorder/reorder_leiden.h
     */
    template <typename K = uint32_t>
    ::GVERabbitCoreResult<K> GVERabbitCore(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        double resolution = 1.0,
        int max_iterations = 5) {
        return ::GVERabbitCore<K, NodeID_, DestID_>(g, resolution, max_iterations);
    }

    /**
     * GenerateGVERabbitMapping - GVE-Rabbit hybrid ordering
     * Delegates to ::GenerateGVERabbitMapping in reorder/reorder_leiden.h
     */
    void GenerateGVERabbitMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        std::vector<std::string> reordering_options) {
        ::GenerateGVERabbitMapping<K, NodeID_, DestID_>(g, new_ids, reordering_options);
    }

    /**
     * GenerateFaithfulMapping - Faithful 1:1 Leiden implementation
     * Uses the new vibe::leiden() algorithm that exactly matches leiden.hxx
     * Delegates to ::GenerateFaithfulMapping in reorder/reorder_leiden.h
     */
    void GenerateFaithfulMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        std::vector<std::string> reordering_options) {
        ::GenerateFaithfulMapping<K, NodeID_, DestID_>(g, new_ids, reordering_options);
    }

    /**
     * GenerateVibeMapping - VIBE: Fully modular Leiden implementation
     * Configurable ordering and aggregation strategies
     * Delegates to ::GenerateVibeMapping in reorder/reorder_leiden.h
     */
    void GenerateVibeMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        std::vector<std::string> reordering_options,
        vibe::OrderingStrategy ordering = vibe::OrderingStrategy::HIERARCHICAL,
        vibe::AggregationStrategy aggregation = vibe::AggregationStrategy::LEIDEN_CSR) {
        ::GenerateVibeMapping<K, NodeID_, DestID_>(g, new_ids, reordering_options, ordering, aggregation);
    }

    // ========================================================================
    // RabbitOrderCSR - Native CSR implementation of Rabbit Order
    // 
    // Types and helper functions (RabbitCSRAtomPacked, RabbitCSRVertex, 
    // RabbitCSRGraph, rabbitCSRTraceCom, rabbitCSRCompactEdges, rabbitCSRUnite,
    // rabbitCSRFindBest, rabbitCSRMerge, rabbitCSRAggregate, rabbitCSRDescendants,
    // rabbitCSRComputePerm, rabbitCSRComputeModularityCSR) are now in:
    //   reorder/reorder_rabbit.h
    // ========================================================================


    /**
     * @brief RabbitOrderCSR - Native CSR implementation of Rabbit Order
     * Delegates to ::GenerateRabbitOrderCSRMapping in reorder/reorder_rabbit.h
     */
    void GenerateRabbitOrderCSRMapping(const CSRGraph<NodeID_, DestID_, invert>& g,
                                       pvector<NodeID_>& new_ids) {
        ::GenerateRabbitOrderCSRMapping<NodeID_, DestID_, WeightT_, invert>(g, new_ids);
    }


    /**
     * GenerateLeidenDendrogramMappingInternal - RabbitOrder-style ordering using Leiden communities
     * 
     * Flavor mapping (internal):
     *   0: Standard DFS traversal
     *   1: DFS with high-degree nodes first
     *   2: DFS with largest subtrees first  
     *   3: BFS by level
     *   4: Sort by (community, degree descending) - Hybrid
     */
    void GenerateLeidenDendrogramMappingInternal(
        CSRGraph<NodeID_, DestID_, invert> &g,
        pvector<NodeID_> &new_ids,
        std::vector<std::string> reordering_options,
        int flavor) {
        
        using K = uint32_t;
        using V = TYPE;
        
        Timer tm;
        int64_t num_nodes = g.num_nodes();
        
        // Default Leiden parameters - use unified constants for fair comparison
        double resolution = LeidenAutoResolution<NodeID_, DestID_>(g);
        int maxIterations = LEIDEN_DEFAULT_ITERATIONS;
        int maxPasses = LEIDEN_DEFAULT_PASSES;
        
        // Parse options if provided - handle "auto", "dynamic", or numeric
        if (!reordering_options.empty() && reordering_options[0].size() > 0) {
            const std::string& res_opt = reordering_options[0];
            if (res_opt == "auto" || res_opt == "0" || res_opt.rfind("dynamic", 0) == 0) {
                // Keep auto-resolution
            } else {
                try {
                    double parsed = std::stod(res_opt);
                    if (parsed > 0 && parsed <= 3) {
                        resolution = parsed;
                    }
                } catch (...) {
                    // Parse error, keep auto-resolution
                }
            }
        }
        if (reordering_options.size() > 1 && reordering_options[1].size() > 0) {
            try { maxIterations = std::stoi(reordering_options[1]); } catch (...) {}
        }
        if (reordering_options.size() > 2 && reordering_options[2].size() > 0) {
            try { maxPasses = std::stoi(reordering_options[2]); } catch (...) {}
        }
        
        PrintTime("Leiden Resolution", resolution);
        PrintTime("Leiden MaxIterations", maxIterations);
        PrintTime("Leiden MaxPasses", maxPasses);
        
        // Build Leiden-compatible graph (PARALLEL edge construction)
        tm.Start();
        int64_t num_edges = g.num_edges_directed();
        std::vector<std::tuple<size_t, size_t, double>> edges(num_edges);
        
        // Parallel edge list construction using CSR offsets
        #pragma omp parallel for
        for (int64_t u = 0; u < num_nodes; ++u) {
            NodeID_ out_start = g.out_offset(u);
            NodeID_ j = 0;
            for (DestID_ neighbor : g.out_neigh(u)) {
                if (g.is_weighted()) {
                    NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                    WeightT_ weight = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;
                    edges[out_start + j] = std::make_tuple((size_t)u, (size_t)dest, (double)weight);
                } else {
                    edges[out_start + j] = std::make_tuple((size_t)u, (size_t)neighbor, 1.0);
                }
                ++j;
            }
        }
        
        bool symmetric = false;
        bool weighted = g.is_weighted();
        DiGraph<K, None, V> x;
        readVecOmpW(x, edges, num_nodes, symmetric, weighted);
        edges.clear();
        x = symmetricizeOmp(x);
        
        tm.Stop();
        PrintTime("DiGraph Build Time", tm.Seconds());
        
        // Run Leiden algorithm
        tm.Start();
        std::random_device dev;
        std::default_random_engine rnd(dev());
        int repeat = 1;
        
        auto result = leidenStaticOmp<false, false>(
            rnd, x,
            {repeat, resolution, 1e-12, 0.8, 1.0, maxIterations, maxPasses});
        
        tm.Stop();
        PrintTime("Leiden Time", tm.Seconds());
        PrintTime("Leiden Passes", result.passes);
        PrintTime("Leiden Iterations", result.iterations);
        
        // Get community mappings per pass
        std::vector<std::vector<K>> communityMappingPerPass = x.communityMappingPerPass;
        PrintTime("Community Passes Stored", communityMappingPerPass.size());
        
        // Count unique communities in last pass for consistency with other Leiden variants
        size_t num_communities = 0;
        if (!communityMappingPerPass.empty()) {
            const auto& last_pass = communityMappingPerPass.back();
            std::set<K> unique_comms(last_pass.begin(), last_pass.end());
            num_communities = unique_comms.size();
        }
        PrintTime("Num Communities", num_communities);
        
        // Get degrees
        std::vector<K> degrees(num_nodes);
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; ++i) {
            degrees[i] = g.out_degree(i);
        }
        
        // Generate ordering based on flavor
        tm.Start();
        
        switch (flavor) {
            case 0: { // DFS
                std::cout << "Ordering Flavor: DFS (Parallel DFS)" << std::endl;
                std::vector<LeidenDendrogramNode> nodes;
                std::vector<int64_t> roots;
                buildLeidenDendrogram(nodes, roots, communityMappingPerPass, degrees, num_nodes);
                PrintTime("Dendrogram Nodes", nodes.size());
                PrintTime("Dendrogram Roots", roots.size());
                orderDendrogramDFSParallel(nodes, roots, new_ids, false, false);
                break;
            }
            case 1: { // DFSHub
                std::cout << "Ordering Flavor: DFSHub (Parallel DFS, hub-first)" << std::endl;
                std::vector<LeidenDendrogramNode> nodes;
                std::vector<int64_t> roots;
                buildLeidenDendrogram(nodes, roots, communityMappingPerPass, degrees, num_nodes);
                PrintTime("Dendrogram Nodes", nodes.size());
                PrintTime("Dendrogram Roots", roots.size());
                orderDendrogramDFSParallel(nodes, roots, new_ids, true, false);
                break;
            }
            case 2: { // DFSSize
                std::cout << "Ordering Flavor: DFSSize (Parallel DFS, size-first)" << std::endl;
                std::vector<LeidenDendrogramNode> nodes;
                std::vector<int64_t> roots;
                buildLeidenDendrogram(nodes, roots, communityMappingPerPass, degrees, num_nodes);
                PrintTime("Dendrogram Nodes", nodes.size());
                PrintTime("Dendrogram Roots", roots.size());
                orderDendrogramDFSParallel(nodes, roots, new_ids, false, true);
                break;
            }
            case 3: { // BFS
                std::cout << "Ordering Flavor: BFS (BFS by level)" << std::endl;
                std::vector<LeidenDendrogramNode> nodes;
                std::vector<int64_t> roots;
                buildLeidenDendrogram(nodes, roots, communityMappingPerPass, degrees, num_nodes);
                PrintTime("Dendrogram Nodes", nodes.size());
                PrintTime("Dendrogram Roots", roots.size());
                orderDendrogramBFS(nodes, roots, new_ids);
                break;
            }
            case 4: { // Hybrid (default)
                std::cout << "Ordering Flavor: Hybrid (community + degree)" << std::endl;
                orderLeidenHybridHubDFS(communityMappingPerPass, degrees, new_ids);
                break;
            }
            default:
                std::cerr << "Unknown LeidenDendrogram flavor: " << flavor << ", using DFSHub" << std::endl;
                std::vector<LeidenDendrogramNode> nodes;
                std::vector<int64_t> roots;
                buildLeidenDendrogram(nodes, roots, communityMappingPerPass, degrees, num_nodes);
                orderDendrogramDFSParallel(nodes, roots, new_ids, true, false);
        }
        
        tm.Stop();
        PrintTime("Ordering Time", tm.Seconds());
    }

    // CommunityFeatures is now defined in reorder/reorder_types.h
    // Alias for backward compatibility within BuilderBase
    using CommunityFeatures = ::CommunityFeatures;

    /**
     * Compute and print global graph topology features.
     * 
     * This computes features ONCE for the entire graph during the build phase,
     * BEFORE any reordering algorithm runs. These features describe the graph
     * structure itself and are used for:
     * 1. Perceptron-based algorithm selection
     * 2. Graph type detection
     * 3. Training weight data for Python scripts
     * 
     * Features computed:
     * - clustering_coeff: Global clustering coefficient (sampled)
     * - avg_path_length: Estimated average shortest path length
     * - diameter: Estimated graph diameter
     * - community_count: Number of communities (via fast Leiden)
     * - degree_variance: Normalized coefficient of variation of degrees
     * - hub_concentration: Fraction of edges from top 10% degree nodes
     * 
     * Output format (for Python parsing):
     *   Graph Topology Features:
     *   Clustering Coefficient: 0.1234
     *   Avg Path Length:        5.6789
     *   Diameter Estimate:      12
     *   Community Count:        45
     */
    void ComputeAndPrintGlobalTopologyFeatures(const CSRGraph<NodeID_, DestID_, invert>& g) {
        Timer t;
        t.Start();
        
        const int64_t num_nodes = g.num_nodes();
        const int64_t num_edges = g.num_edges_directed();
        
        if (num_nodes < 100) {
            // Too small for meaningful topology analysis
            return;
        }
        
        // ============================================================
        // 1. DEGREE STATISTICS (use shared utility function)
        // ============================================================
        auto deg_features = ::ComputeSampledDegreeFeatures(g, 5000, true);
        double avg_degree = deg_features.avg_degree;
        double degree_variance = deg_features.degree_variance;
        double hub_concentration = deg_features.hub_concentration;
        double clustering_coeff = deg_features.clustering_coeff;
        double packing_factor = deg_features.packing_factor;
        double forward_edge_fraction = deg_features.forward_edge_fraction;
        double working_set_ratio = deg_features.working_set_ratio;
        
        // ============================================================
        // 2. DIAMETER & AVG PATH LENGTH (single BFS from high-degree node)
        // ============================================================
        double avg_path_length = 0.0;
        int diameter_estimate = 0;
        
        if (num_nodes >= 500 && num_nodes <= 10000000) {  // Skip for very large graphs
            // Find highest degree node as BFS starting point
            const size_t SAMPLE_SIZE = std::min(static_cast<size_t>(5000), static_cast<size_t>(num_nodes));
            NodeID_ start_node = 0;
            int64_t max_deg = 0;
            for (size_t i = 0; i < std::min(SAMPLE_SIZE, static_cast<size_t>(num_nodes)); ++i) {
                NodeID_ node = (static_cast<size_t>(num_nodes) > SAMPLE_SIZE) ? 
                    static_cast<NodeID_>((i * num_nodes) / SAMPLE_SIZE) : static_cast<NodeID_>(i);
                if (g.out_degree(node) > max_deg) {
                    max_deg = g.out_degree(node);
                    start_node = node;
                }
            }
            
            // BFS with early termination for large graphs
            std::vector<int> dist(num_nodes, -1);
            std::queue<NodeID_> bfs_queue;
            bfs_queue.push(start_node);
            dist[start_node] = 0;
            
            double path_sum = 0.0;
            size_t path_count = 0;
            int max_dist = 0;
            const size_t MAX_BFS_VISITS = std::min(static_cast<size_t>(100000), static_cast<size_t>(num_nodes));
            size_t visits = 0;
            
            while (!bfs_queue.empty() && visits < MAX_BFS_VISITS) {
                NodeID_ curr = bfs_queue.front();
                bfs_queue.pop();
                
                for (DestID_ neighbor : g.out_neigh(curr)) {
                    NodeID_ dest = static_cast<NodeID_>(neighbor);
                    if (dist[dest] == -1) {
                        dist[dest] = dist[curr] + 1;
                        bfs_queue.push(dest);
                        path_sum += dist[dest];
                        ++path_count;
                        ++visits;
                        if (dist[dest] > max_dist) {
                            max_dist = dist[dest];
                        }
                    }
                }
            }
            
            avg_path_length = (path_count > 0) ? path_sum / path_count : 1.0;
            diameter_estimate = max_dist;
        } else if (num_nodes > 10000000) {
            // For very large graphs, use rough estimates
            avg_path_length = std::log2(num_nodes);  // Small-world approximation
            diameter_estimate = static_cast<int>(avg_path_length * 2);
        }
        
        // ============================================================
        // 3. COMMUNITY COUNT (fast Leiden for estimate)
        // ============================================================
        int community_count = 1;  // Default: 1 large component
        
        // For community count, we can use a rough estimate based on graph density
        // More accurate counting is done during AdaptiveOrder when needed
        double density = static_cast<double>(num_edges) / (static_cast<double>(num_nodes) * (num_nodes - 1));
        if (density < 0.001 && num_nodes > 1000) {
            // Sparse graph likely has multiple communities
            // Rough estimate based on graph structure
            community_count = static_cast<int>(std::sqrt(num_nodes / 100.0)) + 1;
        } else if (hub_concentration > 0.5) {
            // Hub-dominated graph
            community_count = static_cast<int>(hub_concentration * 10) + 1;
        }
        
        t.Stop();
        
        // ============================================================
        // OUTPUT (format for Python parsing)
        // ============================================================
        std::cout << "=== Graph Topology Features ===" << std::endl;
        PrintTime("Clustering Coefficient", clustering_coeff);
        PrintTime("Avg Path Length", avg_path_length);
        PrintTime("Diameter Estimate", diameter_estimate);
        PrintTime("Community Count Estimate", community_count);
        PrintTime("Degree Variance", degree_variance);
        PrintTime("Hub Concentration", hub_concentration);
        PrintTime("Avg Degree", avg_degree);
        PrintTime("Graph Density", density);
        PrintTime("Packing Factor", packing_factor);
        PrintTime("Forward Edge Fraction", forward_edge_fraction);
        PrintTime("Working Set Ratio", working_set_ratio);
        PrintTime("Topology Analysis Time", t.Seconds());
        std::cout << "===============================" << std::endl;
    }

    /**
     * Perceptron-style Algorithm Selector
     * 
     * Uses learned weights from multi-algorithm correlation analysis to
     * predict the best reordering algorithm based on community features.
     * 
     * The perceptron computes a score for each candidate algorithm:
     * 
     *   score = bias 
     *         + w_modularity * modularity 
     *         + w_log_nodes * log10(nodes)
     *         + w_log_edges * log10(edges)
     *         + w_density * density
     *         + w_avg_degree * avg_degree / 100
     *         + w_degree_variance * degree_variance
     *         + w_hub_concentration * hub_concentration
     *         + w_clustering_coeff * clustering_coeff      (NEW)
     *         + w_avg_path_length * avg_path_length / 10   (NEW)
     *         + w_diameter * diameter / 50                 (NEW)
     *         + w_community_count * log10(community_count) (NEW)
     *         + w_reorder_time * reorder_time              (NEW)
     * 
     * The score is then multiplied by benchmark_weights[current_benchmark]
     * to adjust for benchmark-specific performance characteristics.
     * 
     * The algorithm with the highest score is selected.
     * 
     * Weights were learned from benchmarking multiple graph algorithms
     * (BFS, PR, SSSP, BC, CC) across graphs with varying modularity.
     * 
     * WEIGHT CATEGORIES:
     * - Core weights: bias, w_modularity, w_log_nodes, w_log_edges, w_density,
     *                 w_avg_degree, w_degree_variance, w_hub_concentration
     * - Extended graph structure: w_clustering_coeff, w_avg_path_length, 
     *                             w_diameter, w_community_count
     * - Cache impact: cache_l1_impact, cache_l2_impact, cache_l3_impact,
     *                 cache_dram_penalty (used during training to adjust bias)
     * - Reorder time: w_reorder_time (penalty for slow reordering)
     * - Benchmark-specific: benchmark_weights[pr|bfs|cc|sssp|bc] (multiplier per benchmark)
     * 
     * WEIGHT LOADING: Weights can be loaded from a JSON file at runtime.
     * If the file exists, it overrides the hardcoded defaults.
     * Environment var: PERCEPTRON_WEIGHTS_FILE can override the default path.
     * 
     * Format: {"AlgorithmName": {"bias": X, "w_modularity": X, ...}, ...}
     * 
     * AUTO-CLUSTERING TYPE SYSTEM:
     * The system uses auto-generated type files for specialized tuning:
     * - scripts/weights/active/type_0.json    (Cluster 0 weights)
     * - scripts/weights/active/type_1.json    (Cluster 1 weights)
     * - scripts/weights/active/type_N.json    (Additional clusters as needed)
     * - scripts/weights/active/type_registry.json (maps graph names → type + centroids)
     * 
     * At runtime, the system:
     * 1. Computes graph features (modularity, density, etc.)
     * 2. Uses FindBestTypeFromFeatures() to find the best matching cluster
     * 3. Loads weights from the corresponding type_N.json file
     * 4. Falls back to hardcoded defaults if no type file exists
     */
    
    // Weight path constants - now defined in reorder/reorder_types.h
    // Aliased here for backward compatibility
    static constexpr const char* DEFAULT_WEIGHTS_FILE = ::DEFAULT_WEIGHTS_FILE;
    static constexpr const char* WEIGHTS_DIR = ::WEIGHTS_DIR;
    static constexpr const char* TYPE_WEIGHTS_DIR = ::TYPE_WEIGHTS_DIR;
    
    /**
     * Graph type enum for graph-type-specific weight selection
     * 
     * Different graph types have different structural properties that
     * benefit from different reordering strategies:
     * 
     * - SOCIAL: High modularity, community structure, power-law degrees
     *           Best: LeidenDendrogram, RabbitOrder
     * - ROAD: Mesh-like, low modularity, planar structure
     *         Best: RCMOrder (bandwidth reduction)
     * - WEB: High hub concentration, bow-tie structure
     *        Best: HubClusterDBG, HubSort
     * - POWERLAW: RMAT-like, highly skewed degree distribution
     *             Best: RabbitOrder, HubCluster
     * - UNIFORM: Random graphs, uniform degree distribution
     *            Best: Original or light reordering (DBG)
     * - GENERIC: Unknown or mixed - use default weights
     * 
     * GraphType enum is now defined in reorder/reorder_graphbrew.h
     * Import into class scope for backward compatibility.
     */
    using GraphType = ::GraphType;
    static constexpr GraphType GRAPH_GENERIC  = ::GRAPH_GENERIC;
    static constexpr GraphType GRAPH_SOCIAL   = ::GRAPH_SOCIAL;
    static constexpr GraphType GRAPH_ROAD     = ::GRAPH_ROAD;
    static constexpr GraphType GRAPH_WEB      = ::GRAPH_WEB;
    static constexpr GraphType GRAPH_POWERLAW = ::GRAPH_POWERLAW;
    static constexpr GraphType GRAPH_UNIFORM  = ::GRAPH_UNIFORM;
    
    /**
     * Convert graph type enum to string (delegates to global function)
     */
    static std::string GraphTypeToString(GraphType type) {
        return ::GraphTypeToString(type);
    }
    
    /**
     * Convert string to graph type enum (delegates to global function)
     */
    static GraphType GetGraphType(const std::string& name) {
        return ::GetGraphType(name);
    }
    
    /**
     * Auto-detect graph type from graph features (delegates to global function)
     */
    static GraphType DetectGraphType(double modularity, double degree_variance, 
                                      double hub_concentration, double avg_degree,
                                      size_t num_nodes) {
        return ::DetectGraphType(modularity, degree_variance, hub_concentration, 
                                 avg_degree, num_nodes);
    }
    
    // BenchmarkType is now defined in reorder/reorder_types.h
    // Alias for backward compatibility within BuilderBase
    using BenchmarkType = ::BenchmarkType;
    
    // Use global GetBenchmarkType function
    static BenchmarkType GetBenchmarkType(const std::string& name) {
        return ::GetBenchmarkType(name);
    }
    
    /**
     * Selection mode for AdaptiveOrder algorithm selection
     * 
     * SelectionMode enum is now defined in reorder/reorder_graphbrew.h
     * Import into class scope for backward compatibility.
     */
    using SelectionMode = ::SelectionMode;
    static constexpr SelectionMode MODE_FASTEST_REORDER    = ::MODE_FASTEST_REORDER;
    static constexpr SelectionMode MODE_FASTEST_EXECUTION  = ::MODE_FASTEST_EXECUTION;
    static constexpr SelectionMode MODE_BEST_ENDTOEND      = ::MODE_BEST_ENDTOEND;
    static constexpr SelectionMode MODE_BEST_AMORTIZATION  = ::MODE_BEST_AMORTIZATION;
    
    /**
     * Convert selection mode to string (delegates to global function)
     */
    static std::string SelectionModeToString(SelectionMode mode) {
        return ::SelectionModeToString(mode);
    }
    
    /**
     * Convert string to selection mode (delegates to global function)
     */
    static SelectionMode GetSelectionMode(const std::string& name) {
        return ::GetSelectionMode(name);
    }
    
    // PerceptronWeights is now defined in reorder/reorder_types.h
    // Alias for backward compatibility within BuilderBase
    using PerceptronWeights = ::PerceptronWeights;

    /**
     * Get default perceptron weights for all reordering algorithms.
     * 
     * Delegates to the global ::GetPerceptronWeights() function in reorder_types.h.
     * This ensures a single source of truth for default weights across the codebase.
     */
    static const std::map<ReorderingAlgo, PerceptronWeights>& GetPerceptronWeights() {
        return ::GetPerceptronWeights();
    }

    /**
     * Simple JSON parser for perceptron weights file
     * 
     * Parses a JSON file with format:
     * {
     *   "ORIGINAL": {"bias": 1.0, "w_modularity": 0.3, ...},
     *   "LeidenHybrid": {"bias": 0.95, ...},
     *   ...
     * }
     * 
     * Delegates to the global ::ParseWeightsFromJSON in reorder_types.h.
     */
    static bool ParseWeightsFromJSON(const std::string& json_content, 
                                      std::map<ReorderingAlgo, PerceptronWeights>& weights) {
        return ::ParseWeightsFromJSON(json_content, weights);
    }

    /**
     * Load perceptron weights from file, or return defaults if file doesn't exist.
     * 
     * Checks for weights file in this order:
     * 1. Path from PERCEPTRON_WEIGHTS_FILE environment variable
     * 2. scripts/weights/active/type_N.json files (via LoadPerceptronWeightsForGraphType)
     * 3. If neither exists, returns hardcoded defaults from GetPerceptronWeights()
     */
    static std::map<ReorderingAlgo, PerceptronWeights> LoadPerceptronWeights(bool verbose = false) {
        return LoadPerceptronWeightsForGraphType(GRAPH_GENERIC, verbose);
    }
    
    /**
     * Select algorithm with fastest reorder time based on w_reorder_time weight.
     * 
     * Delegates to the global ::SelectFastestReorderFromWeights in reorder_types.h.
     * The w_reorder_time weight encodes how fast each algorithm is at reordering:
     * - Higher (less negative) = faster reordering
     * - Lower (more negative) = slower reordering
     */
    static ReorderingAlgo SelectFastestReorderFromWeights(
        const std::map<ReorderingAlgo, PerceptronWeights>& weights, bool verbose = false) {
        return ::SelectFastestReorderFromWeights(weights, verbose);
    }
    
    /**
     * Find the best matching type file from the type registry.
     * 
    /**
     * Find the best matching type file from the type registry.
     * Delegates to the global ::FindBestTypeFromFeatures in reorder_types.h.
     */
    static std::string FindBestTypeFromFeatures(
        double modularity, double degree_variance, double hub_concentration,
        double avg_degree, size_t num_nodes, size_t num_edges, bool verbose = false,
        double clustering_coeff = 0.0) {
        return ::FindBestTypeFromFeatures(modularity, degree_variance, hub_concentration,
                                          avg_degree, num_nodes, num_edges, verbose,
                                          clustering_coeff);
    }
    
    /**
     * Find the best matching type AND return the distance.
     * Delegates to the global ::FindBestTypeWithDistance in reorder_types.h.
     */
    static std::string FindBestTypeWithDistance(
        double modularity, double degree_variance, double hub_concentration,
        double avg_degree, size_t num_nodes, size_t num_edges,
        double& out_distance, bool verbose = false,
        double clustering_coeff = 0.0) {
        return ::FindBestTypeWithDistance(modularity, degree_variance, hub_concentration,
                                          avg_degree, num_nodes, num_edges, out_distance, verbose,
                                          clustering_coeff);
    }
    
    // Threshold constant now defined in reorder/reorder_types.h
    // Alias for backward compatibility
    static constexpr double UNKNOWN_TYPE_DISTANCE_THRESHOLD = ::UNKNOWN_TYPE_DISTANCE_THRESHOLD;
    
    /**
     * Check if a graph is far from known types (delegates to global function)
     */
    static bool IsDistantGraphType(double type_distance) {
        return ::IsDistantGraphType(type_distance);
    }
    
    /**
     * Load perceptron weights for a specific graph type.
     * Delegates to the global ::LoadPerceptronWeightsForGraphType in reorder_types.h.
     */
    static std::map<ReorderingAlgo, PerceptronWeights> LoadPerceptronWeightsForGraphType(
        GraphType graph_type, bool verbose = false) {
        return ::LoadPerceptronWeightsForGraphType(graph_type, verbose);
    }
    
    /**
     * Load perceptron weights using graph features to find the best type match.
     * Delegates to the global ::LoadPerceptronWeightsForFeatures in reorder_types.h.
     */
    static std::map<ReorderingAlgo, PerceptronWeights> LoadPerceptronWeightsForFeatures(
        double modularity, double degree_variance, double hub_concentration,
        double avg_degree, size_t num_nodes, size_t num_edges, bool verbose = false,
        double clustering_coeff = 0.0) {
        return ::LoadPerceptronWeightsForFeatures(modularity, degree_variance, hub_concentration,
                                                   avg_degree, num_nodes, num_edges, verbose,
                                                   clustering_coeff);
    }
    
    /**
     * Get cached weights for a specific graph type.
     * Delegates to the global ::GetCachedWeights in reorder_types.h.
     */
    static const std::map<ReorderingAlgo, PerceptronWeights>& GetCachedWeights(
        GraphType graph_type, bool verbose_first_load = false) {
        return ::GetCachedWeights(graph_type, verbose_first_load);
    }

    /**
     * Select best reordering algorithm using perceptron scores.
     * Delegates to the global ::SelectReorderingPerceptron in reorder_types.h.
     */
    ReorderingAlgo SelectReorderingPerceptron(const CommunityFeatures& feat, 
                                               BenchmarkType bench = BENCH_GENERIC,
                                               GraphType graph_type = GRAPH_GENERIC) {
        return ::SelectReorderingPerceptron(feat, bench, graph_type);
    }
    
    /**
     * Overload that accepts benchmark name as string
     */
    ReorderingAlgo SelectReorderingPerceptron(const CommunityFeatures& feat,
                                               const std::string& benchmark_name,
                                               GraphType graph_type = GRAPH_GENERIC) {
        return SelectReorderingPerceptron(feat, GetBenchmarkType(benchmark_name), graph_type);
    }
    
    /**
     * Select best reordering algorithm using feature-based type matching.
     * Delegates to the global ::SelectReorderingPerceptronWithFeatures in reorder_types.h.
     */
    ReorderingAlgo SelectReorderingPerceptronWithFeatures(
        const CommunityFeatures& feat,
        double global_modularity, double global_degree_variance,
        double global_hub_concentration, size_t num_nodes, size_t num_edges,
        BenchmarkType bench = BENCH_GENERIC) {
        return ::SelectReorderingPerceptronWithFeatures(feat, global_modularity, global_degree_variance,
                                                         global_hub_concentration, num_nodes, num_edges, bench);
    }
    
    /**
     * Select best reordering algorithm with MODE-AWARE selection.
     * Delegates to the global ::SelectReorderingWithMode in reorder_types.h.
     */
    ReorderingAlgo SelectReorderingWithMode(
        const CommunityFeatures& feat,
        double global_modularity, double global_degree_variance,
        double global_hub_concentration, size_t num_nodes, size_t num_edges,
        SelectionMode mode, const std::string& graph_name = "",
        BenchmarkType bench = BENCH_GENERIC, bool verbose = false) {
        return ::SelectReorderingWithMode(feat, global_modularity, global_degree_variance,
                                           global_hub_concentration, num_nodes, num_edges,
                                           mode, graph_name, bench, verbose);
    }

    CommunityFeatures ComputeCommunityFeatures(
        const std::vector<NodeID_>& comm_nodes,
        const CSRGraph<NodeID_, DestID_, invert>& g,
        const std::unordered_set<NodeID_>& node_set,
        bool compute_extended = true)
    {
        // Delegate to standalone function in reorder_types.h
        return ::ComputeCommunityFeaturesStandalone<NodeID_, DestID_, invert>(
            comm_nodes, g, node_set, compute_extended);
    }

    /**
    /**
     * Compute dynamic minimum community size threshold.
     * Delegates to the global ::ComputeDynamicMinCommunitySize in reorder_types.h.
     */
    static size_t ComputeDynamicMinCommunitySize(size_t num_nodes, 
                                                  size_t num_communities,
                                                  size_t avg_community_size = 0) {
        return ::ComputeDynamicMinCommunitySize(num_nodes, num_communities, avg_community_size);
    }
    
    /**
     * Compute dynamic threshold for when to apply local reordering.
     * Delegates to graphbrew::ComputeDynamicLocalReorderThreshold
     */
    static size_t ComputeDynamicLocalReorderThreshold(size_t num_nodes,
                                                       size_t num_communities,
                                                       size_t avg_community_size = 0) {
        return graphbrew::ComputeDynamicLocalReorderThreshold(num_nodes, num_communities, avg_community_size);
    }

    /**
     * Select best reordering algorithm based on community features.
     * Delegates to the global ::SelectBestReorderingForCommunity in reorder_types.h.
     */
    ReorderingAlgo SelectBestReorderingForCommunity(CommunityFeatures feat, 
                                                     double global_modularity,
                                                     double global_degree_variance,
                                                     double global_hub_concentration,
                                                     double global_avg_degree,
                                                     size_t num_nodes, size_t num_edges,
                                                     BenchmarkType bench = BENCH_GENERIC,
                                                     GraphType graph_type = GRAPH_GENERIC,
                                                     SelectionMode mode = MODE_FASTEST_EXECUTION,
                                                     const std::string& graph_name = "",
                                                     size_t dynamic_min_size = 0) {
        return ::SelectBestReorderingForCommunity(feat, global_modularity, global_degree_variance,
                                                   global_hub_concentration, global_avg_degree,
                                                   num_nodes, num_edges, bench, graph_type,
                                                   mode, graph_name, dynamic_min_size);
    }

    // ========================================================================
    // ADAPTIVE REORDERING - Delegates to standalone implementations
    // ========================================================================
    // See reorder_adaptive.h for full implementations using GVE-Leiden (native CSR).
    // These delegates maintain backward compatibility with existing code.
    
    /**
     * Main entry point for Adaptive reordering - delegates to standalone.
     * Format: -o 14[:max_depth[:resolution[:min_recurse_size[:selection_mode[:graph_name]]]]]
     */
    void GenerateAdaptiveMapping(CSRGraph<NodeID_, DestID_, invert> &g,
                                 pvector<NodeID_> &new_ids, bool useOutdeg,
                                 std::vector<std::string> reordering_options) {
        ::GenerateAdaptiveMappingStandalone<NodeID_, DestID_, WeightT_, invert>(
            g, new_ids, useOutdeg, reordering_options);
    }
    
    /**
     * Full-graph adaptive mode - delegates to standalone.
     */
    void GenerateAdaptiveMappingFullGraph(CSRGraph<NodeID_, DestID_, invert> &g,
                                          pvector<NodeID_> &new_ids, bool useOutdeg,
                                          std::vector<std::string> reordering_options) {
        ::GenerateAdaptiveMappingFullGraphStandalone<NodeID_, DestID_, WeightT_, invert>(
            g, new_ids, useOutdeg, reordering_options);
    }
    
    /**
     * Recursive per-community adaptive mode - delegates to standalone.
     */
    void GenerateAdaptiveMappingRecursive(
        const CSRGraph<NodeID_, DestID_, invert> &g,
        pvector<NodeID_> &new_ids, 
        bool useOutdeg,
        std::vector<std::string> reordering_options,
        int depth = 0,
        bool verbose = true,
        SelectionMode selection_mode = MODE_FASTEST_EXECUTION,
        const std::string& graph_name = "") {
        ::GenerateAdaptiveMappingRecursiveStandalone<NodeID_, DestID_, WeightT_, invert>(
            g, new_ids, useOutdeg, reordering_options, depth, verbose, selection_mode, graph_name);
    }


    //==========================================================================
    // GRAPHBREW UNIFIED ENTRY POINT
    //
    // Extended GraphBrewOrder with configurable clustering and final reordering.
    // Format: -o 12:cluster_variant:final_algo:resolution:levels
    //
    // Cluster variants:
    //   leiden  - Original external Leiden library (default, backward compatible)
    //   gve     - GVE-Leiden CSR (native, fast)
    //   gveopt  - GVE-Leiden Optimized (native, cache-optimized)
    //   rabbit  - RabbitOrder clustering
    //   hubcluster - Simple hub-based clustering
    //
    // Final algorithm: Any algorithm ID 0-17 (default: 8 = RabbitOrder)
    // Resolution: Leiden resolution parameter (default: auto)
    // Levels: Recursion depth (default: 2)
    //
    // Examples:
    //   -o 12                      # Default: leiden, RabbitOrder, 2 levels
    //   -o 12:gve                  # GVE clustering, RabbitOrder final
    //   -o 12:gve:6                # GVE clustering, HubSortDBG final
    //   -o 12:gveopt:8:1.0:3       # GVEOpt, RabbitOrder, resolution 1.0, 3 levels
    //==========================================================================
    
    // ========================================================================
    // GRAPHBREW REORDERING - Delegates to standalone implementations
    // ========================================================================
    
    /**
     * Legacy GraphBrew entry point - delegates to unified implementation.
     * Previously used external igraph Leiden library, now uses GVE-Leiden.
     * 
     * Note: numLevels and recursion params preserved for backward compatibility
     * but are now handled internally by the unified implementation.
     */
    void GenerateGraphBrewMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        bool useOutdeg,
        std::vector<std::string> reordering_options,
        int numLevels = 1,
        bool recursion = false) {
        // Delegate to unified implementation (GVE-Leiden based)
        ::GenerateGraphBrewMappingUnifiedStandalone<NodeID_, DestID_, WeightT_, invert>(
            g, new_ids, useOutdeg, reordering_options);
    }
    
    /**
     * Unified GraphBrew entry point - delegates to standalone implementation.
     * See reorder_graphbrew.h for full implementation.
     */
    void GenerateGraphBrewMappingUnified(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        bool useOutdeg,
        std::vector<std::string> reordering_options) {
        ::GenerateGraphBrewMappingUnifiedStandalone<NodeID_, DestID_, WeightT_, invert>(
            g, new_ids, useOutdeg, reordering_options);
    }
    
    /**
     * GraphBrew with GVE-Leiden clustering - delegates to standalone implementation.
     */
    void GenerateGraphBrewGVEMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        bool useOutdeg,
        std::vector<std::string> reordering_options,
        int numLevels = 2,
        bool recursion = false,
        bool use_optimized = false) {
        ::GenerateGraphBrewGVEMappingStandalone<NodeID_, DestID_, WeightT_, invert>(
            g, new_ids, useOutdeg, reordering_options, numLevels, recursion, use_optimized);
    }
    
    /**
     * GraphBrew with RabbitOrder-style clustering - delegates to GVE with tuned params.
     */
    void GenerateGraphBrewRabbitMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        bool useOutdeg,
        std::vector<std::string> reordering_options,
        int numLevels = 2) {
        printf("GraphBrewRabbit: Using GVE with RabbitOrder-optimized parameters\n");
        auto opts = reordering_options;
        if (opts.size() > 2) opts[2] = "0.5";  // Lower resolution
        ::GenerateGraphBrewGVEMappingStandalone<NodeID_, DestID_, WeightT_, invert>(
            g, new_ids, useOutdeg, opts, numLevels, false, true);
    }
    
    /**
     * GraphBrew with HubCluster-based clustering - delegates to standalone implementation.
     */
    void GenerateGraphBrewHubClusterMapping(
        const CSRGraph<NodeID_, DestID_, invert>& g,
        pvector<NodeID_>& new_ids,
        bool useOutdeg,
        std::vector<std::string> reordering_options,
        int numLevels = 2) {
        ::GenerateGraphBrewHubClusterMappingStandalone<NodeID_, DestID_, WeightT_, invert>(
            g, new_ids, useOutdeg, reordering_options, numLevels);
    }
    

};

#endif // BUILDER_H_
