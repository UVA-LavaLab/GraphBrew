// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef BUILDER_H_
#define BUILDER_H_

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <map>
#include <numeric>
#include <parallel/algorithm>
#include <parallel/numeric>
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


    //TRUST Partitions
    int64_t trust_vertex_count, trust_edge_count;
    NodeID_ trust_endofprocess = -1;
    struct trust_edge_list
    {
        NodeID_ vertexID;
        vector<NodeID_> edge;
        NodeID_ newid;
    };

    vector<trust_edge_list> trust_vertex;
    vector<trust_edge_list> trust_vertexb;

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

    CSRGraph<NodeID_, DestID_, invert> MakeLocalGraphFromEL(EdgeList &el)
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

    EdgeList MakeTrustDirectELFromGraph(const CSRGraph<NodeID_, DestID_, invert> &g)
    {
        int64_t num_edges = g.num_edges_directed();
        int64_t num_nodes = g.num_nodes();
        EdgeList el(num_edges);

        std::vector<int> a(num_nodes, 0);

        // Parallelized step to mark vertices that appear in the edge list
        #pragma omp parallel for
        for (NodeID_ i = 0; i < num_nodes; ++i)
        {
            for (DestID_ neighbor : g.out_neigh(i))
            {
                #pragma omp atomic write
                a[i] = 1;

                if (g.is_weighted())
                {
                    NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                    #pragma omp atomic write
                    a[dest] = 1;
                }
                else
                {
                    #pragma omp atomic write
                    a[neighbor] = 1;
                }
            }
        }

        __gnu_parallel::partial_sum(a.begin(), a.end(), a.begin());

        // Parallel loop to construct the edge list
        #pragma omp parallel for
        for (NodeID_ i = 0; i < num_nodes; ++i)
        {
            NodeID_ out_start = g.out_offset(i);
            NodeID_ j = 0;
            for (DestID_ neighbor : g.out_neigh(i))
            {
                NodeID_ new_u = a[i];

                if (g.is_weighted())
                {
                    NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                    WeightT_ weight =
                        static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;
                    NodeID_ new_v = a[dest];

                    if(new_u > new_v)
                        el[out_start + j] = Edge(new_v, NodeWeight<NodeID_, WeightT_>(new_u, weight));
                    else
                        el[out_start + j] = Edge(new_u, NodeWeight<NodeID_, WeightT_>(new_v, weight));
                }
                else
                {
                    NodeID_ new_v = a[neighbor];
                    if(new_u > new_v)
                        el[out_start + j] = Edge(new_v, new_u);
                    else
                        el[out_start + j] = Edge(new_u, new_v);
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

    std::vector<EdgeList>
    MakeTrustPartitionedEL(const CSRGraph<NodeID_, DestID_, invert> &g,
                           int p_n = 1, int p_m = 1)
    {

        int num_partitions = p_n * p_m;
        int num_threads = omp_get_max_threads();
        std::vector<EdgeList> partitions_el(num_partitions);

        // Local edge lists for each thread
        std::vector<std::vector<EdgeList>> local_partitions_el(num_threads);

        for (int thread_id = 0; thread_id < num_threads; ++thread_id)
        {
            local_partitions_el[thread_id].resize(num_partitions);
        }

        trust_endofprocess = std::numeric_limits<NodeID_>::max();

        #pragma omp parallel
        {
            #pragma omp for schedule(static)
            for (NodeID_ i = 0; i < g.num_nodes(); ++i)
            {
                if (g.out_degree(i) < 2)
                {
                    // Use atomic compare-and-swap to update trust_endofprocess
                    int current_value = trust_endofprocess;
                    while (i < current_value && !__sync_bool_compare_and_swap(&trust_endofprocess, current_value, i))
                    {
                        current_value = trust_endofprocess;
                    }
                }
            }
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
                        NodeID_ dest = static_cast<NodeWeight<NodeID_,
                                WeightT_>>(j).v;
                        WeightT_ weight =
                            static_cast<NodeWeight<NodeID_, WeightT_>>(j).w;
                        int
                        partition_idx = (src % p_n) * p_m + (dest % p_m);
                        Edge e =
                            Edge(src / p_n, NodeWeight<NodeID_, WeightT_>(dest / p_m, weight));
                        local_partitions_el[thread_id][partition_idx].push_back(e);
                    }
                    else
                    {
                        NodeID_ dest = j;
                        int partition_idx = (src % p_n) * p_m + (dest % p_m);
                        Edge e = Edge(src / p_n, dest / p_m);
                        // Edge e = Edge(src, dest);
                        // std::cout << partition_idx << " p: " <<  src << " -> " <<
                        //           dest << std::endl;
                        local_partitions_el[thread_id][partition_idx].push_back(e);
                    }
                }
            }
        }

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

        return partitions_el;
    }

    static bool trust_cmp1(trust_edge_list a, trust_edge_list b)
    {
        return a.edge.size() < b.edge.size() ;
    }
    static bool trust_cmp2(trust_edge_list a, trust_edge_list b)
    {
        return a.edge.size() > b.edge.size() ;
    }
    // vector<edge_list> vertexb;

    CSRGraph<NodeID_, DestID_, invert> MakeTrustPreprocessStep(const CSRGraph<NodeID_, DestID_, invert> &g)
    {
        // int64_t num_edges = g.num_edges_directed();
        int64_t num_nodes = g.num_nodes();
        CSRGraph<NodeID_, DestID_, invert> sort_g;

        // Vector to store node-degree pairs
        std::vector<std::pair<NodeID_, int64_t>> node_degree_pairs(num_nodes);

        // Parallel loop to calculate the degree of each node
        #pragma omp parallel for
        for (NodeID_ i = 0; i < num_nodes; ++i)
        {
            int64_t degree = g.out_degree(i);
            node_degree_pairs[i] = std::make_pair(i, degree);
        }

        // Sort the node-degree pairs in descending order of degree
        __gnu_parallel::stable_sort(node_degree_pairs.begin(), node_degree_pairs.end(),
                                    [](const std::pair<NodeID_, int64_t> &a, const std::pair<NodeID_, int64_t> &b)
        {
            if (a.second == 0) return false;  // a stays if it has degree 0
            if (b.second == 0) return true;   // b goes to the end if it has degree 0
            return a.second < b.second;       // otherwise sort in descending orde
        });

        // for (size_t i = 0; i < node_degree_pairs.size(); ++i)
        // {
        //     std::cout << "Node " << std::to_string(node_degree_pairs[i].first)
        //               << ": Degree " << std::to_string(node_degree_pairs[i].second)
        //               << std::endl;
        // }

        // Create a mapping from old IDs to new IDs based on sorted order
        std::vector<NodeID_> old_to_new_id(num_nodes);
        #pragma omp parallel for
        for (NodeID_ i = 0; i < num_nodes; ++i)
        {
            old_to_new_id[node_degree_pairs[i].first] = i;
        }

        // g.PrintTopologyOrdered(node_degree_pairs);
        // Create local edge lists for each thread
        std::vector<EdgeList> local_edge_lists(omp_get_max_threads());

        // Parallel loop to orient the edges and create local edge lists
        #pragma omp parallel
        {
            int thread_id = omp_get_thread_num();
            EdgeList &local_el = local_edge_lists[thread_id];

            #pragma omp for nowait
            for (NodeID_ i = 0; i < num_nodes; ++i)
            {
                NodeID_ new_u = node_degree_pairs[i].first;
                for (DestID_ neighbor : g.out_neigh(new_u))
                {
                    if (g.is_weighted())
                    {
                        NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                        WeightT_ weight = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;
                        NodeID_ new_v = old_to_new_id[dest];
                        if (i < new_v)  // Orient edges from lower to higher new ID
                        {
                            Edge e = Edge(new_u, NodeWeight<NodeID_, WeightT_>(dest, weight));
                            local_el.push_back(e);
                        }
                    }
                    else
                    {
                        NodeID_ new_v = old_to_new_id[neighbor];
                        if (i < new_v)  // Orient edges from lower to higher new ID
                        {
                            Edge e = Edge(new_u, neighbor);
                            local_el.push_back(e);
                        }
                    }
                }
            }
        }

        // Merge local edge lists into a single edge list
        EdgeList el;
        for (const auto &local_el : local_edge_lists)
        {
            el.insert(el.end(), local_el.begin(), local_el.end());
        }

        // PrintEdgeList(el);

        sort_g = MakeLocalGraphFromEL(el);

        sort_g.PrintTopology();

        // Vector to store node-degree pairs
        std::vector<std::pair<NodeID_, int64_t>> node_degree_pairs_sort(sort_g.num_nodes());

        // Parallel loop to calculate the degree of each node
        #pragma omp parallel for
        for (NodeID_ i = 0; i < sort_g.num_nodes(); ++i)
        {
            int64_t degree = sort_g.out_degree(i);
            node_degree_pairs_sort[i] = std::make_pair(i, degree);
        }

        // Sort the node-degree pairs in descending order of degree
        __gnu_parallel::stable_sort(node_degree_pairs_sort.begin(), node_degree_pairs_sort.end(),
                                    [](const std::pair<NodeID_, int64_t> &a, const std::pair<NodeID_, int64_t> &b)
        {
            if (a.second == 0) return false;  // a stays if it has degree 0
            if (b.second == 0) return true;   // b goes to the end if it has degree 0
            return a.second > b.second;       // otherwise sort in descending orde
        });

        sort_g.PrintTopologyOrdered(node_degree_pairs_sort);

        pvector<NodeID_> new_ids(sort_g.num_nodes(), -1);
        GenerateSortMapping(sort_g, new_ids, true, false);
        sort_g = RelabelByMapping(sort_g, new_ids);

        return sort_g;
    }

    void printTrustVertexStructure()
    {
        for (const auto &v : trust_vertex)
        {
            std::cout << "Trust Vertex " << v.vertexID << ": ";
            for (const auto &e : v.edge)
            {
                std::cout << e << " ";
            }
            std::cout << std::endl;
        }
    }

    void trust_orientation()
    {
        NodeID_ *a = new NodeID_[trust_vertex_count];
        for (NodeID_ i = 0; i < trust_vertex_count; i++)
        {
            a[trust_vertex[i].vertexID] = i;
        }

        for (NodeID_ i = 0; i < trust_vertex_count; i++)
        {
            vector<NodeID_> x(trust_vertex[i].edge);
            trust_vertex[i].edge.clear();
            while (!x.empty())
            {
                NodeID_ v = x.back();
                x.pop_back();
                if (a[v] > i) trust_vertex[i].edge.push_back(v);
            }
        }

        delete[] a;
    }

    void trust_reassignID()
    {
        NodeID_ k1 = 0, k2 = -1, k3 = -1;
        for (NodeID_ i = 0; i < trust_vertex_count; i++)
        {
            trust_vertex[i].newid = -1;
            if (k2 == -1 && trust_vertex[i].edge.size() <= 100)
                k2 = i;

            if (k3 == -1 && trust_vertex[i].edge.size() < 2)
                k3 = i;
        }
        std::cout << k2 << ' ' << k3 << std::endl;
        NodeID_ s2 = k2, s3 = k3;
        for (NodeID_ i = 0; i < trust_vertex_count; i++)
        {
            if (trust_vertex[i].edge.size() <= 2) break;
            for (size_t j = 0; j < trust_vertex[i].edge.size(); j++)
            {
                NodeID_ v = trust_vertex[i].edge[j];
                if (trust_vertex[v].newid == -1)
                {
                    if (v >= s3)
                    {
                        trust_vertex[v].newid = k3;
                        k3++;
                    }
                    else if (v >= s2)
                    {
                        trust_vertex[v].newid = k2;
                        k2++;
                    }
                    else
                    {
                        trust_vertex[v].newid = k1;
                        k1++;
                    }
                }
            }
        }
        for (NodeID_ i = 0; i < trust_vertex_count; i++)
        {
            int u = trust_vertex[i].newid;
            if (u == -1)
            {
                if (i >= s3)
                {
                    trust_vertex[i].newid = k3;
                    k3++;
                }
                else if (i >= s2)
                {
                    trust_vertex[i].newid = k2;
                    k2++;
                }
                else
                {
                    trust_vertex[i].newid = k1;
                    k1++;
                }
            }
        }
        trust_vertexb.swap(trust_vertex);
        trust_vertex.resize(trust_vertex_count);

        for (NodeID_ i = 0; i < trust_vertex_count; i++)
        {
            NodeID_ u = trust_vertexb[i].newid;

            for (size_t j = 0; j < trust_vertexb[i].edge.size(); j++)
            {
                NodeID_ v = trust_vertexb[i].edge[j];
                v = trust_vertexb[v].newid;
                // cout<<u<<' '<<v<<endl;
                trust_vertex[u].edge.push_back(v);
            }
        }

    }

    void trust_computeCSR()
    {
        NodeID_ *a = new NodeID_[trust_vertex_count];
        for (NodeID_ i = 0; i < trust_vertex_count; i++)
        {
            a[trust_vertex[i].vertexID] = i;
        }
        for (NodeID_ i = 0; i < trust_vertex_count; i++)
        {
            for (size_t j = 0; j < trust_vertex[i].edge.size(); j++)
            {
                trust_vertex[i].edge[j] = a[trust_vertex[i].edge[j]];
            }
            trust_vertex[i].vertexID = i;
        }

        trust_reassignID();

        delete[] a;
    }

    CSRGraph<NodeID_, DestID_, invert> MakeTrustOrigianlPreprocessStep(const CSRGraph<NodeID_, DestID_, invert> &g)
    {

        CSRGraph<NodeID_, DestID_, invert> sort_g;
        trust_vertex_count = g.num_nodes();
        trust_vertex.resize(trust_vertex_count);
        trust_edge_count = g.num_edges_directed();

        #pragma omp parallel for
        for (NodeID_ i = 0; i < trust_vertex_count; ++i)
        {
            trust_vertex[i].vertexID = i;
            trust_vertex[i].edge.insert(trust_vertex[i].edge.end(), g.out_neigh(i).begin(), g.out_neigh(i).end());
            trust_vertex[i].newid = -1; // Assuming newid is to be initialized to -1
        }

        __gnu_parallel::stable_sort(trust_vertex.begin(), trust_vertex.end(), trust_cmp1);

        if(cli_.logging_en())
        {
            std::cout << "Before orientation:" << std::endl;
            printTrustVertexStructure();
        }

        trust_orientation();

        if(cli_.logging_en())
        {
            std::cout << "After orientation:" << std::endl;
            printTrustVertexStructure();
        }

        __gnu_parallel::stable_sort(trust_vertex.begin(), trust_vertex.end(), trust_cmp2);

        if(cli_.logging_en())
        {
            std::cout << "After sort:" << std::endl;
            printTrustVertexStructure();
        }

        trust_computeCSR();

        if(cli_.logging_en())
        {
            std::cout << "After computeCSR:" << std::endl;
            printTrustVertexStructure();
        }

        trust_edge_count = 0;
        #pragma omp parallel for reduction(+: trust_edge_count)
        for (NodeID_ i = 0; i < trust_vertex_count; ++i)
        {
            trust_edge_count += trust_vertex[i].edge.size();
        }

        if(cli_.logging_en())
        {
            std::cout << "trust_edge_count: " << std::to_string(trust_edge_count) << std::endl;
        }

        vector<NodeID_> trust_vertex_edge_sizes(trust_vertex_count);
        #pragma omp parallel for
        for (NodeID_ i = 0; i < trust_vertex_count; ++i)
        {
            trust_vertex_edge_sizes[i] = trust_vertex[i].edge.size();
        }

        std::vector<size_t> trust_vertex_offset(trust_vertex_count + 1, 0);
        __gnu_parallel::partial_sum(trust_vertex_edge_sizes.begin(), trust_vertex_edge_sizes.end(), trust_vertex_offset.begin() + 1);

        EdgeList trust_el(trust_edge_count);

        #pragma omp parallel for
        for (NodeID_ i = 0; i < trust_vertex_count; ++i)
        {
            NodeID_ out_start = trust_vertex_offset[i];
            NodeID_ j = 0;
            for (NodeID_ neighbor : trust_vertex[i].edge)
            {
                if (g.is_weighted())
                {
                    WeightT_ weight = 1;
                    trust_el[out_start + j] = Edge(i, NodeWeight<NodeID_, WeightT_>(neighbor, weight));
                }
                else
                {
                    trust_el[out_start + j] = Edge(i, neighbor);
                }
                ++j;
            }
        }

        sort_g  = MakeLocalGraphFromEL(trust_el);

        if(cli_.logging_en())
        {
            std::cout << "After sort_g MakeLocalGraphFromEL:" << std::endl;
            printTrustVertexStructure();

            // PrintEdgeList(trust_el);

            sort_g.PrintTopology();
        }
        return sort_g;
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
        CSRGraph<NodeID_, DestID_, invert> dir_g;
        CSRGraph<NodeID_, DestID_, invert> prep_g;
        EdgeList uni_el;
        EdgeList dir_el;

        // Start fromDirectToUndirect Step
        dir_el = MakeTrustDirectELFromGraph(g);
        dir_g  = MakeLocalGraphFromEL(dir_el);
        // dir_g  = SquishGraph(dir_g);

        if(cli_.logging_en())
        {
            dir_g.PrintTopology();
            std::cout << std::endl;
        }

        NodeID_ min_seen = FindMinNodeID(dir_el);

        uni_el = MakeUniDirectELFromGraph(dir_g, min_seen);
        uni_g  = MakeLocalGraphFromEL(uni_el);
        // uni_g  = SquishGraph(uni_g);

        if(cli_.logging_en())
        {
            uni_g.PrintTopology();
            std::cout << std::endl;
        }

        // End fromDirectToUndirect Step
        prep_g = MakeTrustOrigianlPreprocessStep(uni_g);

        if(cli_.logging_en())
        {
            prep_g.PrintTopology();
            std::cout << std::endl;
        }

        partitions_el =
            MakeTrustPartitionedEL(prep_g, p_n, p_m);

        // Create graphs from each partition in column-major order and add to
        // partitions_g
        for (int row = 0; row < p_n; ++row)
        {
            for (int col = 0; col < p_m; ++col)
            {
                int idx = row * p_m + col;
                CSRGraph<NodeID_, DestID_, invert> partition_g =
                    MakeLocalGraphFromEL(partitions_el[idx]);
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
        case AdaptiveOrder:
            return "AdaptiveOrder";
        case LeidenDFS:
            return "LeidenDFS";
        case LeidenDFSHub:
            return "LeidenDFSHub";
        case LeidenDFSSize:
            return "LeidenDFSSize";
        case LeidenBFS:
            return "LeidenBFS";
        case LeidenHybrid:
            return "LeidenHybrid";
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

    void GenerateMapping(CSRGraph<NodeID_, DestID_, invert> &g,
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
        {
            // MakeLocalGraphFromEL(partitions_el_dest[i]);
            pvector<NodeID_> new_ids_local(g.num_nodes(), -1);
            pvector<NodeID_> new_ids_local_2(g.num_nodes(), -1);
            // partition_g = SquishGraph(partition_g);
            GenerateSortMappingRabbit(g, new_ids_local, true, true);
            CSRGraph<NodeID_, DestID_, invert> g_trans = RelabelByMapping(g, new_ids_local);
            GenerateRabbitOrderMapping(g_trans, new_ids_local_2);

            #pragma omp parallel for
            for (NodeID_ n = 0; n < g.num_nodes(); n++)
            {
                new_ids[n] = new_ids_local_2[new_ids_local[n]];
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
            GenerateLeidenMapping(g, new_ids, reordering_options);
            break;
        case LeidenDFS:
            GenerateLeidenDendrogramMapping(g, new_ids, reordering_options, LeidenDFS);
            break;
        case LeidenDFSHub:
            GenerateLeidenDendrogramMapping(g, new_ids, reordering_options, LeidenDFSHub);
            break;
        case LeidenDFSSize:
            GenerateLeidenDendrogramMapping(g, new_ids, reordering_options, LeidenDFSSize);
            break;
        case LeidenBFS:
            GenerateLeidenDendrogramMapping(g, new_ids, reordering_options, LeidenBFS);
            break;
        case LeidenHybrid:
            GenerateLeidenDendrogramMapping(g, new_ids, reordering_options, LeidenHybrid);
            break;
        case GraphBrewOrder:
            GenerateGraphBrewMapping(g, new_ids, useOutdeg, reordering_options, 2);
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

        omp_set_nested(1);
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
            // MakeLocalGraphFromEL(partitions_el_dest[i]);
            pvector<NodeID_> new_ids_local(g_org.num_nodes(), -1);
            pvector<NodeID_> new_ids_local_2(g_org.num_nodes(), -1);
            // partition_g = SquishGraph(partition_g);
            GenerateSortMappingRabbit(g, new_ids_local, true, true);
            g = RelabelByMapping(g, new_ids_local);

            GenerateRabbitOrderMapping(g, new_ids_local_2);

            #pragma omp parallel for
            for (NodeID_ n = 0; n < g_org.num_nodes(); n++)
            {
                new_ids[n] = new_ids_local_2[new_ids_local[n]];
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
            GenerateLeidenMapping(g, new_ids, reordering_options);
            break;
        case LeidenDFS:
            GenerateLeidenDendrogramMapping(g, new_ids, reordering_options, LeidenDFS);
            break;
        case LeidenDFSHub:
            GenerateLeidenDendrogramMapping(g, new_ids, reordering_options, LeidenDFSHub);
            break;
        case LeidenDFSSize:
            GenerateLeidenDendrogramMapping(g, new_ids, reordering_options, LeidenDFSSize);
            break;
        case LeidenBFS:
            GenerateLeidenDendrogramMapping(g, new_ids, reordering_options, LeidenBFS);
            break;
        case LeidenHybrid:
            GenerateLeidenDendrogramMapping(g, new_ids, reordering_options, LeidenHybrid);
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
        case 15:
            return AdaptiveOrder;
        case 16:
            return LeidenDFS;
        case 17:
            return LeidenDFSHub;
        case 18:
            return LeidenDFSSize;
        case 19:
            return LeidenBFS;
        case 20:
            return LeidenHybrid;
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

        auto custom_comparator = [lesser](const degree_nodeid_t &a, const degree_nodeid_t &b)
        {
            // if (a.first == 0 && b.first == 0) return false; // Keep relative order of zero-degree nodes
            // if (a.first == 0) return false; // Zero-degree nodes should be "greater"
            // if (b.first == 0) return true;  // Zero-degree nodes should be "greater"
            return lesser ? (a.first < b.first) : (a.first > b.first);

            // return a.first > b.first;
        };

        __gnu_parallel::stable_sort(degree_id_pairs.begin(), degree_id_pairs.end(), custom_comparator);


        #pragma omp parallel for
        for (int64_t n = 0; n < num_nodes; ++n)
        {
            new_ids[degree_id_pairs[n].second] = n;
        }

        pvector<degree_nodeid_t>().swap(degree_id_pairs);

        t.Stop();
        PrintTime("Sort Map Time", t.Seconds());
    }

    void GenerateSortMappingRabbit(const CSRGraph<NodeID_, DestID_, invert> &g,
                                   pvector<NodeID_> &new_ids, bool useOutdeg,
                                   bool lesser = false)
    {

        typedef std::tuple<int64_t, int64_t, NodeID_> degree_nodeid_t;

        Timer t;
        t.Start();

        int64_t num_nodes = g.num_nodes();
        pvector<degree_nodeid_t> degree_id_pairs(num_nodes);

        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v)
        {
            int64_t out_degree_v = g.out_degree(v);
            int64_t in_degree_v = g.in_degree(v);
            degree_id_pairs[v] = std::make_tuple(out_degree_v, in_degree_v, v);
        }

        auto custom_comparator = [](const degree_nodeid_t &a, const degree_nodeid_t &b)
        {
            int64_t out_a = std::get<0>(a);
            int64_t out_b = std::get<0>(b);
            int64_t in_a = std::get<1>(a);
            int64_t in_b = std::get<1>(b);

            if (out_a == 0 && in_a == 0) return false; // Keep relative order of zero-degree nodes
            if (out_b == 0 && in_b == 0) return true;  // Zero-degree nodes should be "greater"

            if (out_a != out_b) return out_a > out_b; // Primary sort by out-degree
            return in_a > in_b;                       // Secondary sort by in-degree
        };

        __gnu_parallel::stable_sort(degree_id_pairs.begin(), degree_id_pairs.end(), custom_comparator);

        #pragma omp parallel for
        for (int64_t n = 0; n < num_nodes; ++n)
        {
            new_ids[std::get<2>(degree_id_pairs[n])] = n;
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

        // if (const size_t c = count_unused_id(n, edges))
        // {
        // std::cerr << "WARNING: " << c << "/" << n << " vertex IDs are unused"
        // << " (zero-degree vertices or noncontiguous IDs?)\n";
        // }

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

        // if (const size_t c = count_unused_id(n, edges))
        // {
        // std::cerr << "WARNING: " << c << "/" << n << " vertex IDs are unused"
        //           << " (zero-degree vertices or noncontiguous IDs?)\n";
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
        double m2 = 0.0; // total weight of the (bidirectional) edges

        // Find max community ID for flat array sizing
        rabbit_order::vint max_com = 0;
        #pragma omp parallel for reduction(max : max_com)
        for (rabbit_order::vint v = 0; v < n; ++v) {
            if (coms[v] > max_com) max_com = coms[v];
        }
        const size_t com_array_size = static_cast<size_t>(max_com) + 1;

        // Use flat arrays instead of hash maps for better cache performance
        std::vector<double> degs_all(com_array_size, 0.0);
        std::vector<double> degs_loop(com_array_size, 0.0);

        // Thread-local arrays to avoid critical sections
        const int num_threads_mod = omp_get_max_threads();
        std::vector<std::vector<double>> thread_degs_all(num_threads_mod, 
            std::vector<double>(com_array_size, 0.0));
        std::vector<std::vector<double>> thread_degs_loop(num_threads_mod, 
            std::vector<double>(com_array_size, 0.0));

        #pragma omp parallel reduction(+ : m2)
        {
            int tid = omp_get_thread_num();
            auto& local_all = thread_degs_all[tid];
            auto& local_loop = thread_degs_loop[tid];

            #pragma omp for schedule(dynamic, 1024)
            for (rabbit_order::vint v = 0; v < n; ++v)
            {
                const rabbit_order::vint c = coms[v];
                for (const auto& e : adj[v])
                {
                    m2 += e.second;
                    local_all[c] += e.second;
                    if (coms[e.first] == c)
                        local_loop[c] += e.second;
                }
            }
        }

        // Parallel reduction of thread-local arrays
        #pragma omp parallel for schedule(static)
        for (size_t c = 0; c < com_array_size; ++c) {
            for (int t = 0; t < num_threads_mod; ++t) {
                degs_all[c] += thread_degs_all[t][c];
                degs_loop[c] += thread_degs_loop[t][c];
            }
        }

        // Compute modularity
        double q = 0.0;
        #pragma omp parallel for reduction(+ : q) schedule(static, 1024)
        for (size_t c = 0; c < com_array_size; ++c)
        {
            const double all = degs_all[c];
            const double loop = degs_loop[c];
            if (all > 0.0) {
                q += loop / m2 - (all / m2) * (all / m2);
            }
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

        int window = 7;

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges_directed();

        std::vector<std::pair<int, int>> edges(num_edges);
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
                    // WeightT_ weight =
                    //     static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;

                    std::pair<int, int>edge =
                        std::make_pair(i, dest);
                    edges[out_start + j] = edge;
                }
                else
                {
                    std::pair<int, int> edge =
                        std::make_pair(i, neighbor);
                    edges[out_start + j] = edge;
                }
                ++j;
            }
        }

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

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges_directed();

        std::vector<std::pair<int, int>> edges(num_edges);
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
                    // WeightT_ weight =
                    //     static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;

                    std::pair<int, int>edge =
                        std::make_pair(i, dest);
                    edges[out_start + j] = edge;
                }
                else
                {
                    std::pair<int, int> edge =
                        std::make_pair(i, neighbor);
                    edges[out_start + j] = edge;
                }
                ++j;
            }
        }

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

        using V = TYPE;
        install_sigsegv();

        double resolution = 0.75;
        int maxIterations = 30;
        /** Maximum number of passes [10]. */
        int maxPasses = 30;

        if (!reordering_options.empty())
        {
            resolution = std::stod(reordering_options[0]);
            resolution = (resolution > 3) ? 1.0 : resolution;
        }
        if (reordering_options.size() > 1)
        {
            maxIterations = std::stoi(reordering_options[1]);
        }
        if (reordering_options.size() > 2)
        {
            maxPasses = std::stoi(reordering_options[2]);
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
        const size_t last_pass_col = (2 + actual_passes - 1);  // Last (coarsest) pass column
        
        // Sort primarily by last-pass community (coarsest level) 
        // with degree as secondary key within communities
        __gnu_parallel::sort(sort_indices.begin(), sort_indices.end(),
            [&communityDataFlat, stride, last_pass_col](size_t a, size_t b) {
                // Primary: sort by coarsest community
                K comm_a = communityDataFlat[last_pass_col * stride + a];
                K comm_b = communityDataFlat[last_pass_col * stride + b];
                if (comm_a != comm_b) {
                    return comm_a < comm_b;
                }
                // Secondary: within community, sort by degree (descending)
                // This puts hubs at the start of each community - better cache locality
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

        tm.Stop();
        PrintTime("GenID Time", tm.Seconds());
        PrintTime("Num Passes", x.communityMappingPerPass.size());
        PrintTime("Resolution", resolution);
    }

    //==========================================================================
    // LEIDEN DENDROGRAM-BASED ORDERING (RabbitOrder-style traversal)
    //==========================================================================

    /**
     * Dendrogram node for hierarchical community structure
     */
    struct LeidenDendrogramNode {
        int64_t parent;      
        int64_t first_child; 
        int64_t sibling;     
        int64_t vertex_id;   // Original vertex ID (-1 for internal nodes)
        size_t subtree_size; 
        double weight;       // Degree sum
        int level;           
        
        LeidenDendrogramNode() : parent(-1), first_child(-1), sibling(-1), 
                           vertex_id(-1), subtree_size(1), weight(0.0), level(0) {}
    };

    /**
     * Build dendrogram from Leiden's per-pass community mappings
     */
    template<typename K>
    void buildLeidenDendrogram(
        std::vector<LeidenDendrogramNode>& nodes,
        std::vector<int64_t>& roots,
        const std::vector<std::vector<K>>& communityMappingPerPass,
        const std::vector<K>& degrees,
        size_t num_vertices) {
        
        const size_t num_passes = communityMappingPerPass.size();
        
        // Create leaf nodes for all vertices
        nodes.resize(num_vertices);
        #pragma omp parallel for
        for (size_t v = 0; v < num_vertices; ++v) {
            nodes[v].vertex_id = v;
            nodes[v].subtree_size = 1;
            nodes[v].weight = degrees[v];
            nodes[v].level = 0;
        }
        
        if (num_passes == 0) {
            // No community structure - each vertex is its own root
            for (size_t v = 0; v < num_vertices; ++v) {
                roots.push_back(v);
            }
            return;
        }
        
        // Build hierarchy from finest to coarsest
        std::vector<int64_t> current_nodes(num_vertices);
        std::iota(current_nodes.begin(), current_nodes.end(), 0);
        
        for (size_t pass = 0; pass < num_passes; ++pass) {
            const auto& comm_map = communityMappingPerPass[pass];
            
            // Group current nodes by community at this pass
            std::unordered_map<K, std::vector<int64_t>> community_members;
            for (int64_t node_id : current_nodes) {
                // Find representative vertex for this node
                int64_t v = node_id;
                while (v >= 0 && nodes[v].vertex_id < 0) {
                    v = nodes[v].first_child;
                }
                if (v >= 0 && nodes[v].vertex_id >= 0) {
                    size_t vertex = nodes[v].vertex_id;
                    if (vertex < comm_map.size()) {
                        community_members[comm_map[vertex]].push_back(node_id);
                    }
                }
            }
            
            // Create internal nodes for multi-member communities
            std::vector<int64_t> next_nodes;
            for (auto& pair : community_members) {
                auto& members = pair.second;
                if (members.size() == 1) {
                    next_nodes.push_back(members[0]);
                } else {
                    // Create internal node
                    LeidenDendrogramNode internal;
                    internal.vertex_id = -1;
                    internal.level = pass + 1;
                    internal.subtree_size = 0;
                    internal.weight = 0.0;
                    
                    int64_t internal_id = nodes.size();
                    nodes.push_back(internal);
                    
                    // Sort members by weight (degree) descending for hub-first
                    std::sort(members.begin(), members.end(), [&nodes](int64_t a, int64_t b) {
                        return nodes[a].weight > nodes[b].weight;
                    });
                    
                    // Link children
                    int64_t prev_sibling = -1;
                    for (int64_t child_id : members) {
                        nodes[child_id].parent = internal_id;
                        if (nodes[internal_id].first_child == -1) {
                            nodes[internal_id].first_child = child_id;
                        }
                        if (prev_sibling >= 0) {
                            nodes[prev_sibling].sibling = child_id;
                        }
                        prev_sibling = child_id;
                        
                        nodes[internal_id].subtree_size += nodes[child_id].subtree_size;
                        nodes[internal_id].weight += nodes[child_id].weight;
                    }
                    
                    next_nodes.push_back(internal_id);
                }
            }
            current_nodes = std::move(next_nodes);
        }
        
        // Remaining nodes are roots
        for (int64_t node_id : current_nodes) {
            roots.push_back(node_id);
        }
        
        // Sort roots by subtree size
        std::sort(roots.begin(), roots.end(), [&nodes](int64_t a, int64_t b) {
            return nodes[a].subtree_size > nodes[b].subtree_size;
        });
    }

    /**
     * DFS ordering of dendrogram
     */
    void orderDendrogramDFS(
        const std::vector<LeidenDendrogramNode>& nodes,
        const std::vector<int64_t>& roots,
        pvector<NodeID_>& new_ids,
        bool hub_first,
        bool size_first) {
        
        NodeID_ current_id = 0;
        std::deque<int64_t> stack;
        
        for (int64_t root : roots) {
            stack.push_back(root);
            
            while (!stack.empty()) {
                int64_t node_id = stack.back();
                stack.pop_back();
                
                const auto& node = nodes[node_id];
                
                if (node.vertex_id >= 0) {
                    new_ids[node.vertex_id] = current_id++;
                } else {
                    std::vector<int64_t> children;
                    int64_t child = node.first_child;
                    while (child >= 0) {
                        children.push_back(child);
                        child = nodes[child].sibling;
                    }
                    
                    if (hub_first) {
                        std::sort(children.begin(), children.end(), 
                             [&nodes](int64_t a, int64_t b) {
                                 return nodes[a].weight > nodes[b].weight;
                             });
                    } else if (size_first) {
                        std::sort(children.begin(), children.end(),
                             [&nodes](int64_t a, int64_t b) {
                                 return nodes[a].subtree_size > nodes[b].subtree_size;
                             });
                    }
                    
                    for (auto it = children.rbegin(); it != children.rend(); ++it) {
                        stack.push_back(*it);
                    }
                }
            }
        }
    }

    /**
     * BFS ordering of dendrogram (by level)
     */
    void orderDendrogramBFS(
        const std::vector<LeidenDendrogramNode>& nodes,
        const std::vector<int64_t>& roots,
        pvector<NodeID_>& new_ids) {
        
        NodeID_ current_id = 0;
        std::deque<int64_t> queue;
        
        for (int64_t root : roots) {
            queue.push_back(root);
        }
        
        while (!queue.empty()) {
            int64_t node_id = queue.front();
            queue.pop_front();
            
            const auto& node = nodes[node_id];
            
            if (node.vertex_id >= 0) {
                new_ids[node.vertex_id] = current_id++;
            } else {
                int64_t child = node.first_child;
                while (child >= 0) {
                    queue.push_back(child);
                    child = nodes[child].sibling;
                }
            }
        }
    }

    /**
     * Hybrid ordering: sort by (community, degree descending)
     */
    template<typename K>
    void orderLeidenHybridHubDFS(
        const std::vector<std::vector<K>>& communityMappingPerPass,
        const std::vector<K>& degrees,
        pvector<NodeID_>& new_ids) {
        
        const size_t n = degrees.size();
        
        if (communityMappingPerPass.empty()) {
            // No community - sort by degree
            std::vector<std::pair<K, size_t>> deg_vertex(n);
            #pragma omp parallel for
            for (size_t v = 0; v < n; ++v) {
                deg_vertex[v] = {degrees[v], v};
            }
            __gnu_parallel::sort(deg_vertex.begin(), deg_vertex.end(), std::greater<>());
            #pragma omp parallel for
            for (size_t i = 0; i < n; ++i) {
                new_ids[deg_vertex[i].second] = i;
            }
            return;
        }
        
        // Get last-pass communities
        const auto& last_pass = communityMappingPerPass.back();
        
        // Create (vertex, community, degree) tuples
        std::vector<std::tuple<size_t, K, K>> vertices(n);
        #pragma omp parallel for
        for (size_t v = 0; v < n; ++v) {
            vertices[v] = std::make_tuple(v, last_pass[v], degrees[v]);
        }
        
        // Sort by (community, degree descending)
        __gnu_parallel::sort(vertices.begin(), vertices.end(),
            [](const auto& a, const auto& b) {
                if (std::get<1>(a) != std::get<1>(b)) return std::get<1>(a) < std::get<1>(b);
                return std::get<2>(a) > std::get<2>(b);
            });
        
        #pragma omp parallel for
        for (size_t i = 0; i < n; ++i) {
            new_ids[std::get<0>(vertices[i])] = i;
        }
    }

    /**
     * GenerateLeidenDendrogramMapping - RabbitOrder-style ordering using Leiden communities
     * 
     * Flavor mapping:
     *   LeidenDFS     (16): Standard DFS traversal
     *   LeidenDFSHub  (17): DFS with high-degree nodes first
     *   LeidenDFSSize (18): DFS with largest subtrees first  
     *   LeidenBFS     (19): BFS by level
     *   LeidenHybrid  (20): Sort by (community, degree descending)
     */
    void GenerateLeidenDendrogramMapping(
        CSRGraph<NodeID_, DestID_, invert> &g,
        pvector<NodeID_> &new_ids,
        std::vector<std::string> reordering_options,
        ReorderingAlgo flavor) {
        
        using K = uint32_t;
        using V = TYPE;
        
        Timer tm;
        int64_t num_nodes = g.num_nodes();
        
        // Default Leiden parameters
        double resolution = 1.0;
        int maxIterations = 20;
        int maxPasses = 10;
        
        // Parse options if provided
        if (!reordering_options.empty() && reordering_options[0].size() > 0) {
            resolution = std::stod(reordering_options[0]);
        }
        if (reordering_options.size() > 1 && reordering_options[1].size() > 0) {
            maxIterations = std::stoi(reordering_options[1]);
        }
        if (reordering_options.size() > 2 && reordering_options[2].size() > 0) {
            maxPasses = std::stoi(reordering_options[2]);
        }
        
        PrintTime("Leiden Resolution", resolution);
        PrintTime("Leiden MaxIterations", maxIterations);
        PrintTime("Leiden MaxPasses", maxPasses);
        
        // Build Leiden-compatible graph
        tm.Start();
        std::vector<std::tuple<size_t, size_t, double>> edges;
        for (int64_t u = 0; u < num_nodes; ++u) {
            for (DestID_ neighbor : g.out_neigh(u)) {
                if (g.is_weighted()) {
                    NodeID_ dest = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                    WeightT_ weight = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;
                    edges.emplace_back((size_t)u, (size_t)dest, (double)weight);
                } else {
                    edges.emplace_back((size_t)u, (size_t)neighbor, 1.0);
                }
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
        
        // Get degrees
        std::vector<K> degrees(num_nodes);
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; ++i) {
            degrees[i] = g.out_degree(i);
        }
        
        // Generate ordering based on flavor
        tm.Start();
        
        switch (flavor) {
            case LeidenDFS: {
                std::cout << "Ordering Flavor: LeidenDFS (Standard DFS)" << std::endl;
                std::vector<LeidenDendrogramNode> nodes;
                std::vector<int64_t> roots;
                buildLeidenDendrogram(nodes, roots, communityMappingPerPass, degrees, num_nodes);
                PrintTime("Dendrogram Nodes", nodes.size());
                PrintTime("Dendrogram Roots", roots.size());
                orderDendrogramDFS(nodes, roots, new_ids, false, false);
                break;
            }
            case LeidenDFSHub: {
                std::cout << "Ordering Flavor: LeidenDFSHub (DFS, hub-first)" << std::endl;
                std::vector<LeidenDendrogramNode> nodes;
                std::vector<int64_t> roots;
                buildLeidenDendrogram(nodes, roots, communityMappingPerPass, degrees, num_nodes);
                PrintTime("Dendrogram Nodes", nodes.size());
                PrintTime("Dendrogram Roots", roots.size());
                orderDendrogramDFS(nodes, roots, new_ids, true, false);
                break;
            }
            case LeidenDFSSize: {
                std::cout << "Ordering Flavor: LeidenDFSSize (DFS, size-first)" << std::endl;
                std::vector<LeidenDendrogramNode> nodes;
                std::vector<int64_t> roots;
                buildLeidenDendrogram(nodes, roots, communityMappingPerPass, degrees, num_nodes);
                PrintTime("Dendrogram Nodes", nodes.size());
                PrintTime("Dendrogram Roots", roots.size());
                orderDendrogramDFS(nodes, roots, new_ids, false, true);
                break;
            }
            case LeidenBFS: {
                std::cout << "Ordering Flavor: LeidenBFS (BFS by level)" << std::endl;
                std::vector<LeidenDendrogramNode> nodes;
                std::vector<int64_t> roots;
                buildLeidenDendrogram(nodes, roots, communityMappingPerPass, degrees, num_nodes);
                PrintTime("Dendrogram Nodes", nodes.size());
                PrintTime("Dendrogram Roots", roots.size());
                orderDendrogramBFS(nodes, roots, new_ids);
                break;
            }
            case LeidenHybrid: {
                std::cout << "Ordering Flavor: LeidenHybrid (community + degree)" << std::endl;
                orderLeidenHybridHubDFS(communityMappingPerPass, degrees, new_ids);
                break;
            }
            default:
                std::cerr << "Unknown LeidenDendrogram flavor: " << flavor << ", using LeidenDFSHub" << std::endl;
                std::vector<LeidenDendrogramNode> nodes;
                std::vector<int64_t> roots;
                buildLeidenDendrogram(nodes, roots, communityMappingPerPass, degrees, num_nodes);
                orderDendrogramDFS(nodes, roots, new_ids, true, false);
        }
        
        tm.Stop();
        PrintTime("Ordering Time", tm.Seconds());
    }

    /**
     * Community Features for Adaptive Reordering Selection
     * 
     * Based on empirical correlation analysis:
     * - Low density (sparse) communities: RCM works best
     * - Medium density + high degree variance: DBG works best  
     * - High density + uniform degrees: HubSort/HubCluster works best
     * - Very high density (tight clusters): Original or LeidenOrder
     */
    struct CommunityFeatures {
        size_t num_nodes;
        size_t num_edges;
        double internal_density;     // edges / possible_edges
        double avg_degree;
        double degree_variance;      // normalized variance in degrees
        double hub_concentration;    // fraction of edges from top 10% nodes
        double modularity;           // community/subgraph modularity (set externally)
        // Extended features (computed when available)
        double clustering_coeff = 0.0;    // local clustering coefficient (sampled)
        double avg_path_length = 0.0;     // estimated average path length
        double diameter_estimate = 0.0;   // estimated graph diameter  
        double community_count = 0.0;     // number of sub-communities
        double reorder_time = 0.0;        // estimated reorder time (if known)
    };

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
     * File path: ./scripts/perceptron_weights.json (or PERCEPTRON_WEIGHTS_FILE env var)
     * 
     * Format: {"AlgorithmName": {"bias": X, "w_modularity": X, ...}, ...}
     * 
     * The file is typically located at scripts/perceptron_weights.json and is
     * automatically generated by running: python3 scripts/graphbrew_experiment.py --fill-weights
     */
    
    // Default path for perceptron weights file (relative to project root)
    static constexpr const char* DEFAULT_WEIGHTS_FILE = "scripts/perceptron_weights.json";
    
    /**
     * Benchmark type enum for benchmark-specific weight selection
     * 
     * Use BENCH_GENERIC (default) when:
     * - Reordering for general-purpose graph processing
     * - Benchmark type is not known at reorder time
     * - You want balanced performance across all algorithms
     * 
     * Use specific types (BENCH_PR, BENCH_BFS, etc.) when:
     * - You know which benchmark will run on this graph
     * - You want to optimize reordering for that specific workload
     */
    enum BenchmarkType {
        BENCH_GENERIC = 0,  // Generic/default - no benchmark-specific adjustment (multiplier = 1.0)
        BENCH_PR,           // PageRank - iterative, benefits from cache locality
        BENCH_BFS,          // Breadth-First Search - traversal-heavy
        BENCH_CC,           // Connected Components - union-find based
        BENCH_SSSP,         // Single-Source Shortest Path - priority queue based
        BENCH_BC,           // Betweenness Centrality - all-pairs traversal
        BENCH_TC            // Triangle Counting - neighborhood intersection
    };
    
    /**
     * Convert benchmark name string to enum
     * Returns BENCH_GENERIC if name is empty, "generic", or unrecognized
     */
    static BenchmarkType GetBenchmarkType(const std::string& name) {
        if (name.empty() || name == "generic" || name == "GENERIC" || name == "all") return BENCH_GENERIC;
        if (name == "pr" || name == "PR" || name == "pagerank" || name == "PageRank") return BENCH_PR;
        if (name == "bfs" || name == "BFS") return BENCH_BFS;
        if (name == "cc" || name == "CC") return BENCH_CC;
        if (name == "sssp" || name == "SSSP") return BENCH_SSSP;
        if (name == "bc" || name == "BC") return BENCH_BC;
        if (name == "tc" || name == "TC") return BENCH_TC;
        return BENCH_GENERIC;  // Default to generic for any unrecognized name
    }
    
    struct PerceptronWeights {
        // Core weights (original)
        double bias;                // baseline score
        double w_modularity;        // correlation with modularity
        double w_log_nodes;         // scale effect
        double w_log_edges;         // size effect  
        double w_density;           // sparsity effect
        double w_avg_degree;        // connectivity effect
        double w_degree_variance;   // power-law / uniformity effect
        double w_hub_concentration; // hub dominance effect
        
        // Extended graph structure weights (NEW)
        double w_clustering_coeff = 0.0;   // local clustering coefficient effect
        double w_avg_path_length = 0.0;    // average path length sensitivity
        double w_diameter = 0.0;           // diameter effect
        double w_community_count = 0.0;    // sub-community count effect
        
        // Cache impact weights (NEW - from cache simulation)
        double cache_l1_impact = 0.0;      // bonus for high L1 hit rate
        double cache_l2_impact = 0.0;      // bonus for high L2 hit rate
        double cache_l3_impact = 0.0;      // bonus for high L3 hit rate
        double cache_dram_penalty = 0.0;   // penalty for DRAM accesses
        
        // Reorder time weight (NEW)
        double w_reorder_time = 0.0;       // penalty for slow reordering
        
        // Benchmark-specific weights (NEW - multipliers per benchmark type)
        double bench_pr = 1.0;      // PageRank weight multiplier
        double bench_bfs = 1.0;     // BFS weight multiplier
        double bench_cc = 1.0;      // CC weight multiplier
        double bench_sssp = 1.0;    // SSSP weight multiplier
        double bench_bc = 1.0;      // BC weight multiplier
        double bench_tc = 1.0;      // TC weight multiplier
        
        /**
         * Get benchmark-specific multiplier
         * Returns 1.0 for BENCH_GENERIC (no adjustment)
         */
        double getBenchmarkMultiplier(BenchmarkType bench) const {
            switch (bench) {
                case BENCH_PR:   return bench_pr;
                case BENCH_BFS:  return bench_bfs;
                case BENCH_CC:   return bench_cc;
                case BENCH_SSSP: return bench_sssp;
                case BENCH_BC:   return bench_bc;
                case BENCH_TC:   return bench_tc;
                case BENCH_GENERIC:
                default:         return 1.0;  // Generic - no benchmark-specific adjustment
            }
        }
        
        /**
         * Compute base score (without benchmark adjustment)
         * This is used for generic/default scoring across all algorithms
         */
        double scoreBase(const CommunityFeatures& feat) const {
            double log_nodes = std::log10(static_cast<double>(feat.num_nodes) + 1.0);
            double log_edges = std::log10(static_cast<double>(feat.num_edges) + 1.0);
            
            // Core score (original features)
            double s = bias 
                 + w_modularity * feat.modularity
                 + w_log_nodes * log_nodes
                 + w_log_edges * log_edges
                 + w_density * feat.internal_density
                 + w_avg_degree * feat.avg_degree / 100.0  // normalize
                 + w_degree_variance * feat.degree_variance
                 + w_hub_concentration * feat.hub_concentration;
            
            // Extended features (if available)
            s += w_clustering_coeff * feat.clustering_coeff;
            s += w_avg_path_length * feat.avg_path_length / 10.0;  // normalize
            s += w_diameter * feat.diameter_estimate / 50.0;        // normalize
            s += w_community_count * std::log10(feat.community_count + 1.0);
            
            // Reorder time penalty (if known)
            s += w_reorder_time * feat.reorder_time;
            
            return s;
        }
        
        /**
         * Compute score with optional benchmark-specific adjustment
         * 
         * @param feat Community features to evaluate
         * @param bench Benchmark type (default: BENCH_GENERIC for balanced performance)
         * 
         * For generic use (no specific benchmark), multiplier = 1.0 and score = base score.
         * For specific benchmarks, score = base score * benchmark_multiplier.
         */
        double score(const CommunityFeatures& feat, BenchmarkType bench = BENCH_GENERIC) const {
            double base = scoreBase(feat);
            double multiplier = getBenchmarkMultiplier(bench);
            return base * multiplier;
        }
        
        // Legacy overload for backward compatibility (uses generic/default scoring)
        double score(const CommunityFeatures& feat) const {
            return scoreBase(feat);
        }
    };

    /**
     * Learned Perceptron Weights for each reordering algorithm
     * 
     * These weights were trained from multi-algorithm benchmarks across:
     * - Graph algorithms: BFS, PageRank, SSSP, BC, CC
     * - Graph types: SBM (varying modularity), RMAT (power-law)
     * - Metrics: Speedup relative to Original ordering
     * 
     * Key insights from training:
     * - RabbitOrder: Strong negative correlation with modularity (better on low-mod graphs)
     * - HubClusterDBG: Positive correlation with degree variance and hub concentration
     * - RCMOrder: Best for sparse graphs (high w_density penalty on dense)
     * - GraphBrewOrder: Mixed - good baseline but overhead matters on small graphs
     * - AdaptiveOrder: Recursive application shows benefit on large communities
     */
    static const std::map<ReorderingAlgo, PerceptronWeights> GetPerceptronWeights() {
        return {
            // ORIGINAL: baseline, no reordering overhead
            {ORIGINAL, {
                .bias = 1.0,
                .w_modularity = 0.3,      // good on high-mod (no overhead)
                .w_log_nodes = -0.05,     // worse as graph grows
                .w_log_edges = -0.02,
                .w_density = 0.0,
                .w_avg_degree = 0.0,
                .w_degree_variance = -0.1,
                .w_hub_concentration = -0.1,
                // Extended weights (0.0 = neutral, override via JSON)
                .w_clustering_coeff = 0.0, .w_avg_path_length = 0.0, .w_diameter = 0.0, .w_community_count = 0.0,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = 0.0
            }},
            // HubSort: light reordering, puts hubs first
            {HubSort, {
                .bias = 0.85,
                .w_modularity = 0.0,
                .w_log_nodes = 0.02,
                .w_log_edges = 0.02,
                .w_density = -0.5,        // worse on dense
                .w_avg_degree = 0.02,
                .w_degree_variance = 0.15,
                .w_hub_concentration = 0.25,  // best when hubs dominate
                .w_clustering_coeff = 0.0, .w_avg_path_length = 0.0, .w_diameter = 0.0, .w_community_count = 0.0,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = -0.1     // fast reordering
            }},
            // HubCluster: groups hubs together
            {HubCluster, {
                .bias = 0.82,
                .w_modularity = 0.05,
                .w_log_nodes = 0.03,
                .w_log_edges = 0.03,
                .w_density = -0.3,
                .w_avg_degree = 0.03,
                .w_degree_variance = 0.2,
                .w_hub_concentration = 0.3,
                .w_clustering_coeff = 0.0, .w_avg_path_length = 0.0, .w_diameter = 0.0, .w_community_count = 0.0,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = -0.1
            }},
            // DBG: degree-based grouping
            {DBG, {
                .bias = 0.8,
                .w_modularity = -0.1,     // better on low-mod
                .w_log_nodes = 0.02,
                .w_log_edges = 0.02,
                .w_density = -0.4,
                .w_avg_degree = 0.0,
                .w_degree_variance = 0.25,    // best with varied degrees
                .w_hub_concentration = 0.1,
                .w_clustering_coeff = 0.0, .w_avg_path_length = 0.0, .w_diameter = 0.0, .w_community_count = 0.0,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = -0.05
            }},
            // HubClusterDBG: combination approach
            // Benchmark: avg_speedup=2.38x, good but outperformed by Leiden variants
            {HubClusterDBG, {
                .bias = 0.62,             // lowered - Leiden variants are better
                .w_modularity = 0.10,     // better on higher mod
                .w_log_nodes = 0.04,
                .w_log_edges = 0.04,
                .w_density = -0.20,
                .w_avg_degree = 0.02,
                .w_degree_variance = 0.25,
                .w_hub_concentration = 0.20,
                .w_clustering_coeff = 0.0, .w_avg_path_length = 0.0, .w_diameter = 0.0, .w_community_count = 0.0,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = -0.1
            }},
#ifdef RABBIT_ENABLE
            // RabbitOrder: recursive bisection
            // Benchmark: avg_speedup=4.06x, good on social networks
            {RabbitOrder, {
                .bias = 0.70,
                .w_modularity = -0.30,    // better on LOW modularity (synth graphs)
                .w_log_nodes = 0.05,      // scales well
                .w_log_edges = 0.05,
                .w_density = -0.30,       // worse on dense
                .w_avg_degree = 0.0,
                .w_degree_variance = 0.20,
                .w_hub_concentration = 0.15,
                .w_clustering_coeff = 0.05, .w_avg_path_length = -0.02, .w_diameter = 0.0, .w_community_count = 0.0,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = -0.3     // higher reorder cost
            }},
#endif
            // RCMOrder: reverse Cuthill-McKee
            {RCMOrder, {
                .bias = 0.7,
                .w_modularity = 0.0,
                .w_log_nodes = 0.03,
                .w_log_edges = 0.02,
                .w_density = -0.8,        // strong penalty on dense
                .w_avg_degree = -0.03,    // worse on high degree
                .w_degree_variance = 0.05,
                .w_hub_concentration = 0.0,
                .w_clustering_coeff = 0.0, .w_avg_path_length = 0.1, .w_diameter = 0.05, .w_community_count = 0.0,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = -0.2
            }},
            // LeidenOrder: community-based
            // Benchmark: avg_speedup=4.40x (highest trial speedup but high reorder cost)
            {LeidenOrder, {
                .bias = 0.76,             // win_trial=2, high trial speedup
                .w_modularity = 0.40,     // best on high modularity
                .w_log_nodes = 0.05,
                .w_log_edges = 0.05,
                .w_density = -0.20,
                .w_avg_degree = 0.01,
                .w_degree_variance = 0.10,
                .w_hub_concentration = 0.10,
                .w_clustering_coeff = 0.1, .w_avg_path_length = 0.0, .w_diameter = 0.0, .w_community_count = 0.05,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = -0.4     // high reorder cost
            }},
            // GraphBrewOrder: Leiden + per-community RabbitOrder
            {GraphBrewOrder, {
                .bias = 0.6,
                .w_modularity = -0.25,    // better on low-mod (community reorder helps)
                .w_log_nodes = 0.1,       // scales well
                .w_log_edges = 0.08,
                .w_density = -0.3,
                .w_avg_degree = 0.0,
                .w_degree_variance = 0.2,
                .w_hub_concentration = 0.1,
                .w_clustering_coeff = 0.05, .w_avg_path_length = -0.02, .w_diameter = 0.0, .w_community_count = 0.1,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = -0.5     // highest reorder cost
            }},
            // =================================================================
            // LEIDEN DENDROGRAM VARIANTS (RabbitOrder-style tree traversal)
            // Weights tuned from benchmark analysis (Jan 2026):
            // - LeidenHybrid: BEST overall (5/5 wins, 4.26x speedup, 0.0038s reorder)
            // - LeidenDFS: Best total time on RMAT synthetic
            // - LeidenOrder: Best trial time on synthetic (but high reorder cost)
            // =================================================================
            // LeidenDFS: Standard DFS traversal of community dendrogram
            {LeidenDFS, {
                .bias = 0.73,             // win_total=1, good for synthetic
                .w_modularity = 0.35,     // good on high modularity
                .w_log_nodes = 0.05,      // better on larger graphs
                .w_log_edges = 0.04,
                .w_density = -0.40,       // better on sparse
                .w_avg_degree = 0.0,
                .w_degree_variance = 0.10,
                .w_hub_concentration = 0.10,
                .w_clustering_coeff = 0.08, .w_avg_path_length = 0.0, .w_diameter = 0.0, .w_community_count = 0.05,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = -0.35
            }},
            // LeidenDFSHub: DFS with high-degree nodes first within communities
            // Good for hub-dominated graphs but LeidenHybrid is generally better
            {LeidenDFSHub, {
                .bias = 0.65,             // lowered - LeidenHybrid is usually better
                .w_modularity = 0.28,
                .w_log_nodes = 0.05,
                .w_log_edges = 0.05,
                .w_density = -0.35,
                .w_avg_degree = 0.02,
                .w_degree_variance = 0.20,
                .w_hub_concentration = 0.25,
                .w_clustering_coeff = 0.08, .w_avg_path_length = 0.0, .w_diameter = 0.0, .w_community_count = 0.05,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = -0.35
            }},
            // LeidenDFSSize: DFS with largest subtrees first
            {LeidenDFSSize, {
                .bias = 0.72,             // win_total=1
                .w_modularity = 0.32,
                .w_log_nodes = 0.05,      // better on larger graphs
                .w_log_edges = 0.05,
                .w_density = -0.35,
                .w_avg_degree = 0.01,
                .w_degree_variance = 0.15,
                .w_hub_concentration = 0.15,
                .w_clustering_coeff = 0.08, .w_avg_path_length = 0.0, .w_diameter = 0.0, .w_community_count = 0.05,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = -0.35
            }},
            // LeidenBFS: BFS by hierarchy level
            {LeidenBFS, {
                .bias = 0.70,             // moderate - path-aware locality
                .w_modularity = 0.28,
                .w_log_nodes = 0.05,
                .w_log_edges = 0.05,
                .w_density = -0.30,
                .w_avg_degree = 0.01,
                .w_degree_variance = 0.08,
                .w_hub_concentration = 0.05,
                .w_clustering_coeff = 0.05, .w_avg_path_length = 0.05, .w_diameter = 0.02, .w_community_count = 0.05,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = -0.35
            }},
            // LeidenHybrid: Sort by (community, degree descending) 
            // BENCHMARK WINNER: 5/5 wins on trial_time AND total_time
            // Amazon: 0.0073s vs HubClusterDBG 0.0507s (7x faster!)
            {LeidenHybrid, {
                .bias = 0.95,             // HIGHEST - clear benchmark winner
                .w_modularity = 0.45,     // best on high modularity graphs
                .w_log_nodes = 0.06,      // scales well
                .w_log_edges = 0.06,
                .w_density = -0.15,       // tolerates density better
                .w_avg_degree = 0.02,
                .w_degree_variance = 0.15,
                .w_hub_concentration = 0.25,  // also good with hubs
                .w_clustering_coeff = 0.1, .w_avg_path_length = 0.0, .w_diameter = 0.0, .w_community_count = 0.08,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = -0.15    // fast reordering (hybrid is efficient)
            }},
        };
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
     */
    static bool ParseWeightsFromJSON(const std::string& json_content, 
                                      std::map<ReorderingAlgo, PerceptronWeights>& weights) {
        // Simple JSON parser - looks for algorithm names and their weights
        auto find_double = [](const std::string& s, const std::string& key) -> double {
            size_t pos = s.find("\"" + key + "\"");
            if (pos == std::string::npos) return 0.0;
            pos = s.find(':', pos);
            if (pos == std::string::npos) return 0.0;
            pos++;
            while (pos < s.size() && (s[pos] == ' ' || s[pos] == '\t')) pos++;
            size_t end = pos;
            while (end < s.size() && (isdigit(s[end]) || s[end] == '.' || s[end] == '-' || s[end] == 'e' || s[end] == 'E' || s[end] == '+')) end++;
            try {
                return std::stod(s.substr(pos, end - pos));
            } catch (...) {
                return 0.0;
            }
        };
        
        // Map string names to enum (support both naming conventions)
        static const std::map<std::string, ReorderingAlgo> name_to_algo = {
            {"ORIGINAL", ORIGINAL}, {"HubSort", HubSort}, {"HUBSORT", HubSort},
            {"HubCluster", HubCluster}, {"HUBCLUSTER", HubCluster},
            {"DBG", DBG}, {"HubClusterDBG", HubClusterDBG}, {"HUBCLUSTERDBG", HubClusterDBG},
            {"HubSortDBG", HubSortDBG}, {"HUBSORTDBG", HubSortDBG},
            {"SORT", Sort},
            {"RANDOM", Random},
#ifdef RABBIT_ENABLE
            {"RabbitOrder", RabbitOrder}, {"RABBITORDER", RabbitOrder},
#endif
            {"GOrder", GOrder}, {"GORDER", GOrder},
            {"COrder", COrder}, {"CORDER", COrder},
            {"RCMOrder", RCMOrder}, {"RCM", RCMOrder},
            {"LeidenOrder", LeidenOrder},
            {"GraphBrewOrder", GraphBrewOrder},
            {"LeidenDFS", LeidenDFS},
            {"LeidenDFSHub", LeidenDFSHub},
            {"LeidenDFSSize", LeidenDFSSize},
            {"LeidenBFS", LeidenBFS},
            {"LeidenHybrid", LeidenHybrid},
            {"AdaptiveOrder", AdaptiveOrder}
        };
        
        for (const auto& kv : name_to_algo) {
            size_t pos = json_content.find("\"" + kv.first + "\"");
            if (pos == std::string::npos) continue;
            
            // Find the block for this algorithm
            size_t start = json_content.find('{', pos);
            if (start == std::string::npos) continue;
            size_t end = json_content.find('}', start);
            if (end == std::string::npos) continue;
            
            std::string block = json_content.substr(start, end - start + 1);
            
            PerceptronWeights w;
            // Core weights
            w.bias = find_double(block, "bias");
            w.w_modularity = find_double(block, "w_modularity");
            w.w_log_nodes = find_double(block, "w_log_nodes");
            w.w_log_edges = find_double(block, "w_log_edges");
            w.w_density = find_double(block, "w_density");
            w.w_avg_degree = find_double(block, "w_avg_degree");
            w.w_degree_variance = find_double(block, "w_degree_variance");
            w.w_hub_concentration = find_double(block, "w_hub_concentration");
            
            // Extended graph structure weights
            w.w_clustering_coeff = find_double(block, "w_clustering_coeff");
            w.w_avg_path_length = find_double(block, "w_avg_path_length");
            w.w_diameter = find_double(block, "w_diameter");
            w.w_community_count = find_double(block, "w_community_count");
            
            // Cache impact weights
            w.cache_l1_impact = find_double(block, "cache_l1_impact");
            w.cache_l2_impact = find_double(block, "cache_l2_impact");
            w.cache_l3_impact = find_double(block, "cache_l3_impact");
            w.cache_dram_penalty = find_double(block, "cache_dram_penalty");
            
            // Reorder time weight
            w.w_reorder_time = find_double(block, "w_reorder_time");
            
            // Benchmark-specific weights (parse nested benchmark_weights object)
            // Find the benchmark_weights block within the algorithm block
            size_t bw_pos = block.find("\"benchmark_weights\"");
            if (bw_pos != std::string::npos) {
                size_t bw_start = block.find('{', bw_pos);
                size_t bw_end = block.find('}', bw_start);
                if (bw_start != std::string::npos && bw_end != std::string::npos) {
                    std::string bw_block = block.substr(bw_start, bw_end - bw_start + 1);
                    // Parse individual benchmark weights (default to 1.0 if not found)
                    double pr_w = find_double(bw_block, "pr");
                    double bfs_w = find_double(bw_block, "bfs");
                    double cc_w = find_double(bw_block, "cc");
                    double sssp_w = find_double(bw_block, "sssp");
                    double bc_w = find_double(bw_block, "bc");
                    double tc_w = find_double(bw_block, "tc");
                    // Only set if non-zero (JSON default is 1.0, but find_double returns 0.0 if missing)
                    w.bench_pr = (pr_w != 0.0) ? pr_w : 1.0;
                    w.bench_bfs = (bfs_w != 0.0) ? bfs_w : 1.0;
                    w.bench_cc = (cc_w != 0.0) ? cc_w : 1.0;
                    w.bench_sssp = (sssp_w != 0.0) ? sssp_w : 1.0;
                    w.bench_bc = (bc_w != 0.0) ? bc_w : 1.0;
                    w.bench_tc = (tc_w != 0.0) ? tc_w : 1.0;
                }
            }
            
            weights[kv.second] = w;
        }
        
        return !weights.empty();
    }

    /**
     * Load perceptron weights from file, or return defaults if file doesn't exist.
     * 
     * Checks for weights file in this order:
     * 1. Path from PERCEPTRON_WEIGHTS_FILE environment variable
     * 2. ./perceptron_weights.json (relative to working directory)
     * 3. If neither exists, returns hardcoded defaults from GetPerceptronWeights()
     */
    static std::map<ReorderingAlgo, PerceptronWeights> LoadPerceptronWeights(bool verbose = false) {
        // Start with defaults
        auto weights = GetPerceptronWeights();
        
        // Check for weights file
        std::string weights_file;
        const char* env_path = std::getenv("PERCEPTRON_WEIGHTS_FILE");
        if (env_path != nullptr) {
            weights_file = env_path;
        } else {
            weights_file = DEFAULT_WEIGHTS_FILE;
        }
        
        std::ifstream file(weights_file);
        if (!file.is_open()) {
            if (verbose) {
                std::cout << "Perceptron: Using hardcoded default weights (no " 
                          << weights_file << " found)\n";
            }
            return weights;
        }
        
        // Read file content
        std::string json_content((std::istreambuf_iterator<char>(file)),
                                  std::istreambuf_iterator<char>());
        file.close();
        
        // Parse and merge with defaults
        std::map<ReorderingAlgo, PerceptronWeights> loaded_weights;
        if (ParseWeightsFromJSON(json_content, loaded_weights)) {
            for (const auto& kv : loaded_weights) {
                weights[kv.first] = kv.second;
            }
            if (verbose) {
                std::cout << "Perceptron: Loaded " << loaded_weights.size() 
                          << " algorithm weights from " << weights_file << "\n";
            }
        } else if (verbose) {
            std::cout << "Perceptron: Failed to parse " << weights_file 
                      << ", using defaults\n";
        }
        
        return weights;
    }

    /**
     * Select best reordering algorithm using perceptron scores
     * 
     * Evaluates all candidate algorithms and returns the one with
     * the highest perceptron score based on community features.
     * 
     * @param feat Community features for scoring
     * @param bench Benchmark type (default: BENCH_GENERIC for balanced performance)
     *              - BENCH_GENERIC: Optimizes for all graph algorithms equally
     *              - BENCH_PR, BENCH_BFS, etc.: Optimizes for specific benchmark
     * 
     * Loads weights from file if available, otherwise uses defaults.
     */
    ReorderingAlgo SelectReorderingPerceptron(const CommunityFeatures& feat, 
                                               BenchmarkType bench = BENCH_GENERIC) {
        // Use cached weights - load once from file or defaults
        static const auto weights = LoadPerceptronWeights(false);
        
        ReorderingAlgo best_algo = ORIGINAL;
        double best_score = -std::numeric_limits<double>::infinity();
        
        for (const auto& kv : weights) {
            double score = kv.second.score(feat, bench);
            if (score > best_score) {
                best_score = score;
                best_algo = kv.first;
            }
        }
        
        return best_algo;
    }
    
    /**
     * Overload that accepts benchmark name as string
     * Pass empty string or "generic" for balanced/default scoring
     */
    ReorderingAlgo SelectReorderingPerceptron(const CommunityFeatures& feat,
                                               const std::string& benchmark_name) {
        return SelectReorderingPerceptron(feat, GetBenchmarkType(benchmark_name));
    }

    CommunityFeatures ComputeCommunityFeatures(
        const std::vector<NodeID_>& comm_nodes,
        const CSRGraph<NodeID_, DestID_, invert>& g,
        const std::unordered_set<NodeID_>& node_set)
    {
        CommunityFeatures feat;
        feat.num_nodes = comm_nodes.size();
        feat.modularity = 0.0;  // Set externally if needed
        
        if (feat.num_nodes < 2) {
            feat.num_edges = 0;
            feat.internal_density = 0.0;
            feat.avg_degree = 0.0;
            feat.degree_variance = 0.0;
            feat.hub_concentration = 0.0;
            return feat;
        }

        // Count internal edges and compute degrees
        std::vector<size_t> internal_degrees(feat.num_nodes, 0);
        size_t total_internal_edges = 0;
        
        #pragma omp parallel for reduction(+:total_internal_edges)
        for (size_t i = 0; i < feat.num_nodes; ++i) {
            NodeID_ node = comm_nodes[i];
            size_t local_internal = 0;
            for (DestID_ neighbor : g.out_neigh(node)) {
                NodeID_ dest = static_cast<NodeID_>(neighbor);
                if (node_set.count(dest)) {
                    ++local_internal;
                }
            }
            internal_degrees[i] = local_internal;
            total_internal_edges += local_internal;
        }
        
        feat.num_edges = total_internal_edges / 2; // undirected
        
        // Internal density: actual edges / possible edges
        size_t possible_edges = feat.num_nodes * (feat.num_nodes - 1) / 2;
        feat.internal_density = (possible_edges > 0) ? 
            static_cast<double>(feat.num_edges) / possible_edges : 0.0;
        
        // Average degree
        feat.avg_degree = (feat.num_nodes > 0) ? 
            static_cast<double>(total_internal_edges) / feat.num_nodes : 0.0;
        
        // Degree variance (normalized)
        double sum_sq_diff = 0.0;
        #pragma omp parallel for reduction(+:sum_sq_diff)
        for (size_t i = 0; i < feat.num_nodes; ++i) {
            double diff = internal_degrees[i] - feat.avg_degree;
            sum_sq_diff += diff * diff;
        }
        double variance = (feat.num_nodes > 1) ? sum_sq_diff / (feat.num_nodes - 1) : 0.0;
        feat.degree_variance = (feat.avg_degree > 0) ? 
            std::sqrt(variance) / feat.avg_degree : 0.0; // coefficient of variation
        
        // Hub concentration: fraction of edges from top 10% degree nodes
        std::vector<size_t> sorted_degrees = internal_degrees;
        std::sort(sorted_degrees.rbegin(), sorted_degrees.rend());
        size_t top_10_percent = std::max(size_t(1), feat.num_nodes / 10);
        size_t top_edges = 0;
        for (size_t i = 0; i < top_10_percent; ++i) {
            top_edges += sorted_degrees[i];
        }
        feat.hub_concentration = (total_internal_edges > 0) ? 
            static_cast<double>(top_edges) / total_internal_edges : 0.0;
        
        return feat;
    }

    /**
     * Select best reordering algorithm based on community features.
     * 
     * Uses a PERCEPTRON-STYLE neural network approach:
     * 1. Each candidate algorithm has learned weights for each feature
     * 2. Compute score = weighted sum of features for each algorithm
     * 3. Select algorithm with highest score
     * 
     * This replaces the hand-tuned heuristics with data-driven selection.
     * Weights were trained from multi-algorithm benchmarks across:
     * - Graph algorithms: BFS, PageRank, SSSP, BC, CC
     * - Graph types: SBM (varying modularity), RMAT (power-law)
     * 
     * @param feat Community features
     * @param global_modularity Graph modularity score
     * @param bench Benchmark type (default: BENCH_GENERIC for balanced performance)
     *              - BENCH_GENERIC: Optimizes for all graph algorithms equally
     *              - BENCH_PR, BENCH_BFS, etc.: Optimizes for specific benchmark
     * 
     * Fallback to heuristics for edge cases (very small communities).
     */
    ReorderingAlgo SelectBestReorderingForCommunity(CommunityFeatures feat, double global_modularity,
                                                     BenchmarkType bench = BENCH_GENERIC)
    {
        // Small communities: reordering overhead exceeds benefit
        const size_t MIN_COMMUNITY_SIZE = 200;
        
        if (feat.num_nodes < MIN_COMMUNITY_SIZE) {
            return ORIGINAL;
        }
        
        // Set the modularity in features for perceptron scoring
        feat.modularity = global_modularity;
        
        // Use perceptron to select best algorithm (with benchmark-specific adjustment)
        ReorderingAlgo selected = SelectReorderingPerceptron(feat, bench);
        
        // Safety check: if perceptron selects an unavailable algorithm, fallback
#ifndef RABBIT_ENABLE
        if (selected == RabbitOrder) {
            // Recompute without RabbitOrder by using heuristic fallback
            if (feat.degree_variance > 0.8) {
                selected = HubClusterDBG;
            } else if (feat.hub_concentration > 0.3) {
                selected = HubSort;
            } else {
                selected = DBG;
            }
        }
#endif
        
        return selected;
    }

    /**
     * Generate Adaptive Reordering - Recursive Per-Community Selection
     * 
     * This implementation:
     * 1. Runs Leiden community detection to find communities
     * 2. For each community, computes features (density, degree variance, hub concentration)
     * 3. Selects the best reordering algorithm for that community
     * 4. For large communities with sub-structure, recursively applies AdaptiveOrder
     * 
     * Unlike GraphBrew which always uses RabbitOrder for communities,
     * this adaptively picks the best algorithm based on each community's characteristics.
     */
    void GenerateAdaptiveMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                 pvector<NodeID_> &new_ids, bool useOutdeg,
                                 std::vector<std::string> reordering_options)
    {
        GenerateAdaptiveMappingRecursive(g, new_ids, useOutdeg, reordering_options, 0, true);
    }

    /**
     * Recursive Adaptive Reordering for Subgraphs
     *  
     * For each community detected by Leiden:
     * 1. Compute local features (density, degree variance, hub concentration)
     * 2. Select the best reordering algorithm based on features
     * 3. For large communities with good structure, recursively apply AdaptiveOrder
     * 4. For smaller communities, apply the selected local reordering
     * 
     * This is the key insight: instead of always using RabbitOrder like GraphBrew,
     * we adaptively pick the best algorithm for each community's characteristics.
     */
    void GenerateAdaptiveMappingRecursive(
        const CSRGraph<NodeID_, DestID_, invert> &g,
        pvector<NodeID_> &new_ids, 
        bool useOutdeg,
        std::vector<std::string> reordering_options,
        int depth = 0,
        bool verbose = true)
    {
        Timer tm;
        tm.Start();

        using V = TYPE;
        install_sigsegv();

        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges_directed();
        
        const int MAX_DEPTH = 3;  // Limit recursion depth
        const size_t MIN_COMMUNITY_FOR_RECURSION = 10000;  // Don't recurse into tiny communities
        
        // Parse options
        double resolution = 0.75;
        int maxIterations = 30;
        int maxPasses = 30;

        if (reordering_options.size() > 0) {
            resolution = std::stod(reordering_options[0]);
        }
        if (reordering_options.size() > 1) {
            maxIterations = std::stoi(reordering_options[1]);
        }
        if (reordering_options.size() > 2) {
            maxPasses = std::stoi(reordering_options[2]);
        }

        if (depth == 0 && verbose) {
            PrintTime("Adaptive Resolution", resolution);
        }
        
        // Step 1: Run Leiden community detection
        vector<vector<K>> communityMappingPerPass;
        std::vector<size_t> comm_ids(num_nodes, 0);
        double global_modularity = 0.0;
        
        {
            std::vector<std::tuple<size_t, size_t, double>> edges(num_edges);
            const bool is_weighted = g.is_weighted();
            
            #pragma omp parallel for schedule(dynamic, 1024)
            for (NodeID_ i = 0; i < num_nodes; ++i) {
                NodeID_ out_start = g.out_offset(i);
                NodeID_ j = 0;
                for (DestID_ neighbor : g.out_neigh(i)) {
                    auto& edge = edges[out_start + j];
                    if (is_weighted) {
                        std::get<0>(edge) = i;
                        std::get<1>(edge) = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                        std::get<2>(edge) = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;
                    } else {
                        std::get<0>(edge) = i;
                        std::get<1>(edge) = neighbor;
                        std::get<2>(edge) = 1.0;
                    }
                    ++j;
                }
            }

            DiGraph<K, None, V> x;
            readVecOmpW(x, edges, num_nodes, false, is_weighted);
            edges.clear();
            x = symmetricizeOmp(x);

            // Run Leiden
            std::random_device dev;
            std::default_random_engine rnd(dev());
            double M = edgeWeightOmp(x) / 2;
            
            auto result = leidenStaticOmp<false, false>(
                rnd, x, {1, resolution, 1e-12, 0.8, 1.0, maxIterations, maxPasses});
            
            global_modularity = getModularity(x, result, M);
            x.passAndClear(communityMappingPerPass);
        }

        // Get final community assignments
        if (!communityMappingPerPass.empty()) {
            const auto& final_pass = communityMappingPerPass.back();
            #pragma omp parallel for
            for (size_t i = 0; i < static_cast<size_t>(num_nodes); ++i) {
                comm_ids[i] = (i < final_pass.size()) ? final_pass[i] : 0;
            }
        }

        size_t num_comm = *std::max_element(comm_ids.begin(), comm_ids.end()) + 1;
        
        if (depth == 0 && verbose) {
            PrintTime("Modularity", global_modularity);
            PrintTime("Num Communities", static_cast<double>(num_comm));
        }

        // Step 2: Group nodes by community
        std::vector<std::vector<NodeID_>> community_nodes(num_comm);
        for (NodeID_ i = 0; i < num_nodes; ++i) {
            community_nodes[comm_ids[i]].push_back(i);
        }

        // Step 3: Compute features and select algorithm for each community
        std::vector<ReorderingAlgo> selected_algos(num_comm);
        std::vector<CommunityFeatures> all_features(num_comm);
        std::vector<bool> should_recurse(num_comm, false);
        
        if (depth == 0 && verbose) {
            std::cout << "\n=== Adaptive Reordering Selection (Depth " << depth 
                      << ", Modularity: " << std::fixed << std::setprecision(4) 
                      << global_modularity << ") ===\n";
            std::cout << "Comm\tNodes\tEdges\tDensity\tDegVar\tHubConc\tSelected\n";
        }
        
        for (size_t c = 0; c < num_comm; ++c) {
            if (community_nodes[c].empty()) {
                selected_algos[c] = ORIGINAL;
                continue;
            }
            
            size_t comm_size = community_nodes[c].size();
            
            // Create node set for fast lookup
            std::unordered_set<NodeID_> node_set(
                community_nodes[c].begin(), community_nodes[c].end());
            
            // Compute features
            CommunityFeatures feat = ComputeCommunityFeatures(
                community_nodes[c], g, node_set);
            all_features[c] = feat;
            
            // Decide: recurse or apply local algorithm
            // Recurse if: large community, not too deep, and has structure worth exploiting
            if (comm_size >= MIN_COMMUNITY_FOR_RECURSION && 
                depth < MAX_DEPTH &&
                feat.internal_density < 0.1) {  // Sparse enough to have sub-communities
                
                should_recurse[c] = true;
                selected_algos[c] = AdaptiveOrder;  // Mark for recursion
            }
            else {
                // Select local algorithm based on features
                selected_algos[c] = SelectBestReorderingForCommunity(feat, global_modularity);
            }
            
            // Print selection rationale
            if (depth == 0 && verbose && feat.num_nodes >= 100) {
                std::cout << c << "\t" 
                          << feat.num_nodes << "\t"
                          << feat.num_edges << "\t"
                          << std::fixed << std::setprecision(4) << feat.internal_density << "\t"
                          << feat.degree_variance << "\t"
                          << feat.hub_concentration << "\t"
                          << (should_recurse[c] ? "RECURSE" : ReorderingAlgoStr(selected_algos[c])) << "\n";
            }
        }

        // Step 4: Sort communities by size (largest first for cache efficiency)
        std::vector<size_t> sorted_comms(num_comm);
        std::iota(sorted_comms.begin(), sorted_comms.end(), 0);
        std::sort(sorted_comms.begin(), sorted_comms.end(),
            [&community_nodes](size_t a, size_t b) {
                return community_nodes[a].size() > community_nodes[b].size();
            });

        // Step 5: Apply per-community reordering and assign global IDs
        NodeID_ current_id = 0;
        
        for (size_t c : sorted_comms) {
            auto& nodes = community_nodes[c];
            if (nodes.empty()) continue;
            
            size_t comm_size = nodes.size();
            ReorderingAlgo algo = selected_algos[c];
            
            if (comm_size < 10 || algo == ORIGINAL) {
                // Small or ORIGINAL: just assign sequentially
                for (NodeID_ node : nodes) {
                    new_ids[node] = current_id++;
                }
            }
            else if (should_recurse[c]) {
                // RECURSIVE CASE: Build subgraph and call AdaptiveOrder recursively
                std::unordered_map<NodeID_, NodeID_> global_to_local;
                std::vector<NodeID_> local_to_global(comm_size);
                for (size_t i = 0; i < comm_size; ++i) {
                    global_to_local[nodes[i]] = static_cast<NodeID_>(i);
                    local_to_global[i] = nodes[i];
                }
                
                // Create edge list for subgraph
                EdgeList sub_edges;
                std::unordered_set<NodeID_> node_set(nodes.begin(), nodes.end());
                
                for (NodeID_ node : nodes) {
                    NodeID_ local_src = global_to_local[node];
                    for (DestID_ neighbor : g.out_neigh(node)) {
                        NodeID_ dest = static_cast<NodeID_>(neighbor);
                        if (node_set.count(dest)) {
                            NodeID_ local_dst = global_to_local[dest];
                            sub_edges.push_back(Edge(local_src, local_dst));
                        }
                    }
                }
                
                // Build local graph
                CSRGraph<NodeID_, DestID_, invert> sub_g = MakeLocalGraphFromEL(sub_edges);
                pvector<NodeID_> sub_new_ids(comm_size, -1);
                
                // RECURSIVE CALL with increased depth
                GenerateAdaptiveMappingRecursive(sub_g, sub_new_ids, useOutdeg, 
                                                  reordering_options, depth + 1, false);
                
                // Map local reordered IDs back to global IDs
                std::vector<NodeID_> reordered_nodes(comm_size);
                for (size_t i = 0; i < comm_size; ++i) {
                    if (sub_new_ids[i] >= 0 && sub_new_ids[i] < static_cast<NodeID_>(comm_size)) {
                        reordered_nodes[sub_new_ids[i]] = local_to_global[i];
                    } else {
                        reordered_nodes[i] = local_to_global[i];
                    }
                }
                
                // Assign global IDs
                for (NodeID_ node : reordered_nodes) {
                    new_ids[node] = current_id++;
                }
            }
            else {
                // LOCAL REORDERING CASE
                std::unordered_map<NodeID_, NodeID_> global_to_local;
                std::vector<NodeID_> local_to_global(comm_size);
                for (size_t i = 0; i < comm_size; ++i) {
                    global_to_local[nodes[i]] = static_cast<NodeID_>(i);
                    local_to_global[i] = nodes[i];
                }
                
                // Create edge list for subgraph
                EdgeList sub_edges;
                std::unordered_set<NodeID_> node_set(nodes.begin(), nodes.end());
                
                for (NodeID_ node : nodes) {
                    NodeID_ local_src = global_to_local[node];
                    for (DestID_ neighbor : g.out_neigh(node)) {
                        NodeID_ dest = static_cast<NodeID_>(neighbor);
                        if (node_set.count(dest)) {
                            NodeID_ local_dst = global_to_local[dest];
                            sub_edges.push_back(Edge(local_src, local_dst));
                        }
                    }
                }
                
                // Build local graph
                CSRGraph<NodeID_, DestID_, invert> sub_g = MakeLocalGraphFromEL(sub_edges);
                pvector<NodeID_> sub_new_ids(comm_size, -1);
                
                // Apply the selected local algorithm
                switch (algo) {
                    case HubSort:
                        GenerateHubSortMapping(sub_g, sub_new_ids, useOutdeg);
                        break;
                    case HubCluster:
                        GenerateHubClusterMapping(sub_g, sub_new_ids, useOutdeg);
                        break;
                    case DBG:
                        GenerateDBGMapping(sub_g, sub_new_ids, useOutdeg);
                        break;
                    case HubClusterDBG:
                        GenerateHubClusterDBGMapping(sub_g, sub_new_ids, useOutdeg);
                        break;
                    case RCMOrder:
                        GenerateRCMOrderMapping(sub_g, sub_new_ids);
                        break;
#ifdef RABBIT_ENABLE
                    case RabbitOrder:
                        GenerateRabbitOrderMapping(sub_g, sub_new_ids);
                        break;
#endif
                    default:
                        GenerateOriginalMapping(sub_g, sub_new_ids);
                        break;
                }
                
                // Map local reordered IDs back to global IDs
                std::vector<NodeID_> reordered_nodes(comm_size);
                for (size_t i = 0; i < comm_size; ++i) {
                    if (sub_new_ids[i] >= 0 && sub_new_ids[i] < static_cast<NodeID_>(comm_size)) {
                        reordered_nodes[sub_new_ids[i]] = local_to_global[i];
                    } else {
                        reordered_nodes[i] = local_to_global[i];
                    }
                }
                
                // Assign global IDs
                for (NodeID_ node : reordered_nodes) {
                    new_ids[node] = current_id++;
                }
            }
        }
        
        tm.Stop();
        
        if (depth == 0 && verbose) {
            PrintTime("Adaptive Map Time", tm.Seconds());
            
            // Summary statistics
            std::map<ReorderingAlgo, int> algo_counts;
            int recurse_count = 0;
            for (size_t c = 0; c < num_comm; ++c) {
                if (!community_nodes[c].empty()) {
                    if (should_recurse[c]) {
                        recurse_count++;
                    } else {
                        algo_counts[selected_algos[c]]++;
                    }
                }
            }
            
            std::cout << "\n=== Algorithm Selection Summary ===\n";
            if (recurse_count > 0) {
                std::cout << "Recursive: " << recurse_count << " communities\n";
            }
            for (auto& [algo, count] : algo_counts) {
                std::cout << ReorderingAlgoStr(algo) << ": " << count << " communities\n";
            }
        }
    }

    void GenerateGraphBrewMapping(const CSRGraph<NodeID_, DestID_, invert> &g,
                                  pvector<NodeID_> &new_ids, bool useOutdeg,
                                  std::vector<std::string> reordering_options, int numLevels = 1,
                                  bool recursion = false)
    {

        Timer tm;
        Timer tm_2;

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
        double resolution = 0.75;
        // double total_time = 0.0;
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
            resolution = (resolution > 3) ? 1.0 : resolution;
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
            // Parallel loop to construct the edge list - write directly without temp objects
            const bool is_weighted = g.is_weighted();
            #pragma omp parallel for schedule(dynamic, 1024)
            for (NodeID_ i = 0; i < num_nodes; ++i)
            {
                NodeID_ out_start = g.out_offset(i);
                NodeID_ j = 0;
                for (DestID_ neighbor : g.out_neigh(i))
                {
                    auto& edge = edges[out_start + j];
                    if (is_weighted)
                    {
                        std::get<0>(edge) = i;
                        std::get<1>(edge) = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                        std::get<2>(edge) = static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;
                    }
                    else
                    {
                        std::get<0>(edge) = i;
                        std::get<1>(edge) = neighbor;
                        std::get<2>(edge) = 1.0;
                    }
                    ++j;
                }
            }

            tm.Start();
            bool symmetric = false;
            bool weighted = is_weighted;
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
            communityDataFlat[stride + i] = g.out_degree(i); // column 1: degree
        }

        // Copy community mappings per pass
        for (size_t p = 0; p < num_passes - 2; ++p)
        {
            K* dest_col = &communityDataFlat[(2 + p) * stride];
            const auto& src = communityMappingPerPass[p];
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
         * DENDROGRAM-BASED ORDERING for GraphBrew
         * Same optimization as LeidenOrder - use multi-level hierarchical sort
         */
        const size_t actual_passes = num_passes - 2;
        const size_t last_pass_col = (2 + actual_passes - 1);  // Last (coarsest) pass column
        
        // Sort primarily by last-pass community (coarsest level) 
        // with degree as secondary key within communities
        __gnu_parallel::sort(sort_indices.begin(), sort_indices.end(),
            [&communityDataFlat, stride, last_pass_col](size_t a, size_t b) {
                // Primary: sort by coarsest community  
                K comm_a = communityDataFlat[last_pass_col * stride + a];
                K comm_b = communityDataFlat[last_pass_col * stride + b];
                if (comm_a != comm_b) {
                    return comm_a < comm_b;
                }
                // Secondary: within community, sort by degree (descending)
                K deg_a = communityDataFlat[stride + a];
                K deg_b = communityDataFlat[stride + b];
                return deg_a > deg_b;
            });
        
        // Get the last pass community assignment for further processing
        const K* sort_key_col = &communityDataFlat[(num_passes - 1) * stride];

        // Extract comm_ids in sorted order
        #pragma omp parallel for
        for (size_t i = 0; i < num_nodesx; ++i)
        {
            comm_ids[i] = sort_key_col[sort_indices[i]];
        }

        // std::cout << std::endl;
        // for (size_t i = 0; i < num_nodesx; ++i)
        // {
        //     std::cout <<  i << "->" << comm_ids[i] << std::endl;
        // }
        // std::cout << std::endl;

        communityMappingPerPass.clear();
        communityDataFlat.clear();
        communityDataFlat.shrink_to_fit();

        num_comm = *(__gnu_parallel::max_element(comm_ids.begin(), comm_ids.end()));

        tm.Stop();
        // Create the frequency arrays
        std::vector<size_t> frequency_array_pre(num_comm + 1, 0);
        std::vector<size_t> frequency_array(num_comm + 1,
                                            0); // +1 to include num_comm index

        // Fill the frequency array using thread-local histograms (avoids atomic contention)
        const int num_threads_freq = omp_get_max_threads();
        std::vector<std::vector<size_t>> thread_freq_arrays(num_threads_freq, 
            std::vector<size_t>(num_comm + 1, 0));
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local_freq = thread_freq_arrays[tid];
            
            #pragma omp for nowait
            for (size_t i = 0; i < comm_ids.size(); ++i)
            {
                ++local_freq[comm_ids[i]];
            }
        }
        
        // Merge thread-local histograms in parallel
        #pragma omp parallel for schedule(static)
        for (size_t c = 0; c <= num_comm; ++c)
        {
            for (int t = 0; t < num_threads_freq; ++t)
            {
                frequency_array_pre[c] += thread_freq_arrays[t][c];
            }
        }

        // Find the community ID with the minimum frequency
        size_t min_freq_comm_id =
            __gnu_parallel::min_element(frequency_array_pre.begin(),
                                        frequency_array_pre.end()) -
            frequency_array_pre.begin();

        // Reassign nodes with frequency below the threshold

        if (!frequency_array_pre.empty())
        {
            // Use nth_element instead of full sort - we only need the k-th largest value
            std::vector<size_t> sorted_freq_array = frequency_array_pre;
            if (frequency_threshold > 0 && frequency_threshold <= sorted_freq_array.size())
            {
                // nth_element is O(n) vs O(n log n) for full sort
                std::nth_element(sorted_freq_array.begin(), 
                                 sorted_freq_array.begin() + frequency_threshold - 1,
                                 sorted_freq_array.end(),
                                 std::greater<size_t>());
                frequency_threshold = sorted_freq_array[frequency_threshold - 1];
            }
            else
            {
                frequency_threshold = *std::min_element(sorted_freq_array.begin(), sorted_freq_array.end());
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

        // Fill the frequency array using thread-local histograms (reuse buffers)
        // Parallel reset of thread-local arrays
        #pragma omp parallel for
        for (int t = 0; t < num_threads_freq; ++t) {
            std::memset(thread_freq_arrays[t].data(), 0, thread_freq_arrays[t].size() * sizeof(size_t));
        }
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local_freq = thread_freq_arrays[tid];
            
            #pragma omp for nowait
            for (size_t i = 0; i < comm_ids.size(); ++i)
            {
                ++local_freq[comm_ids[i]];
            }
        }
        
        // Merge thread-local histograms
        #pragma omp parallel for schedule(static)
        for (size_t c = 0; c <= num_comm; ++c)
        {
            for (int t = 0; t < num_threads_freq; ++t)
            {
                frequency_array[c] += thread_freq_arrays[t][c];
            }
        }

        // Create a vector of pairs (frequency, community ID) for sorting
        // First count non-zero entries using parallel reduction
        size_t non_zero_count = 0;
        #pragma omp parallel for reduction(+:non_zero_count)
        for (size_t i = 0; i < frequency_array.size(); ++i)
        {
            if (frequency_array[i] >= 1) ++non_zero_count;
        }
        
        // Pre-allocate with exact size and fill
        std::vector<std::pair<size_t, size_t>> freq_comm_pairs;
        freq_comm_pairs.reserve(non_zero_count);
        for (size_t i = 0; i < frequency_array.size(); ++i)
        {
            if (frequency_array[i] >= 1)
                freq_comm_pairs.emplace_back(frequency_array[i], i);
        }

        // Sort the pairs by frequency in descending order
        __gnu_parallel::sort(freq_comm_pairs.begin(), freq_comm_pairs.end(),
                             std::greater<>());

        // Get community IDs from sorted pairs and build lookup array in one pass
        const size_t top_comms_size = freq_comm_pairs.size();
        std::vector<size_t> top_communities(top_comms_size);
        std::vector<bool> is_top_community(num_comm + 1, false);
        
        #pragma omp parallel for
        for (size_t i = 0; i < top_comms_size; ++i)
        {
            size_t comm_id = freq_comm_pairs[i].second;
            top_communities[i] = comm_id;
            is_top_community[comm_id] = true;
        }

        // Create thread-private edge lists for each community using flat vectors
        // (better cache performance than unordered_map)
        const int num_threads_edge = omp_get_max_threads();
        std::vector<std::vector<EdgeList>> thread_edge_lists(num_threads_edge);
        #pragma omp parallel for
        for (int t = 0; t < num_threads_edge; ++t)
        {
            thread_edge_lists[t].resize(num_comm + 1);
        }

        // Loop through the original graph and add edges to the appropriate community
        // edge list using blocked iteration for better cache locality
        const NodeID_ BLOCK_SIZE = 1024;
        const bool graph_is_weighted = g.is_weighted();  // Hoist outside hot loop
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto &local_edge_lists = thread_edge_lists[tid];

            #pragma omp for schedule(dynamic, 1) nowait
            for (NodeID_ block_start = 0; block_start < num_nodes; block_start += BLOCK_SIZE)
            {
                NodeID_ block_end = (block_start + BLOCK_SIZE < num_nodes) ? (block_start + BLOCK_SIZE) : num_nodes;
                
                for (NodeID_ i = block_start; i < block_end; ++i)
                {
                    size_t src_comm_id = comm_ids[i];
                    if (is_top_community[src_comm_id])
                    {
                        auto& target_list = local_edge_lists[src_comm_id];
                        if (graph_is_weighted)
                        {
                            for (DestID_ neighbor : g.out_neigh(i))
                            {
                                NodeID_ dest =
                                    static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).v;
                                WeightT_ weight =
                                    static_cast<NodeWeight<NodeID_, WeightT_>>(neighbor).w;
                                target_list.push_back(Edge(i, NodeWeight<NodeID_, WeightT_>(dest, weight)));
                            }
                        }
                        else
                        {
                            for (DestID_ neighbor : g.out_neigh(i))
                            {
                                target_list.push_back(Edge(i, neighbor));
                            }
                        }
                    }
                }
            }
        }

        // Merge thread-private edge lists into the final community edge lists
        // Calculate total sizes per community using flat array - parallelized by community
        std::vector<size_t> comm_sizes(num_comm + 1, 0);
        #pragma omp parallel for
        for (size_t c = 0; c <= num_comm; ++c)
        {
            for (int t = 0; t < num_threads_edge; ++t)
            {
                comm_sizes[c] += thread_edge_lists[t][c].size();
            }
        }
        
        // Pre-allocate with known sizes using flat vector
        std::vector<EdgeList> community_edge_lists(num_comm + 1);
        #pragma omp parallel for
        for (size_t c = 0; c <= num_comm; ++c)
        {
            if (comm_sizes[c] > 0)
                community_edge_lists[c].reserve(comm_sizes[c]);
        }
        
        // Now merge without reallocations - can parallelize by community
        #pragma omp parallel for schedule(dynamic)
        for (size_t c = 0; c <= num_comm; ++c)
        {
            if (comm_sizes[c] > 0)
            {
                auto &comm_edge_list = community_edge_lists[c];
                for (const auto &local_edge_lists : thread_edge_lists)
                {
                    const auto &local_list = local_edge_lists[c];
                    comm_edge_list.insert(comm_edge_list.end(), local_list.begin(),
                                          local_list.end());
                }
            }
        }

        // Print the segmented edge lists (optional)
        // for (const auto &[comm_id, edge_list] : community_edge_lists)
        // {
        //     std::cout << "Community ID " << comm_id
        //               << " edge list: " << edge_list.size() << "\n";
        //     // for (const auto &edge : edge_list)
        //     // {
        //     //     std::cout << "(" << std::get<0>(edge) << ", " << std::get<1>(edge)
        //     //     << ", " << std::get<2>(edge) << ")\n";
        //     // }
        // }

        // Create flat vector to store the new_ids vectors for each community
        // (indexed by community ID for O(1) access instead of hash lookup)
        std::vector<std::vector<std::pair<size_t, NodeID_>>> community_id_mappings(num_comm + 1);

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

        tm_2.Start();

        // Hoist allocations outside the loop to avoid repeated allocation
        const int num_threads_inner = omp_get_max_threads();
        std::vector<std::vector<std::pair<size_t, NodeID_>>> thread_local_mappings(num_threads_inner);
        pvector<NodeID_> new_ids_sub(num_nodes);

        for (size_t idx = 0; idx < top_communities.size(); ++idx)
        {
            size_t comm_id = top_communities[idx];
            auto &edge_list = community_edge_lists[comm_id];
            
            // Reset new_ids_sub instead of reallocating
            #pragma omp parallel for
            for (size_t i = 0; i < static_cast<size_t>(num_nodes); ++i)
            {
                new_ids_sub[i] = -1;
            }
            // GenerateRabbitOrderMappingEdgelist(edge_list, new_ids_sub);

            // double modularity =
            //     GenerateRabbitModularityEdgelist(edge_list, g.is_weighted());

            // PrintTime("Sub-Modularity", modularity);
            std::cout << "Community ID " << comm_id
                      << " edge list: " << edge_list.size() << "\n";


            ReorderingAlgo reordering_algo_nest = algo;

            if (numLevels > 1 && idx == 0)
            {
                reordering_algo_nest = ReorderingAlgo::GraphBrewOrder;
                // reordering_options[0] = std::to_string(static_cast<int>(frequency_threshold * 1.5));
                if (reordering_options.size() > 2)
                {
                    reordering_options[2] = std::to_string(static_cast<double>(resolution));
                }
                else
                {
                    reordering_options.push_back(std::to_string(static_cast<double>(resolution)));
                }
            }

            // if (numLevels == 1 && idx == 0)
            // {
            //     reordering_algo_nest = ReorderingAlgo::RabbitOrder;
            // }


            // Initialize with default values
            std::vector<std::string> leiden_reordering_options = {"1.0", "30", "30"};
            std::vector<std::string> next_reordering_options;

            if (reordering_algo_nest == ReorderingAlgo::LeidenOrder)
            {
                // Customize options for LeidenOrder
                leiden_reordering_options[0] = reordering_options[2];
                leiden_reordering_options[1] = std::to_string(static_cast<int>(maxIterations));
                leiden_reordering_options[2] = std::to_string(static_cast<int>(maxPasses));
                std::cout << "Resolution Next: " << reordering_options[2] << std::endl;
            }


            next_reordering_options = (reordering_algo_nest == ReorderingAlgo::LeidenOrder) ? leiden_reordering_options : reordering_options;


            if(edge_list.size() > 0)
                GenerateMappingLocalEdgelist(g, edge_list, new_ids_sub, reordering_algo_nest, true,
                                             next_reordering_options, numLevels, true);
            else
                return;

            // Add id pairs to the corresponding community list using thread-local buffers
            // to avoid critical section bottleneck (buffers hoisted outside loop)
            // Clear thread-local buffers for reuse (parallel clear)
            #pragma omp parallel for
            for (int t = 0; t < num_threads_inner; ++t) {
                thread_local_mappings[t].clear();
            }
            
            #pragma omp parallel
            {
                int tid = omp_get_thread_num();
                auto& local_buf = thread_local_mappings[tid];
                
                #pragma omp for nowait
                for (size_t i = 0; i < new_ids_sub.size(); ++i)
                {
                    size_t src_comm_id = comm_ids[i];
                    if (new_ids_sub[i] != -1 && comm_id == src_comm_id)
                    {
                        local_buf.emplace_back(i, new_ids_sub[i]);
                    }
                }
            }
            
            // Merge thread-local buffers (single-threaded merge is fast for this)
            for (int t = 0; t < num_threads_inner; ++t) {
                auto& target = community_id_mappings[comm_id];
                target.insert(target.end(), 
                              thread_local_mappings[t].begin(), 
                              thread_local_mappings[t].end());
            }
        }

        tm_2.Stop();

        // Calculate the total size and assign consecutive indices directly
        // Note: The sort was redundant since we overwrite id_pairs[i].second = i anyway
        // Compute prefix sums for running indices (allows parallelization)
        std::vector<size_t> comm_start_indices(top_communities.size() + 1);
        comm_start_indices[0] = 0;
        for (size_t c = 0; c < top_communities.size(); ++c)
        {
            comm_start_indices[c + 1] = comm_start_indices[c] + 
                community_id_mappings[top_communities[c]].size();
        }
        
        // Initialize new_ids and assign indices in a single parallel pass
        #pragma omp parallel for
        for (size_t i = 0; i < new_ids.size(); ++i)
        {
            new_ids[i] = (NodeID_)-1;
        }
        
        // Parallel index assignment and new_ids population (fused loops)
        #pragma omp parallel for schedule(dynamic)
        for (size_t c = 0; c < top_communities.size(); ++c)
        {
            size_t comm_id = top_communities[c];
            auto &id_pairs = community_id_mappings[comm_id];
            const size_t start_idx = comm_start_indices[c];
            
            for (size_t i = 0; i < id_pairs.size(); ++i)
            {
                id_pairs[i].second = start_idx + i;
                new_ids[id_pairs[i].first] = (NodeID_)(start_idx + i);
            }
        }

        // std::cout << std::endl;
        // for (size_t i = 0; i < new_ids.size(); ++i)
        // {
        //     std::cout <<  i << "->" << new_ids[i] << std::endl;
        // }
        // std::cout << std::endl;

        if(!recursion)
            PrintTime("GraphBrew Map Time", tm_2.Seconds());

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

        // Pre-allocate edges array with exact size
        std::vector<std::tuple<size_t, size_t, double>> edges(num_edges);
        const bool is_weighted = g.is_weighted();
        
        // Parallel edge construction
        #pragma omp parallel for schedule(dynamic, 1024)
        for (NodeID_ i = 0; i < g.num_nodes(); i++)
        {
            NodeID_ out_start = g.out_offset(i);
            NodeID_ idx = 0;
            for (DestID_ j : g.out_neigh(i))
            {
                auto& edge = edges[out_start + idx];
                if (is_weighted) {
                    std::get<0>(edge) = i;
                    std::get<1>(edge) = static_cast<NodeWeight<NodeID_, WeightT_>>(j).v;
                    std::get<2>(edge) = static_cast<NodeWeight<NodeID_, WeightT_>>(j).w;
                } else {
                    std::get<0>(edge) = i;
                    std::get<1>(edge) = j;
                    std::get<2>(edge) = 1.0;
                }
                ++idx;
            }
        }

        if (g.directed())
        {
            if (num_edges < g.num_edges_directed())
            {
                // Count additional edges needed from in-neighbors
                int64_t additional_edges = g.num_edges_directed() - num_edges;
                size_t old_size = edges.size();
                edges.resize(old_size + additional_edges);
                
                // Count in-degrees to compute offsets
                pvector<NodeID_> in_degrees(g.num_nodes());
                #pragma omp parallel for
                for (NodeID_ i = 0; i < g.num_nodes(); i++)
                {
                    in_degrees[i] = g.in_degree(i);
                }
                
                // Use parallel prefix sum for write positions
                pvector<SGOffset> in_offsets_base = ParallelPrefixSum(in_degrees);
                
                // Parallel edge construction
                bool is_weighted = g.is_weighted();
                #pragma omp parallel for schedule(dynamic, 1024)
                for (NodeID_ i = 0; i < g.num_nodes(); i++)
                {
                    int64_t write_idx = old_size + in_offsets_base[i];
                    for (DestID_ j : g.in_neigh(i))
                    {
                        auto &edge = edges[write_idx++];
                        if (is_weighted)
                        {
                            std::get<0>(edge) = i;
                            std::get<1>(edge) = static_cast<NodeWeight<NodeID_, WeightT_>>(j).v;
                            std::get<2>(edge) = static_cast<NodeWeight<NodeID_, WeightT_>>(j).w;
                        }
                        else
                        {
                            std::get<0>(edge) = i;
                            std::get<1>(edge) = j;
                            std::get<2>(edge) = 1.0;
                        }
                    }
                }
            }
        }

        // No need for shrink_to_fit since we sized exactly

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

        // Use flat array with stride for better cache locality (SoA pattern)
        const size_t stride = num_nodesx;
        std::vector<K> communityDataFlat(num_nodesx * num_passes);
        
        #pragma omp parallel for
        for (size_t i = 0; i < num_nodesx; ++i)
        {
            communityDataFlat[i] = i;                    // column 0: node ID
            communityDataFlat[stride + i] = x.degree(i); // column 1: degree
        }

        for (size_t p = 0; p < num_passes - 2; ++p)
        {
            K* dest_col = &communityDataFlat[(2 + p) * stride];
            #pragma omp parallel for
            for (size_t j = 0; j < num_nodesx; ++j)
            {
                dest_col[j] = x.communityMappingPerPass[p][j];
            }
        }

        // Sort by each pass using index array
        std::vector<size_t> sort_indices(num_nodesx);
        #pragma omp parallel for
        for (size_t i = 0; i < num_nodesx; ++i) {
            sort_indices[i] = i;
        }
        
        for (size_t p = 1; p < num_passes; ++p)
        {
            const K* sort_key_col = &communityDataFlat[p * stride];
            __gnu_parallel::stable_sort(sort_indices.begin(), sort_indices.end(),
                [sort_key_col](size_t a, size_t b) {
                    return sort_key_col[a] < sort_key_col[b];
                });
        }

        pvector<NodeID_> interim_ids(num_nodes, -1);

        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; i++)
        {
            interim_ids[communityDataFlat[sort_indices[i]]] = (NodeID_)i;
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
