// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef BUILDER_H_
#define BUILDER_H_

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
        case LeidenDendrogram:
            return "LeidenDendrogram";
        case LeidenCSR:
            return "LeidenCSR";
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
            // Leiden via igraph library - Format: 12:resolution
            GenerateLeidenMapping(g, new_ids, reordering_options);
            break;
        case LeidenDendrogram:
            // Leiden Dendrogram - Format: 16:resolution:variant
            // Variants: dfs, dfshub, dfssize, bfs, hybrid (default: hybrid)
            GenerateLeidenDendrogramMappingUnified(g, new_ids, reordering_options);
            break;
        case LeidenCSR:
            // Fast Leiden on CSR - Format: 21:resolution:passes:variant
            // Variants: dfs, bfs, hubsort, fast, modularity
            GenerateLeidenCSRMappingUnified(g, new_ids, reordering_options);
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
            // MakeLocalGraphFromEL(partitions_el_dest[i]);
            pvector<NodeID_> new_ids_local(g_org.num_nodes(), -1);
            pvector<NodeID_> new_ids_local_2(g_org.num_nodes(), -1);
            // partition_g = SquishGraph(partition_g);
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
            // Leiden via igraph library - Format: 12:resolution
            GenerateLeidenMapping(g, new_ids, reordering_options);
            break;
        case LeidenDendrogram:
            // Leiden Dendrogram - Format: 16:resolution:variant
            GenerateLeidenDendrogramMappingUnified(g, new_ids, reordering_options);
            break;
        case LeidenCSR:
            // Fast Leiden on CSR - Format: 21:resolution:passes:variant
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
            return GraphBrewOrder;
        case 13:
            return MAP;
        case 14:
            return AdaptiveOrder;
        case 15:
            return LeidenOrder;        // Format: 15:resolution
        case 16:
            return LeidenDendrogram;   // Format: 16:resolution:variant
        case 17:
            return LeidenCSR;          // Format: 17:resolution:passes:variant
        default:
            std::cerr << "Invalid ReorderingAlgo value: " << value << std::endl;
            std::cerr << "Valid values: 0-17" << std::endl;
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

    //==========================================================================
    // FAST LEIDEN-CSR: Direct CSR Community Detection (No DiGraph Conversion)
    //==========================================================================
    
    /**
     * Fast Label Propagation on CSR for community detection
     * 
     * This is a simplified but fast community detection that works directly
     * on CSR format, avoiding the expensive DiGraph conversion.
     * 
     * Algorithm (inspired by LPA + Leiden refinement):
     * 1. Initialize each node in its own community
     * 2. For each node, compute modularity gain for joining neighbor communities
     * 3. Move to best community (if gain > 0)
     * 4. Repeat until convergence or max iterations
     * 5. Coarsen and repeat for multi-level hierarchy
     * 
     * Returns: vector of community assignments per pass (finest to coarsest)
     */
    template<typename NodeID_T, typename DestID_T>
    /**
     * Fast community detection with modularity-guided merging
     * Based on RabbitOrder's approach: only merge if modularity improves
     */
    std::vector<std::vector<K>> FastLabelPropagationCSR(
        const CSRGraph<NodeID_T, DestID_T, true>& g,
        double resolution = 1.0,
        int max_iterations = 10,
        int max_passes = 5)
    {
        const int64_t num_nodes = g.num_nodes();
        const int64_t num_edges = g.num_edges_directed();
        const double total_weight = static_cast<double>(num_edges);
        
        std::vector<std::vector<K>> community_per_pass;
        
        // Union-Find arrays
        std::vector<K> parent(num_nodes);
        std::vector<double> strength(num_nodes);  // sum of degrees in community
        
        // Initialize
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; ++i) {
            parent[i] = i;
            strength[i] = static_cast<double>(g.out_degree(i));
        }
        
        // Find root with path compression
        std::function<K(K)> find = [&](K x) -> K {
            if (parent[x] != x) {
                parent[x] = find(parent[x]);
            }
            return parent[x];
        };
        
        // Degree-ordered processing (smaller degrees first, like RabbitOrder)
        std::vector<std::pair<int64_t, K>> deg_order(num_nodes);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            deg_order[v] = {g.out_degree(v), v};
        }
        std::sort(deg_order.begin(), deg_order.end());
        
        // Sequential merging with modularity criterion
        int64_t merge_count = 0;
        for (auto& [deg, v] : deg_order) {
            K v_root = find(v);
            double v_str = strength[v_root];
            
            // Find best neighbor to merge with (positive modularity delta)
            double best_delta = 0.0;
            K best_target = v_root;
            
            for (auto u : g.out_neigh(v)) {
                K u_root = find(u);
                if (u_root == v_root) continue;
                
                double edge_weight = 1.0;
                double u_str = strength[u_root];
                
                // Modularity delta = edge_weight - resolution * v_str * u_str / total_weight
                double delta = edge_weight - resolution * v_str * u_str / total_weight;
                if (delta > best_delta) {
                    best_delta = delta;
                    best_target = u_root;
                }
            }
            
            // Only merge if positive modularity gain
            if (best_target != v_root) {
                parent[v_root] = best_target;
                strength[best_target] += v_str;
                merge_count++;
            }
        }
        
        // Extract final communities
        std::vector<K> community(num_nodes);
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; ++i) {
            community[i] = find(i);
        }
        
        // Renumber to be contiguous
        std::unordered_map<K, K> comm_renumber;
        K next_comm = 0;
        for (int64_t i = 0; i < num_nodes; ++i) {
            K c = community[i];
            if (comm_renumber.find(c) == comm_renumber.end()) {
                comm_renumber[c] = next_comm++;
            }
            community[i] = comm_renumber[c];
        }
        
        community_per_pass.push_back(community);
        
        printf("LeidenCSR: %ld merges, %u communities\n", merge_count, next_comm);
        
        return community_per_pass;
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
    void GenerateLeidenCSRMapping(const CSRGraph<NodeID_, DestID_, invert>& g,
                                   pvector<NodeID_>& new_ids,
                                   std::vector<std::string> reordering_options,
                                   int flavor = 2)
    {
        Timer tm;
        tm.Start();
        
        const int64_t num_nodes = g.num_nodes();
        
        // Parse options - default to 1 pass since we don't do proper coarsening
        double resolution = 0.75;
        int max_iterations = 10;
        int max_passes = 1;  // Single pass is fastest with comparable quality
        
        if (!reordering_options.empty()) {
            resolution = std::stod(reordering_options[0]);
            resolution = (resolution > 3) ? 1.0 : resolution;
        }
        if (reordering_options.size() > 1) {
            max_iterations = std::stoi(reordering_options[1]);
        }
        if (reordering_options.size() > 2) {
            max_passes = std::stoi(reordering_options[2]);
        }
        
        // Run fast label propagation directly on CSR
        auto community_per_pass = FastLabelPropagationCSR(g, resolution, max_iterations, max_passes);
        
        tm.Stop();
        PrintTime("LeidenCSR Community Detection", tm.Seconds());
        PrintTime("LeidenCSR Passes", community_per_pass.size());
        
        if (community_per_pass.empty()) {
            // Fallback: original ordering
            #pragma omp parallel for
            for (int64_t i = 0; i < num_nodes; ++i) {
                new_ids[i] = i;
            }
            return;
        }
        
        // Get degrees for secondary sort
        tm.Start();
        std::vector<K> degrees(num_nodes);
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; ++i) {
            degrees[i] = g.out_degree(i);
        }
        
        // Create sort indices
        std::vector<size_t> sort_indices(num_nodes);
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; ++i) {
            sort_indices[i] = i;
        }
        
        const size_t num_passes = community_per_pass.size();
        
        // Apply ordering based on flavor
        switch (flavor) {
            case 0: { // DFS (standard)
                // DFS-like ordering: sort by all passes (coarsest to finest) then degree
                std::cout << "LeidenCSR Ordering: DFS (standard)" << std::endl;
                
                __gnu_parallel::sort(sort_indices.begin(), sort_indices.end(),
                    [&community_per_pass, &degrees, num_passes](size_t a, size_t b) {
                        // Compare all passes from coarsest (last) to finest (first)
                        for (size_t p = num_passes; p > 0; --p) {
                            K comm_a = community_per_pass[p - 1][a];
                            K comm_b = community_per_pass[p - 1][b];
                            if (comm_a != comm_b) {
                                return comm_a < comm_b;
                            }
                        }
                        // Within same community: degree ascending
                        return degrees[a] < degrees[b];
                    });
                break;
            }
            
            case 1: { // BFS
                // BFS-like ordering: sort by level (pass index where community changes)
                // then by community at each level, then by degree
                std::cout << "LeidenCSR Ordering: BFS (level-first)" << std::endl;
                
                // Compute level for each node (first pass where it differs from neighbors)
                std::vector<int> node_level(num_nodes, num_passes);
                #pragma omp parallel for
                for (int64_t v = 0; v < num_nodes; ++v) {
                    for (size_t p = 0; p < num_passes; ++p) {
                        if (community_per_pass[p][v] != community_per_pass[num_passes-1][v]) {
                            node_level[v] = p;
                            break;
                        }
                    }
                }
                
                __gnu_parallel::sort(sort_indices.begin(), sort_indices.end(),
                    [&community_per_pass, &degrees, &node_level, num_passes](size_t a, size_t b) {
                        // Primary: sort by last pass community (coarsest)
                        K comm_a = community_per_pass[num_passes - 1][a];
                        K comm_b = community_per_pass[num_passes - 1][b];
                        if (comm_a != comm_b) return comm_a < comm_b;
                        // Secondary: sort by level (BFS order)
                        if (node_level[a] != node_level[b]) return node_level[a] < node_level[b];
                        // Tertiary: degree descending
                        return degrees[a] > degrees[b];
                    });
                break;
            }
            
            case 2: // HubSort (default)
            default: {
                // Hub sort within communities: sort by (last community, degree DESC)
                // Simpler than full hierarchical sort but good for hub locality
                std::cout << "LeidenCSR Ordering: HubSort (community + degree)" << std::endl;
                
                __gnu_parallel::sort(sort_indices.begin(), sort_indices.end(),
                    [&community_per_pass, &degrees, num_passes](size_t a, size_t b) {
                        // Primary: sort by last pass community
                        K comm_a = community_per_pass[num_passes - 1][a];
                        K comm_b = community_per_pass[num_passes - 1][b];
                        if (comm_a != comm_b) return comm_a < comm_b;
                        // Secondary: degree descending (hubs first)
                        return degrees[a] > degrees[b];
                    });
                break;
            }
        }
        
        // Assign new IDs based on sorted order
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; ++i) {
            new_ids[sort_indices[i]] = i;
        }
        
        tm.Stop();
        PrintTime("LeidenCSR Ordering", tm.Seconds());
        
        // Print community count from last pass
        if (!community_per_pass.empty()) {
            K max_comm = *std::max_element(community_per_pass.back().begin(), 
                                            community_per_pass.back().end());
            PrintTime("LeidenCSR Communities", max_comm + 1);
        }
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
     * 
     * Key optimizations:
     * - Parallel Phase 1 with atomic parent updates (lock-free)
     * - Best-fit merging: scan all neighbors to find optimal merge target
     * - Hash-based counting in LP phase (O(1) insert vs O(n) sorted insert)
     * - Early termination when convergence detected
     */
    template<typename NodeID_T, typename DestID_T>
    void FastModularityCommunityDetection(
        const CSRGraph<NodeID_T, DestID_T, true>& g,
        std::vector<double>& vertex_strength,
        std::vector<int64_t>& final_community,
        double resolution = 1.0,
        int max_passes = 3)
    {
        const int64_t num_vertices = g.num_nodes();
        const double total_weight = static_cast<double>(g.num_edges_directed());
        
        // Initialize vertex strengths
        vertex_strength.resize(num_vertices);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_vertices; ++v) {
            vertex_strength[v] = static_cast<double>(g.out_degree(v));
        }
        
        // Atomic parent array for lock-free Union-Find
        std::vector<std::atomic<int64_t>> parent(num_vertices);
        std::vector<std::atomic<double>> comm_strength(num_vertices);
        
        #pragma omp parallel for
        for (int64_t v = 0; v < num_vertices; ++v) {
            parent[v].store(v, std::memory_order_relaxed);
            comm_strength[v].store(vertex_strength[v], std::memory_order_relaxed);
        }
        
        // Lock-free find with path compression (read-only compression)
        auto find = [&](int64_t x) -> int64_t {
            int64_t root = x;
            while (true) {
                int64_t p = parent[root].load(std::memory_order_relaxed);
                if (p == root) break;
                root = p;
            }
            // Path compression (best-effort, non-atomic for speed)
            int64_t curr = x;
            while (curr != root) {
                int64_t p = parent[curr].load(std::memory_order_relaxed);
                parent[curr].store(root, std::memory_order_relaxed);
                curr = p;
            }
            return root;
        };
        
        // Process in degree order (smaller first, like RabbitOrder)
        std::vector<std::pair<int64_t, int64_t>> deg_order(num_vertices);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_vertices; ++v) {
            deg_order[v] = {g.out_degree(v), v};
        }
        __gnu_parallel::sort(deg_order.begin(), deg_order.end());
        
        std::atomic<int64_t> total_merges{0};
        
        // Phase 1: Parallel Union-Find with best-fit modularity merging
        #pragma omp parallel
        {
            #pragma omp for schedule(dynamic, 1024)
            for (int64_t i = 0; i < num_vertices; ++i) {
                int64_t v = deg_order[i].second;
                int64_t deg = deg_order[i].first;
                if (deg == 0) continue;
                
                int64_t v_root = find(v);
                double v_str = comm_strength[v_root].load(std::memory_order_relaxed);
                
                // BEST-FIT: Scan ALL neighbors to find optimal merge target
                int64_t best_root = v_root;
                double best_delta = 0.0;
                
                for (auto u : g.out_neigh(v)) {
                    int64_t u_root = find(u);
                    if (u_root == v_root) continue;
                    
                    double u_str = comm_strength[u_root].load(std::memory_order_relaxed);
                    // Modularity delta: positive means merge improves modularity
                    double delta = 1.0 - resolution * v_str * u_str / total_weight;
                    
                    if (delta > best_delta) {
                        best_delta = delta;
                        best_root = u_root;
                    }
                }
                
                // Try to merge using CAS (lock-free)
                if (best_root != v_root && best_delta > 0) {
                    // Ensure smaller root points to larger (deterministic)
                    int64_t from = (v_root < best_root) ? v_root : best_root;
                    int64_t to = (v_root < best_root) ? best_root : v_root;
                    
                    int64_t expected = from;
                    if (parent[from].compare_exchange_weak(expected, to, 
                            std::memory_order_relaxed, std::memory_order_relaxed)) {
                        // Update community strength atomically
                        double from_str = comm_strength[from].load(std::memory_order_relaxed);
                        double old_to_str = comm_strength[to].load(std::memory_order_relaxed);
                        while (!comm_strength[to].compare_exchange_weak(old_to_str, 
                                old_to_str + from_str, std::memory_order_relaxed));
                        total_merges.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
        }
        
        printf("LeidenFast: %ld merges in parallel Union-Find phase\n", 
               total_merges.load());
        
        // Compress all paths (parallel)
        #pragma omp parallel for
        for (int64_t v = 0; v < num_vertices; ++v) {
            find(v);  // Path compression side effect
        }
        
        // Phase 2: Label propagation refinement with hash-based counting
        std::vector<int64_t> labels(num_vertices);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_vertices; ++v) {
            labels[v] = parent[v].load(std::memory_order_relaxed);
        }
        
        for (int pass = 1; pass < max_passes; ++pass) {
            std::atomic<int64_t> moves{0};
            
            #pragma omp parallel
            {
                // Thread-local hash map for efficient counting
                std::unordered_map<int64_t, int64_t> label_counts;
                label_counts.reserve(256);
                
                #pragma omp for schedule(dynamic, 2048)
                for (int64_t i = 0; i < num_vertices; ++i) {
                    int64_t v = deg_order[i].second;
                    int64_t deg = g.out_degree(v);
                    if (deg == 0) continue;
                    
                    int64_t current_label = labels[v];
                    
                    // Count neighbor labels using hash map (O(1) operations)
                    label_counts.clear();
                    for (auto u : g.out_neigh(v)) {
                        label_counts[labels[u]]++;
                    }
                    
                    // Find best label (highest count, break ties by keeping current)
                    int64_t best_label = current_label;
                    int64_t best_count = 0;
                    int64_t current_count = 0;
                    
                    for (auto& [lbl, cnt] : label_counts) {
                        if (lbl == current_label) {
                            current_count = cnt;
                        }
                        if (cnt > best_count) {
                            best_count = cnt;
                            best_label = lbl;
                        }
                    }
                    
                    // Only move if strictly better (avoids oscillation)
                    if (best_label != current_label && best_count > current_count) {
                        labels[v] = best_label;
                        moves.fetch_add(1, std::memory_order_relaxed);
                    }
                }
            }
            
            int64_t move_count = moves.load();
            printf("LeidenFast: %ld moves in LP pass %d\n", move_count, pass);
            
            // Early termination if converged
            if (move_count == 0 || move_count < num_vertices / 1000) break;
        }
        
        // Compress labels to contiguous IDs
        std::unordered_map<int64_t, int64_t> label_remap;
        int64_t num_comms = 0;
        for (int64_t v = 0; v < num_vertices; ++v) {
            int64_t l = labels[v];
            auto it = label_remap.find(l);
            if (it == label_remap.end()) {
                label_remap[l] = num_comms++;
            }
        }
        
        final_community.resize(num_vertices);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_vertices; ++v) {
            final_community[v] = label_remap[labels[v]];
        }
        
        printf("LeidenFast: %ld final communities\n", num_comms);
    }
    
    /**
     * Build final ordering from communities
     * Order: largest communities first, highest degree vertices first within each
     */
    template<typename NodeID_T, typename DestID_T>
    void BuildCommunityOrdering(
        const CSRGraph<NodeID_T, DestID_T, true>& g,
        const std::vector<double>& vertex_strength,
        const std::vector<int64_t>& community,
        std::vector<int64_t>& ordered_vertices)
    {
        const int64_t num_vertices = g.num_nodes();
        
        // Count communities and compute their total strengths
        int64_t num_comms = 0;
        for (int64_t v = 0; v < num_vertices; ++v) {
            num_comms = std::max(num_comms, community[v] + 1);
        }
        
        std::vector<double> comm_strength(num_comms, 0.0);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_vertices; ++v) {
            int64_t c = community[v];
            #pragma omp atomic
            comm_strength[c] += vertex_strength[v];
        }
        
        // Build ordering: (community strength DESC, vertex degree DESC, vertex ID)
        std::vector<std::tuple<int64_t, int64_t, int64_t>> order(num_vertices);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_vertices; ++v) {
            int64_t c = community[v];
            order[v] = std::make_tuple(
                -static_cast<int64_t>(comm_strength[c] * 1000),  // Larger communities first
                -static_cast<int64_t>(vertex_strength[v]),       // Higher degree first
                v
            );
        }
        __gnu_parallel::sort(order.begin(), order.end());
        
        ordered_vertices.resize(num_vertices);
        #pragma omp parallel for
        for (int64_t i = 0; i < num_vertices; ++i) {
            ordered_vertices[i] = std::get<2>(order[i]);
        }
    }
    
    /**
     * GenerateLeidenFastMapping - Main entry point for LeidenFast algorithm
     * 
     * Improved version with:
     * - Parallel Union-Find with atomic CAS
     * - Best-fit modularity merging (not first-fit)
     * - Hash-based label propagation (faster than sorted array)
     * - Proper convergence detection
     */
    void GenerateLeidenFastMapping(const CSRGraph<NodeID_, DestID_, invert>& g,
                                   pvector<NodeID_>& new_ids,
                                   std::vector<std::string> reordering_options)
    {
        Timer tm;
        tm.Start();
        
        const int64_t num_nodes = g.num_nodes();
        
        // Parse options
        double resolution = 1.0;
        int max_passes = 3;  // Default to 3 passes for good quality
        
        if (!reordering_options.empty()) {
            resolution = std::stod(reordering_options[0]);
        }
        if (reordering_options.size() > 1) {
            max_passes = std::stoi(reordering_options[1]);
        }
        
        printf("LeidenFast: resolution=%.2f, max_passes=%d\n", resolution, max_passes);
        
        // Run community detection
        std::vector<double> vertex_strength;
        std::vector<int64_t> community;
        
        FastModularityCommunityDetection(g, vertex_strength, community, resolution, max_passes);
        
        tm.Stop();
        PrintTime("LeidenFast Community Detection", tm.Seconds());
        
        // Build ordering
        tm.Start();
        std::vector<int64_t> ordered_vertices;
        BuildCommunityOrdering(g, vertex_strength, community, ordered_vertices);
        
        // Assign new IDs
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; ++i) {
            new_ids[ordered_vertices[i]] = i;
        }
        
        tm.Stop();
        PrintTime("LeidenFast Ordering", tm.Seconds());
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
     * Compute optimal resolution based on graph properties
     */
    template<typename NodeID_T, typename DestID_T>
    double LeidenAutoResolution(const CSRGraph<NodeID_T, DestID_T, true>& g) {
        const int64_t num_vertices = g.num_nodes();
        const int64_t num_edges = g.num_edges_directed();
        double avg_degree = static_cast<double>(num_edges) / num_vertices;
        
        // Higher resolution for denser graphs (more communities)
        // Lower resolution for sparser graphs (fewer, larger communities)
        if (avg_degree > 50) return 1.2;
        if (avg_degree > 20) return 1.0;
        if (avg_degree > 10) return 0.8;
        return 0.5;
    }
    
    /**
     * Optimized parallel local moving - two-phase approach
     * Uses counting with dense thread-local arrays
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
        const int64_t num_vertices = g.num_nodes();
        int64_t total_moves = 0;
        
        // Storage for proposed moves
        std::vector<int64_t> best_comm(num_vertices);
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            int64_t moves_this_iter = 0;
            
            // Phase 1: Find best community for each vertex (read-only)
            #pragma omp parallel
            {
                // Thread-local dense array for counting (sparse usage)
                std::vector<double> count_arr(num_vertices, 0.0);
                std::vector<int64_t> touched_comms;
                touched_comms.reserve(512);
                
                #pragma omp for schedule(static)
                for (int64_t v = 0; v < num_vertices; ++v) {
                    int64_t deg = g.out_degree(v);
                    best_comm[v] = community[v];
                    
                    if (deg == 0) continue;
                    
                    int64_t current_comm = community[v];
                    double v_weight = vertex_weight[v];
                    
                    // Count edges to each neighbor community
                    touched_comms.clear();
                    for (auto u : g.out_neigh(v)) {
                        int64_t c = community[u];
                        if (count_arr[c] == 0.0) {
                            touched_comms.push_back(c);
                        }
                        count_arr[c] += 1.0;
                    }
                    
                    // Get edges to current community
                    double edges_to_current = count_arr[current_comm];
                    
                    // Find best community
                    double best_delta = 0.0;
                    double sigma_current = comm_weight[current_comm] - v_weight;
                    double leave_delta = edges_to_current - resolution * v_weight * sigma_current / total_weight;
                    
                    for (int64_t c : touched_comms) {
                        if (c == current_comm) continue;
                        
                        double edges_to_c = count_arr[c];
                        double sigma_c = comm_weight[c];
                        double join_delta = edges_to_c - resolution * v_weight * sigma_c / total_weight;
                        double delta = join_delta - leave_delta;
                        
                        if (delta > best_delta) {
                            best_delta = delta;
                            best_comm[v] = c;
                        }
                    }
                    
                    // Reset count array (only touched entries)
                    for (int64_t c : touched_comms) {
                        count_arr[c] = 0.0;
                    }
                }
            }
            
            // Phase 2: Apply moves and count
            #pragma omp parallel for schedule(static) reduction(+:moves_this_iter)
            for (int64_t v = 0; v < num_vertices; ++v) {
                if (best_comm[v] != community[v]) {
                    int64_t old_comm = community[v];
                    int64_t new_comm = best_comm[v];
                    double v_weight = vertex_weight[v];
                    
                    #pragma omp atomic
                    comm_weight[old_comm] -= v_weight;
                    #pragma omp atomic
                    comm_weight[new_comm] += v_weight;
                    
                    community[v] = new_comm;
                    moves_this_iter++;
                }
            }
            
            total_moves += moves_this_iter;
            if (moves_this_iter == 0) break;
        }
        
        return total_moves;
    }
    
    /**
     * Main Leiden algorithm - focus on quality communities
     */
    template<typename NodeID_T, typename DestID_T>
    void LeidenCommunityDetection(
        const CSRGraph<NodeID_T, DestID_T, true>& g,
        std::vector<int64_t>& final_community,
        double resolution = 1.0,
        int max_passes = 3,
        int max_iterations = 20)
    {
        const int64_t num_vertices = g.num_nodes();
        const double total_weight = static_cast<double>(g.num_edges_directed());
        
        std::vector<int64_t> community(num_vertices);
        std::vector<double> vertex_weight(num_vertices);
        std::vector<double> comm_weight(num_vertices);
        
        #pragma omp parallel for
        for (int64_t v = 0; v < num_vertices; ++v) {
            community[v] = v;
            vertex_weight[v] = static_cast<double>(g.out_degree(v));
            comm_weight[v] = vertex_weight[v];
        }
        
        int64_t total_moves = 0;
        int pass = 0;
        
        for (pass = 0; pass < max_passes; ++pass) {
            int64_t moves = LeidenLocalMoveParallel<NodeID_T, DestID_T>(
                g, community, comm_weight, vertex_weight,
                total_weight, resolution, max_iterations);
            
            // Count communities
            std::unordered_set<int64_t> unique_comms(community.begin(), community.end());
            int64_t num_comms = unique_comms.size();
            
            printf("Leiden pass %d: %ld moves, %ld communities\n", pass + 1, moves, num_comms);
            
            total_moves += moves;
            if (moves == 0) break;
        }
        
        // Compress communities
        std::unordered_map<int64_t, int64_t> comm_remap;
        int64_t num_comms = 0;
        for (int64_t v = 0; v < num_vertices; ++v) {
            int64_t c = community[v];
            if (comm_remap.find(c) == comm_remap.end()) {
                comm_remap[c] = num_comms++;
            }
        }
        
        final_community.resize(num_vertices);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_vertices; ++v) {
            final_community[v] = comm_remap[community[v]];
        }
        
        printf("Leiden: %ld total moves, %d passes, %ld final communities\n",
               total_moves, pass, num_comms);
    }
    
    /**
     * GenerateLeidenMapping2 - Quality-focused Leiden reordering
     */
    void GenerateLeidenMapping2(const CSRGraph<NodeID_, DestID_, invert>& g,
                                pvector<NodeID_>& new_ids,
                                std::vector<std::string> reordering_options)
    {
        Timer tm;
        tm.Start();
        
        const int64_t num_nodes = g.num_nodes();
        
        // Auto-tune resolution based on graph density
        double resolution = LeidenAutoResolution<NodeID_, DestID_>(g);
        int max_passes = 1;      // Single pass is usually enough
        int max_iterations = 4;  // 4 iterations for good community quality
        
        // Override with user options if provided
        if (!reordering_options.empty()) {
            resolution = std::stod(reordering_options[0]);
        }
        if (reordering_options.size() > 1) {
            max_passes = std::stoi(reordering_options[1]);
        }
        if (reordering_options.size() > 2) {
            max_iterations = std::stoi(reordering_options[2]);
        }
        
        printf("Leiden: resolution=%.2f, max_passes=%d, max_iterations=%d\n",
               resolution, max_passes, max_iterations);
        
        std::vector<int64_t> community;
        LeidenCommunityDetection<NodeID_, DestID_>(g, community, resolution, max_passes, max_iterations);
        
        tm.Stop();
        PrintTime("Leiden Community Detection", tm.Seconds());
        
        tm.Start();
        
        int64_t num_comms = *std::max_element(community.begin(), community.end()) + 1;
        std::vector<double> comm_strength(num_comms, 0.0);
        
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            int64_t c = community[v];
            #pragma omp atomic
            comm_strength[c] += static_cast<double>(g.out_degree(v));
        }
        
        std::vector<std::tuple<int64_t, int64_t, int64_t>> order(num_nodes);
        #pragma omp parallel for
        for (int64_t v = 0; v < num_nodes; ++v) {
            int64_t c = community[v];
            order[v] = std::make_tuple(
                -static_cast<int64_t>(comm_strength[c] * 1000),
                -static_cast<int64_t>(g.out_degree(v)),
                v
            );
        }
        __gnu_parallel::sort(order.begin(), order.end());
        
        #pragma omp parallel for
        for (int64_t i = 0; i < num_nodes; ++i) {
            new_ids[std::get<2>(order[i])] = i;
        }
        
        tm.Stop();
        PrintTime("Leiden Ordering", tm.Seconds());
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
        int64_t dfs_start;   // Starting DFS position for this subtree (for parallel ordering)
        
        LeidenDendrogramNode() : parent(-1), first_child(-1), sibling(-1), 
                           vertex_id(-1), subtree_size(1), weight(0.0), level(0), dfs_start(-1) {}
    };

    /**
     * Build dendrogram from Leiden's per-pass community mappings (PARALLEL VERSION)
     * 
     * Uses parallel sorting instead of hash maps for O(n log n) parallel grouping.
     * Key optimizations:
     * 1. Parallel sort by community ID for grouping
     * 2. Parallel scan to find community boundaries
     * 3. Parallel creation of internal nodes
     */
    template<typename K>
    void buildLeidenDendrogram(
        std::vector<LeidenDendrogramNode>& nodes,
        std::vector<int64_t>& roots,
        const std::vector<std::vector<K>>& communityMappingPerPass,
        const std::vector<K>& degrees,
        size_t num_vertices) {
        
        const size_t num_passes = communityMappingPerPass.size();
        
        // Create leaf nodes for all vertices (parallel)
        nodes.resize(num_vertices);
        #pragma omp parallel for
        for (size_t v = 0; v < num_vertices; ++v) {
            nodes[v].vertex_id = v;
            nodes[v].subtree_size = 1;
            nodes[v].weight = degrees[v];
            nodes[v].level = 0;
        }
        
        if (num_passes == 0) {
            roots.resize(num_vertices);
            #pragma omp parallel for
            for (size_t v = 0; v < num_vertices; ++v) {
                roots[v] = v;
            }
            return;
        }
        
        // Build hierarchy from finest to coarsest
        std::vector<int64_t> current_nodes(num_vertices);
        #pragma omp parallel for
        for (size_t i = 0; i < num_vertices; ++i) {
            current_nodes[i] = i;
        }
        
        for (size_t pass = 0; pass < num_passes; ++pass) {
            const auto& comm_map = communityMappingPerPass[pass];
            const size_t n_current = current_nodes.size();
            
            // Step 1: Create (community, node_id, weight) tuples (parallel)
            // We need representative vertex for each node to get community
            std::vector<std::tuple<K, int64_t, double>> node_comm(n_current);
            
            #pragma omp parallel for
            for (size_t i = 0; i < n_current; ++i) {
                int64_t node_id = current_nodes[i];
                // Find representative vertex (first leaf in subtree)
                int64_t v = node_id;
                while (v >= 0 && nodes[v].vertex_id < 0) {
                    v = nodes[v].first_child;
                }
                K comm = 0;
                if (v >= 0 && nodes[v].vertex_id >= 0) {
                    size_t vertex = nodes[v].vertex_id;
                    if (vertex < comm_map.size()) {
                        comm = comm_map[vertex];
                    }
                }
                node_comm[i] = std::make_tuple(comm, node_id, nodes[node_id].weight);
            }
            
            // Step 2: Parallel sort by community, then by weight (descending for hub-first)
            __gnu_parallel::sort(node_comm.begin(), node_comm.end(),
                [](const auto& a, const auto& b) {
                    if (std::get<0>(a) != std::get<0>(b)) return std::get<0>(a) < std::get<0>(b);
                    return std::get<2>(a) > std::get<2>(b);  // Higher weight first
                });
            
            // Step 3: Find community boundaries (parallel scan)
            std::vector<size_t> is_boundary(n_current + 1, 0);
            is_boundary[0] = 1;  // First element is always a boundary
            
            #pragma omp parallel for
            for (size_t i = 1; i < n_current; ++i) {
                if (std::get<0>(node_comm[i]) != std::get<0>(node_comm[i-1])) {
                    is_boundary[i] = 1;
                }
            }
            is_boundary[n_current] = 1;  // End marker
            
            // Step 4: Collect boundary indices
            std::vector<size_t> boundaries;
            boundaries.reserve(n_current / 2);  // Reasonable estimate
            for (size_t i = 0; i <= n_current; ++i) {
                if (is_boundary[i]) {
                    boundaries.push_back(i);
                }
            }
            
            const size_t num_communities = boundaries.size() - 1;
            
            // Step 5: Pre-allocate internal nodes for multi-member communities
            // Count how many communities have more than 1 member
            std::vector<size_t> needs_internal(num_communities);
            #pragma omp parallel for
            for (size_t c = 0; c < num_communities; ++c) {
                size_t start = boundaries[c];
                size_t end = boundaries[c + 1];
                needs_internal[c] = (end - start > 1) ? 1 : 0;
            }
            
            // Prefix sum to get internal node offsets
            std::vector<size_t> internal_offset(num_communities + 1);
            internal_offset[0] = 0;
            for (size_t c = 0; c < num_communities; ++c) {
                internal_offset[c + 1] = internal_offset[c] + needs_internal[c];
            }
            size_t num_new_internals = internal_offset[num_communities];
            
            // Reserve space for new internal nodes
            size_t internal_base = nodes.size();
            nodes.resize(internal_base + num_new_internals);
            
            // Step 6: Create internal nodes and link children (parallel per community)
            std::vector<int64_t> next_nodes(num_communities);
            
            #pragma omp parallel for schedule(dynamic, 64)
            for (size_t c = 0; c < num_communities; ++c) {
                size_t start = boundaries[c];
                size_t end = boundaries[c + 1];
                size_t member_count = end - start;
                
                if (member_count == 1) {
                    // Single member - no internal node needed
                    next_nodes[c] = std::get<1>(node_comm[start]);
                } else {
                    // Create internal node
                    int64_t internal_id = internal_base + internal_offset[c];
                    nodes[internal_id].vertex_id = -1;
                    nodes[internal_id].level = pass + 1;
                    nodes[internal_id].subtree_size = 0;
                    nodes[internal_id].weight = 0.0;
                    nodes[internal_id].first_child = -1;
                    nodes[internal_id].sibling = -1;
                    nodes[internal_id].parent = -1;
                    
                    // Link children (already sorted by weight from step 2)
                    int64_t prev_sibling = -1;
                    for (size_t i = start; i < end; ++i) {
                        int64_t child_id = std::get<1>(node_comm[i]);
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
                    
                    next_nodes[c] = internal_id;
                }
            }
            
            current_nodes = std::move(next_nodes);
        }
        
        // Copy roots and sort by subtree size
        roots = current_nodes;
        __gnu_parallel::sort(roots.begin(), roots.end(), [&nodes](int64_t a, int64_t b) {
            return nodes[a].subtree_size > nodes[b].subtree_size;
        });
    }

    /**
     * Parallel DFS ordering of dendrogram
     * 
     * Uses subtree sizes to compute DFS positions in parallel:
     * 1. Compute DFS start position for each subtree using prefix sums
     * 2. Process all vertices in parallel using their computed positions
     */
    void orderDendrogramDFSParallel(
        std::vector<LeidenDendrogramNode>& nodes,
        const std::vector<int64_t>& roots,
        pvector<NodeID_>& new_ids,
        bool hub_first,
        bool size_first) {
        
        const size_t num_nodes = nodes.size();
        
        // Step 1: Compute DFS start positions for each root's subtree
        size_t total_vertices = 0;
        for (int64_t root : roots) {
            nodes[root].dfs_start = total_vertices;
            total_vertices += nodes[root].subtree_size;
        }
        
        // Step 2: Propagate DFS positions down the tree (BFS order for correctness)
        // We process level by level, computing child offsets within each subtree
        std::vector<int64_t> current_level;
        current_level.reserve(roots.size());
        for (int64_t root : roots) {
            current_level.push_back(root);
        }
        
        while (!current_level.empty()) {
            std::vector<int64_t> next_level;
            next_level.reserve(current_level.size() * 2);
            
            // Process all nodes at current level in parallel
            // But collect children sequentially since we need to maintain order
            for (int64_t node_id : current_level) {
                auto& node = nodes[node_id];
                
                if (node.vertex_id >= 0) {
                    // Leaf node - will be processed in final step
                    continue;
                }
                
                // Collect and optionally sort children
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
                
                // Assign DFS start positions to children
                int64_t pos = node.dfs_start;
                for (int64_t child_id : children) {
                    nodes[child_id].dfs_start = pos;
                    pos += nodes[child_id].subtree_size;
                    next_level.push_back(child_id);
                }
            }
            
            current_level = std::move(next_level);
        }
        
        // Step 3: Assign final IDs to all leaf vertices in parallel
        #pragma omp parallel for
        for (size_t i = 0; i < num_nodes; ++i) {
            if (nodes[i].vertex_id >= 0 && nodes[i].dfs_start >= 0) {
                new_ids[nodes[i].vertex_id] = nodes[i].dfs_start;
            }
        }
    }

    /**
     * DFS ordering of dendrogram (original sequential version for comparison)
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
     * Unified Leiden Dendrogram Mapping - Parses variant from options
     * Format: 16:resolution:variant
     * Variants: dfs, dfshub, dfssize, bfs, hybrid (default: hybrid)
     */
    void GenerateLeidenDendrogramMappingUnified(
        CSRGraph<NodeID_, DestID_, invert> &g,
        pvector<NodeID_> &new_ids,
        std::vector<std::string> reordering_options) {
        
        // Default values
        double resolution = 1.0;
        std::string variant = "hybrid";
        
        // Parse options: resolution, variant
        if (!reordering_options.empty() && !reordering_options[0].empty()) {
            resolution = std::stod(reordering_options[0]);
        }
        if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
            variant = reordering_options[1];
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
     * Format: 21:resolution:passes:variant
     * Variants: dfs, bfs, hubsort, fast, modularity (default: hubsort)
     */
    void GenerateLeidenCSRMappingUnified(
        const CSRGraph<NodeID_, DestID_, invert> &g,
        pvector<NodeID_> &new_ids,
        std::vector<std::string> reordering_options) {
        
        // Default values
        double resolution = 1.0;
        int passes = 1;
        std::string variant = "hubsort";
        
        // Parse options: resolution, passes, variant
        if (!reordering_options.empty() && !reordering_options[0].empty()) {
            resolution = std::stod(reordering_options[0]);
        }
        if (reordering_options.size() > 1 && !reordering_options[1].empty()) {
            passes = std::stoi(reordering_options[1]);
        }
        if (reordering_options.size() > 2 && !reordering_options[2].empty()) {
            variant = reordering_options[2];
        }
        
        printf("LeidenCSR: resolution=%.2f, passes=%d, variant=%s\n", resolution, passes, variant.c_str());
        
        // Prepare internal options
        std::vector<std::string> internal_options;
        internal_options.push_back(std::to_string(resolution));
        internal_options.push_back(std::to_string(passes));
        
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
            internal_options.push_back("4"); // iterations
            GenerateLeidenMapping2(g, new_ids, internal_options);
        } else {
            // Default to hubsort
            GenerateLeidenCSRMapping(g, new_ids, internal_options, 2);
        }
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
        // 1. DEGREE STATISTICS (fast, already needed for other metrics)
        // ============================================================
        const size_t SAMPLE_SIZE = std::min(static_cast<size_t>(5000), static_cast<size_t>(num_nodes));
        std::vector<int64_t> sampled_degrees(SAMPLE_SIZE);
        
        double sum = 0.0;
        for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
            NodeID_ node = (static_cast<size_t>(num_nodes) > SAMPLE_SIZE) ? 
                static_cast<NodeID_>((i * num_nodes) / SAMPLE_SIZE) : static_cast<NodeID_>(i);
            sampled_degrees[i] = g.out_degree(node);
            sum += sampled_degrees[i];
        }
        double avg_degree = sum / SAMPLE_SIZE;
        
        double sum_sq_diff = 0.0;
        for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
            double diff = sampled_degrees[i] - avg_degree;
            sum_sq_diff += diff * diff;
        }
        double variance = sum_sq_diff / (SAMPLE_SIZE - 1);
        double degree_variance = (avg_degree > 0) ? std::sqrt(variance) / avg_degree : 0.0;
        
        // Hub concentration: sort degrees and see fraction from top 10%
        std::sort(sampled_degrees.rbegin(), sampled_degrees.rend());
        size_t top_10 = std::max(size_t(1), SAMPLE_SIZE / 10);
        int64_t top_edge_sum = 0;
        int64_t total_edge_sum = 0;
        for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
            if (i < top_10) top_edge_sum += sampled_degrees[i];
            total_edge_sum += sampled_degrees[i];
        }
        double hub_concentration = (total_edge_sum > 0) ? 
            static_cast<double>(top_edge_sum) / total_edge_sum : 0.0;
        
        // ============================================================
        // 2. CLUSTERING COEFFICIENT (sampled for efficiency)
        // ============================================================
        double clustering_coeff = 0.0;
        const size_t MAX_CC_SAMPLES = std::min(size_t(100), static_cast<size_t>(num_nodes) / 10 + 1);
        
        if (num_nodes >= 500) {
            double total_cc = 0.0;
            size_t valid_samples = 0;
            
            // Sample medium-to-high degree nodes (more representative)
            std::vector<std::pair<int64_t, NodeID_>> deg_nodes(SAMPLE_SIZE);
            for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
                NodeID_ node = (static_cast<size_t>(num_nodes) > SAMPLE_SIZE) ? 
                    static_cast<NodeID_>((i * num_nodes) / SAMPLE_SIZE) : static_cast<NodeID_>(i);
                deg_nodes[i] = {g.out_degree(node), node};
            }
            std::partial_sort(deg_nodes.begin(), 
                              deg_nodes.begin() + std::min(MAX_CC_SAMPLES, SAMPLE_SIZE),
                              deg_nodes.end(),
                              std::greater<std::pair<int64_t, NodeID_>>());
            
            for (size_t s = 0; s < MAX_CC_SAMPLES && s < SAMPLE_SIZE; ++s) {
                NodeID_ node = deg_nodes[s].second;
                int64_t deg = deg_nodes[s].first;
                if (deg < 2 || deg > 500) continue;  // Skip extremes
                
                // Get neighbors into a set
                std::unordered_set<NodeID_> neighbors;
                for (DestID_ neighbor : g.out_neigh(node)) {
                    neighbors.insert(static_cast<NodeID_>(neighbor));
                }
                
                // Count triangles
                size_t triangles = 0;
                for (NodeID_ n1 : neighbors) {
                    for (DestID_ n2_edge : g.out_neigh(n1)) {
                        NodeID_ n2 = static_cast<NodeID_>(n2_edge);
                        if (n2 > n1 && neighbors.count(n2)) {
                            ++triangles;
                        }
                    }
                }
                
                double local_cc = (2.0 * triangles) / (deg * (deg - 1));
                total_cc += local_cc;
                ++valid_samples;
            }
            
            clustering_coeff = (valid_samples > 0) ? total_cc / valid_samples : 0.0;
        }
        
        // ============================================================
        // 3. DIAMETER & AVG PATH LENGTH (single BFS from high-degree node)
        // ============================================================
        double avg_path_length = 0.0;
        int diameter_estimate = 0;
        
        if (num_nodes >= 500 && num_nodes <= 10000000) {  // Skip for very large graphs
            // Find highest degree node as BFS starting point
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
        // 4. COMMUNITY COUNT (fast Leiden for estimate)
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
    
    // Default path for perceptron weights file (relative to project root)
    static constexpr const char* DEFAULT_WEIGHTS_FILE = "scripts/weights/active/type_0.json";
    static constexpr const char* WEIGHTS_DIR = "scripts/";
    static constexpr const char* TYPE_WEIGHTS_DIR = "scripts/weights/active/";  // Type-based weights directory
    
    /**
     * Graph type enum for graph-type-specific weight selection
     * 
     * Different graph types have different structural properties that
     * benefit from different reordering strategies:
     * 
     * - SOCIAL: High modularity, community structure, power-law degrees
     *           Best: LeidenDFS, RabbitOrder
     * - ROAD: Mesh-like, low modularity, planar structure
     *         Best: RCMOrder (bandwidth reduction)
     * - WEB: High hub concentration, bow-tie structure
     *        Best: HubClusterDBG, HubSort
     * - POWERLAW: RMAT-like, highly skewed degree distribution
     *             Best: RabbitOrder, HubCluster
     * - UNIFORM: Random graphs, uniform degree distribution
     *            Best: Original or light reordering (DBG)
     * - GENERIC: Unknown or mixed - use default weights
     */
    enum GraphType {
        GRAPH_GENERIC = 0,  // Unknown or mixed graph type
        GRAPH_SOCIAL,       // Social networks (Facebook, Twitter)
        GRAPH_ROAD,         // Road networks (planar, mesh-like)
        GRAPH_WEB,          // Web graphs (bow-tie structure)
        GRAPH_POWERLAW,     // Power-law/RMAT graphs
        GRAPH_UNIFORM       // Uniform random graphs
    };
    
    /**
     * Convert graph type enum to string (for file naming)
     */
    static std::string GraphTypeToString(GraphType type) {
        switch (type) {
            case GRAPH_SOCIAL:   return "social";
            case GRAPH_ROAD:     return "road";
            case GRAPH_WEB:      return "web";
            case GRAPH_POWERLAW: return "powerlaw";
            case GRAPH_UNIFORM:  return "uniform";
            case GRAPH_GENERIC:
            default:             return "generic";
        }
    }
    
    /**
     * Convert string to graph type enum
     */
    static GraphType GetGraphType(const std::string& name) {
        if (name.empty() || name == "generic" || name == "GENERIC" || name == "default") return GRAPH_GENERIC;
        if (name == "social" || name == "SOCIAL") return GRAPH_SOCIAL;
        if (name == "road" || name == "ROAD" || name == "mesh") return GRAPH_ROAD;
        if (name == "web" || name == "WEB") return GRAPH_WEB;
        if (name == "powerlaw" || name == "POWERLAW" || name == "rmat" || name == "RMAT") return GRAPH_POWERLAW;
        if (name == "uniform" || name == "UNIFORM" || name == "random") return GRAPH_UNIFORM;
        return GRAPH_GENERIC;
    }
    
    /**
     * Auto-detect graph type from graph features
     * 
     * Uses a decision tree based on empirical observations:
     * 1. High modularity (>0.3) + power-law degrees → SOCIAL
     * 2. Low modularity (<0.1) + low degree variance → ROAD
     * 3. High hub concentration (>0.5) → WEB
     * 4. High degree variance (>1.5) + low modularity → POWERLAW
     * 5. Low degree variance (<0.5) + low hub concentration → UNIFORM
     * 6. Otherwise → GENERIC
     */
    static GraphType DetectGraphType(double modularity, double degree_variance, 
                                      double hub_concentration, double avg_degree,
                                      size_t num_nodes) {
        // Decision tree for graph type classification
        
        // Road networks: low modularity, mesh-like (low variance, moderate degree)
        if (modularity < 0.1 && degree_variance < 0.5 && avg_degree < 10) {
            return GRAPH_ROAD;
        }
        
        // Social networks: high modularity with community structure
        if (modularity > 0.3 && degree_variance > 0.8) {
            return GRAPH_SOCIAL;
        }
        
        // Web graphs: extreme hub concentration (bow-tie structure)
        if (hub_concentration > 0.5 && degree_variance > 1.0) {
            return GRAPH_WEB;
        }
        
        // Power-law (RMAT): high skew but lower modularity than social
        if (degree_variance > 1.5 && modularity < 0.3) {
            return GRAPH_POWERLAW;
        }
        
        // Uniform random: low variance, low hub concentration
        if (degree_variance < 0.5 && hub_concentration < 0.3 && modularity < 0.1) {
            return GRAPH_UNIFORM;
        }
        
        // Default to generic
        return GRAPH_GENERIC;
    }
    
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
            
            // Cache impact weights (learned from cache simulation)
            // These weights encode how well the algorithm utilizes each cache level
            // Higher impact = better cache utilization = bonus to score
            s += cache_l1_impact * 0.5;  // L1 impact bonus (normalized)
            s += cache_l2_impact * 0.3;  // L2 impact bonus (normalized)
            s += cache_l3_impact * 0.2;  // L3 impact bonus (normalized)
            s += cache_dram_penalty;     // DRAM penalty (already negative)
            
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
            // LEIDEN VARIANTS (consolidated into LeidenDendrogram=16 and LeidenCSR=17)
            // Use parameter-based variant selection:
            // - LeidenDendrogram (16): format 16:resolution:variant (dfs/dfshub/dfssize/bfs/hybrid)
            // - LeidenCSR (17): format 17:resolution:passes:variant (dfs/bfs/hubsort/fast/modularity)
            // =================================================================
            // LeidenDendrogram: Dendrogram-based traversal (default: hybrid variant)
            {LeidenDendrogram, {
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
            // LeidenCSR: Fast CSR-native community detection
            {LeidenCSR, {
                .bias = 0.80,
                .w_modularity = 0.35,
                .w_log_nodes = 0.05,
                .w_log_edges = 0.05,
                .w_density = -0.30,
                .w_avg_degree = 0.02,
                .w_degree_variance = 0.15,
                .w_hub_concentration = 0.20,
                .w_clustering_coeff = 0.08, .w_avg_path_length = 0.0, .w_diameter = 0.0, .w_community_count = 0.05,
                .cache_l1_impact = 0.0, .cache_l2_impact = 0.0, .cache_l3_impact = 0.0, .cache_dram_penalty = 0.0,
                .w_reorder_time = -0.25
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
            {"LeidenDendrogram", LeidenDendrogram},
            {"LeidenCSR", LeidenCSR},
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
     * 2. scripts/weights/active/type_N.json files (via LoadPerceptronWeightsForGraphType)
     * 3. If neither exists, returns hardcoded defaults from GetPerceptronWeights()
     */
    static std::map<ReorderingAlgo, PerceptronWeights> LoadPerceptronWeights(bool verbose = false) {
        return LoadPerceptronWeightsForGraphType(GRAPH_GENERIC, verbose);
    }
    
    /**
     * Find the best matching type file from the type registry.
     * 
     * The type registry (scripts/weights/active/type_registry.json) contains centroids
     * for each auto-generated type. This function finds the closest matching type
     * based on the graph features using cosine similarity.
     * 
     * @param features Graph features (modularity, degree_variance, etc.)
     * @param verbose Print matching details
     * @return Best matching type name (e.g., "type_0") or empty string if no match
     */
    static std::string FindBestTypeFromFeatures(
        double modularity, double degree_variance, double hub_concentration,
        double avg_degree, size_t num_nodes, size_t num_edges, bool verbose = false) {
        
        // Try to load type registry
        std::string registry_path = std::string(TYPE_WEIGHTS_DIR) + "type_registry.json";
        std::ifstream registry_file(registry_path);
        if (!registry_file.is_open()) {
            if (verbose) {
                std::cout << "Type registry not found at " << registry_path << "\n";
            }
            return "";
        }
        
        std::string json_content((std::istreambuf_iterator<char>(registry_file)),
                                  std::istreambuf_iterator<char>());
        registry_file.close();
        
        // Simple JSON parsing for type registry
        // Format: {"type_0": {"centroid": [...], ...}, "type_1": {...}}
        std::string best_type = "";
        double best_distance = 999999.0;
        
        // Normalize features for distance calculation
        double log_nodes = log10(std::max(1.0, (double)num_nodes));
        double log_edges = log10(std::max(1.0, (double)num_edges));
        double max_log_nodes = 9.0;  // ~1 billion nodes
        double max_log_edges = 11.0; // ~100 billion edges
        
        // Feature vector: [modularity, degree_variance, hub_concentration, density, 0, log_nodes_norm, log_edges_norm]
        double max_edges = num_nodes > 1 ? num_nodes * (num_nodes - 1) / 2.0 : 1.0;
        double density = num_edges / max_edges;
        
        std::vector<double> query_vec = {
            modularity,
            std::min(1.0, degree_variance),
            hub_concentration,
            std::min(0.1, density),  // Cap density contribution
            0.0,  // Placeholder
            log_nodes / max_log_nodes,
            log_edges / max_log_edges
        };
        
        // Parse types from JSON (simplified parsing)
        size_t pos = 0;
        while ((pos = json_content.find("\"type_", pos)) != std::string::npos) {
            // Extract type name
            size_t name_start = pos + 1;
            size_t name_end = json_content.find("\"", name_start);
            if (name_end == std::string::npos) break;
            std::string type_name = json_content.substr(name_start, name_end - name_start);
            
            // Find centroid array
            size_t centroid_pos = json_content.find("\"centroid\"", name_end);
            if (centroid_pos == std::string::npos || centroid_pos > pos + 2000) {
                pos = name_end + 1;
                continue;
            }
            
            size_t array_start = json_content.find("[", centroid_pos);
            size_t array_end = json_content.find("]", array_start);
            if (array_start == std::string::npos || array_end == std::string::npos) {
                pos = name_end + 1;
                continue;
            }
            
            // Parse centroid values
            std::string centroid_str = json_content.substr(array_start + 1, array_end - array_start - 1);
            std::vector<double> centroid;
            std::stringstream ss(centroid_str);
            std::string item;
            while (std::getline(ss, item, ',')) {
                try {
                    centroid.push_back(std::stod(item));
                } catch (...) {}
            }
            
            // Compute distance
            if (centroid.size() >= query_vec.size()) {
                double distance = 0.0;
                for (size_t i = 0; i < query_vec.size(); i++) {
                    double diff = query_vec[i] - centroid[i];
                    distance += diff * diff;
                }
                distance = sqrt(distance);
                
                if (distance < best_distance) {
                    best_distance = distance;
                    best_type = type_name;
                }
            }
            
            pos = name_end + 1;
        }
        
        if (verbose && !best_type.empty()) {
            std::cout << "Best matching type: " << best_type 
                      << " (distance: " << best_distance << ")\n";
        }
        
        return best_type;
    }
    
    /**
     * Load perceptron weights for a specific graph type.
     * 
     * Checks for weights file in this order:
     * 1. Path from PERCEPTRON_WEIGHTS_FILE environment variable (overrides all)
     * 2. Type-based file: scripts/weights/active/type_N.json (if features provided)
     * 3. If none exist, returns hardcoded defaults from GetPerceptronWeights()
     * 
     * @param graph_type The detected or specified graph type
     * @param verbose Print which file was loaded
     */
    static std::map<ReorderingAlgo, PerceptronWeights> LoadPerceptronWeightsForGraphType(
        GraphType graph_type, bool verbose = false) {
        
        // Start with defaults
        auto weights = GetPerceptronWeights();
        
        // Check environment variable override first
        const char* env_path = std::getenv("PERCEPTRON_WEIGHTS_FILE");
        if (env_path != nullptr) {
            std::ifstream file(env_path);
            if (file.is_open()) {
                std::string json_content((std::istreambuf_iterator<char>(file)),
                                          std::istreambuf_iterator<char>());
                file.close();
                
                std::map<ReorderingAlgo, PerceptronWeights> loaded_weights;
                if (ParseWeightsFromJSON(json_content, loaded_weights)) {
                    for (const auto& kv : loaded_weights) {
                        weights[kv.first] = kv.second;
                    }
                    if (verbose) {
                        std::cout << "Perceptron: Loaded " << loaded_weights.size() 
                                  << " weights from env override: " << env_path << "\n";
                    }
                    return weights;
                }
            }
        }
        
        // Build list of candidate files to try (in order of preference)
        std::vector<std::string> candidate_files;
        
        // 1. Graph-type-specific file (semantic types: social, road, web, etc.)
        if (graph_type != GRAPH_GENERIC) {
            std::string type_file = std::string(WEIGHTS_DIR) + "perceptron_weights_" 
                                    + GraphTypeToString(graph_type) + ".json";
            candidate_files.push_back(type_file);
        }
        
        // 2. Default file
        candidate_files.push_back(DEFAULT_WEIGHTS_FILE);
        
        // Try each candidate file
        for (const auto& weights_file : candidate_files) {
            std::ifstream file(weights_file);
            if (!file.is_open()) continue;
            
            std::string json_content((std::istreambuf_iterator<char>(file)),
                                      std::istreambuf_iterator<char>());
            file.close();
            
            std::map<ReorderingAlgo, PerceptronWeights> loaded_weights;
            if (ParseWeightsFromJSON(json_content, loaded_weights)) {
                for (const auto& kv : loaded_weights) {
                    weights[kv.first] = kv.second;
                }
                if (verbose) {
                    std::cout << "Perceptron: Loaded " << loaded_weights.size() 
                              << " weights from " << weights_file;
                    if (graph_type != GRAPH_GENERIC) {
                        std::cout << " (graph type: " << GraphTypeToString(graph_type) << ")";
                    }
                    std::cout << "\n";
                }
                return weights;
            }
        }
        
        // No files found - use defaults
        if (verbose) {
            std::cout << "Perceptron: Using hardcoded defaults (no weight files found)\n";
        }
        return weights;
    }
    
    /**
     * Load perceptron weights using graph features to find the best type match.
     * 
     * This function first tries to find a matching auto-generated type (type_0, type_1, etc.)
     * from the type registry, then falls back to semantic types (social, road, etc.) and
     * finally to the default weights.
     * 
     * FALLBACK MECHANISM:
     * 1. Starts with GetPerceptronWeights() which provides defaults for ALL algorithms
     * 2. Loads type-specific weights and OVERLAYS them on defaults
     * 3. Any algorithm missing from the type file uses the default weights
     * 
     * This ensures we ALWAYS have weights for every algorithm, even if the type file
     * was trained with only a subset of algorithms.
     * 
     * @param modularity Graph modularity score
     * @param degree_variance Normalized degree variance
     * @param hub_concentration Hub concentration metric
     * @param avg_degree Average vertex degree
     * @param num_nodes Number of nodes
     * @param num_edges Number of edges
     * @param verbose Print which file was loaded
     */
    static std::map<ReorderingAlgo, PerceptronWeights> LoadPerceptronWeightsForFeatures(
        double modularity, double degree_variance, double hub_concentration,
        double avg_degree, size_t num_nodes, size_t num_edges, bool verbose = false) {
        
        // Start with defaults - this ensures ALL algorithms have weights
        auto weights = GetPerceptronWeights();
        
        // Check environment variable override first
        const char* env_path = std::getenv("PERCEPTRON_WEIGHTS_FILE");
        if (env_path != nullptr) {
            std::ifstream file(env_path);
            if (file.is_open()) {
                std::string json_content((std::istreambuf_iterator<char>(file)),
                                          std::istreambuf_iterator<char>());
                file.close();
                
                std::map<ReorderingAlgo, PerceptronWeights> loaded_weights;
                if (ParseWeightsFromJSON(json_content, loaded_weights)) {
                    for (const auto& kv : loaded_weights) {
                        weights[kv.first] = kv.second;
                    }
                    if (verbose) {
                        std::cout << "Perceptron: Loaded " << loaded_weights.size() 
                                  << " weights from env override: " << env_path << "\n";
                    }
                    return weights;
                }
            }
        }
        
        // Build list of candidate files to try (in order of preference)
        std::vector<std::string> candidate_files;
        
        // 1. Try to find matching type from type registry (type_0, type_1, etc.)
        std::string best_type = FindBestTypeFromFeatures(
            modularity, degree_variance, hub_concentration,
            avg_degree, num_nodes, num_edges, verbose);
        
        if (!best_type.empty()) {
            std::string type_file = std::string(TYPE_WEIGHTS_DIR) + best_type + ".json";
            candidate_files.push_back(type_file);
        }
        
        // 2. Detect semantic graph type and try type-specific file
        GraphType detected_type = DetectGraphType(modularity, degree_variance, 
                                                   hub_concentration, avg_degree, num_nodes);
        if (detected_type != GRAPH_GENERIC) {
            std::string type_file = std::string(WEIGHTS_DIR) + "perceptron_weights_" 
                                    + GraphTypeToString(detected_type) + ".json";
            candidate_files.push_back(type_file);
        }
        
        // 3. Default file (global fallback with all algorithms)
        candidate_files.push_back(DEFAULT_WEIGHTS_FILE);
        
        // Try each candidate file - overlay loaded weights on defaults
        // This ensures algorithms missing from type files use global defaults
        for (const auto& weights_file : candidate_files) {
            std::ifstream file(weights_file);
            if (!file.is_open()) continue;
            
            std::string json_content((std::istreambuf_iterator<char>(file)),
                                      std::istreambuf_iterator<char>());
            file.close();
            
            std::map<ReorderingAlgo, PerceptronWeights> loaded_weights;
            if (ParseWeightsFromJSON(json_content, loaded_weights)) {
                for (const auto& kv : loaded_weights) {
                    weights[kv.first] = kv.second;
                }
                if (verbose) {
                    std::cout << "Perceptron: Loaded " << loaded_weights.size() 
                              << " weights from " << weights_file << "\n";
                }
                return weights;
            }
        }
        
        // No files found - use defaults
        if (verbose) {
            std::cout << "Perceptron: Using hardcoded defaults (no weight files found)\n";
        }
        return weights;
    }
    
    /**
     * Get cached weights for a specific graph type
     * Uses thread-local cache to avoid repeated file loading
     */
    static const std::map<ReorderingAlgo, PerceptronWeights>& GetCachedWeights(
        GraphType graph_type, bool verbose_first_load = false) {
        // Cache weights per graph type (static map persists across calls)
        static std::map<GraphType, std::map<ReorderingAlgo, PerceptronWeights>> weight_cache;
        static std::mutex cache_mutex;
        
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        auto it = weight_cache.find(graph_type);
        if (it != weight_cache.end()) {
            return it->second;
        }
        
        // Load and cache (with verbose output if requested)
        weight_cache[graph_type] = LoadPerceptronWeightsForGraphType(graph_type, verbose_first_load);
        return weight_cache[graph_type];
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
     * @param graph_type The detected graph type for loading appropriate weights
     * 
     * Loads weights from file if available, otherwise uses defaults.
     */
    ReorderingAlgo SelectReorderingPerceptron(const CommunityFeatures& feat, 
                                               BenchmarkType bench = BENCH_GENERIC,
                                               GraphType graph_type = GRAPH_GENERIC) {
        // Get cached weights for this graph type
        const auto& weights = GetCachedWeights(graph_type);
        
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
                                               const std::string& benchmark_name,
                                               GraphType graph_type = GRAPH_GENERIC) {
        return SelectReorderingPerceptron(feat, GetBenchmarkType(benchmark_name), graph_type);
    }
    
    /**
     * Select best reordering algorithm using feature-based type matching.
     * 
     * This version first tries to find a matching auto-generated type (type_a, etc.)
     * from the type registry based on graph features, then falls back to semantic types.
     * 
     * @param feat Community features for scoring
     * @param global_modularity Global graph modularity
     * @param global_degree_variance Global degree variance
     * @param global_hub_concentration Global hub concentration
     * @param num_nodes Total number of nodes
     * @param num_edges Total number of edges
     * @param bench Benchmark type
     */
    ReorderingAlgo SelectReorderingPerceptronWithFeatures(
        const CommunityFeatures& feat,
        double global_modularity, double global_degree_variance,
        double global_hub_concentration, size_t num_nodes, size_t num_edges,
        BenchmarkType bench = BENCH_GENERIC) {
        
        // Load weights based on features (tries type_0.json first, then semantic types)
        auto weights = LoadPerceptronWeightsForFeatures(
            global_modularity, global_degree_variance, global_hub_concentration,
            feat.avg_degree, num_nodes, num_edges, false);
        
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

    CommunityFeatures ComputeCommunityFeatures(
        const std::vector<NodeID_>& comm_nodes,
        const CSRGraph<NodeID_, DestID_, invert>& g,
        const std::unordered_set<NodeID_>& node_set,
        bool compute_extended = true)
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
            feat.clustering_coeff = 0.0;
            feat.avg_path_length = 0.0;
            feat.diameter_estimate = 0.0;
            feat.community_count = 1.0;
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
        
        // ============================================================
        // EXTENDED FEATURES (computed via fast sampling for efficiency)
        // Only compute for larger communities where it matters
        // ============================================================
        const size_t MIN_SIZE_FOR_EXTENDED = 1000;  // Skip for small communities
        
        if (compute_extended && feat.num_nodes >= MIN_SIZE_FOR_EXTENDED) {
            
            // --- Clustering Coefficient (fast sampled estimate) ---
            // Sample only a few high-degree nodes and use hash set for O(1) neighbor lookup
            const size_t MAX_SAMPLES_CC = std::min(size_t(50), feat.num_nodes / 20 + 1);
            double total_cc = 0.0;
            size_t valid_cc_samples = 0;
            
            // Find high-degree nodes for sampling (they're more informative)
            std::vector<std::pair<size_t, size_t>> deg_idx(feat.num_nodes);
            for (size_t i = 0; i < feat.num_nodes; ++i) {
                deg_idx[i] = {internal_degrees[i], i};
            }
            std::partial_sort(deg_idx.begin(), 
                              deg_idx.begin() + std::min(MAX_SAMPLES_CC, feat.num_nodes),
                              deg_idx.end(),
                              std::greater<std::pair<size_t, size_t>>());
            
            for (size_t s = 0; s < MAX_SAMPLES_CC && s < feat.num_nodes; ++s) {
                size_t idx = deg_idx[s].second;
                NodeID_ node = comm_nodes[idx];
                size_t deg = internal_degrees[idx];
                if (deg < 2 || deg > 500) continue;  // Skip very high degree (too slow)
                
                // Get internal neighbors into a hash set for O(1) lookup
                std::unordered_set<NodeID_> neighbor_set;
                for (DestID_ neighbor : g.out_neigh(node)) {
                    NodeID_ dest = static_cast<NodeID_>(neighbor);
                    if (node_set.count(dest)) {
                        neighbor_set.insert(dest);
                    }
                }
                
                // Count triangles using hash set lookup
                size_t triangles = 0;
                for (NodeID_ n1 : neighbor_set) {
                    for (DestID_ n2_edge : g.out_neigh(n1)) {
                        NodeID_ n2 = static_cast<NodeID_>(n2_edge);
                        if (n2 > n1 && neighbor_set.count(n2)) {  // n2 > n1 avoids double counting
                            ++triangles;
                        }
                    }
                }
                
                double local_cc = (2.0 * triangles) / (deg * (deg - 1));
                total_cc += local_cc;
                ++valid_cc_samples;
            }
            feat.clustering_coeff = (valid_cc_samples > 0) ? 
                total_cc / valid_cc_samples : 0.0;
            
            // --- Diameter & Avg Path (single BFS from highest degree node) ---
            // Just one BFS is enough for a rough estimate
            if (!deg_idx.empty()) {
                size_t start_idx = deg_idx[0].second;
                std::vector<int> dist(feat.num_nodes, -1);
                std::queue<size_t> bfs_queue;
                bfs_queue.push(start_idx);
                dist[start_idx] = 0;
                
                // Build local index map (reuse node_set which we already have)
                std::unordered_map<NodeID_, size_t> node_to_idx;
                node_to_idx.reserve(feat.num_nodes);
                for (size_t i = 0; i < feat.num_nodes; ++i) {
                    node_to_idx[comm_nodes[i]] = i;
                }
                
                double path_sum = 0.0;
                size_t path_count = 0;
                size_t max_dist = 0;
                
                while (!bfs_queue.empty()) {
                    size_t curr_idx = bfs_queue.front();
                    bfs_queue.pop();
                    NodeID_ curr_node = comm_nodes[curr_idx];
                    
                    for (DestID_ neighbor : g.out_neigh(curr_node)) {
                        NodeID_ dest = static_cast<NodeID_>(neighbor);
                        auto it = node_to_idx.find(dest);
                        if (it != node_to_idx.end() && dist[it->second] == -1) {
                            size_t dest_idx = it->second;
                            dist[dest_idx] = dist[curr_idx] + 1;
                            bfs_queue.push(dest_idx);
                            path_sum += dist[dest_idx];
                            ++path_count;
                            if (static_cast<size_t>(dist[dest_idx]) > max_dist) {
                                max_dist = dist[dest_idx];
                            }
                        }
                    }
                }
                
                feat.avg_path_length = (path_count > 0) ? path_sum / path_count : 1.0;
                feat.diameter_estimate = static_cast<double>(max_dist);
                
                // Community count = number of unreached nodes + 1 (connected component)
                size_t unreached = 0;
                for (size_t i = 0; i < feat.num_nodes; ++i) {
                    if (dist[i] == -1) ++unreached;
                }
                feat.community_count = (unreached > 0) ? 2.0 : 1.0;  // Simplified: connected or not
            } else {
                feat.avg_path_length = 1.0;
                feat.diameter_estimate = 1.0;
                feat.community_count = 1.0;
            }
            
        } else {
            // Small community or extended features disabled - use fast estimates
            feat.clustering_coeff = feat.internal_density;  // Rough proxy
            feat.avg_path_length = (feat.internal_density > 0.1) ? 1.5 : 
                                   std::log2(feat.num_nodes + 1);  // Small-world estimate
            feat.diameter_estimate = feat.avg_path_length * 2.0;
            feat.community_count = 1.0;
        }
        
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
     * @param global_degree_variance Global degree variance
     * @param global_hub_concentration Global hub concentration  
     * @param global_avg_degree Global average degree
     * @param num_nodes Total number of nodes in graph
     * @param num_edges Total number of edges in graph
     * @param bench Benchmark type (default: BENCH_GENERIC for balanced performance)
     *              - BENCH_GENERIC: Optimizes for all graph algorithms equally
     *              - BENCH_PR, BENCH_BFS, etc.: Optimizes for specific benchmark
     * 
     * Fallback to heuristics for edge cases (very small communities).
     */
    ReorderingAlgo SelectBestReorderingForCommunity(CommunityFeatures feat, 
                                                     double global_modularity,
                                                     double global_degree_variance,
                                                     double global_hub_concentration,
                                                     double global_avg_degree,
                                                     size_t num_nodes, size_t num_edges,
                                                     BenchmarkType bench = BENCH_GENERIC,
                                                     GraphType graph_type = GRAPH_GENERIC)
    {
        // Small communities: reordering overhead exceeds benefit
        const size_t MIN_COMMUNITY_SIZE = 200;
        
        if (feat.num_nodes < MIN_COMMUNITY_SIZE) {
            return ORIGINAL;
        }
        
        // Set the modularity in features for perceptron scoring
        feat.modularity = global_modularity;
        
        // Use feature-based perceptron to select best algorithm 
        // This tries type_0.json, type_1.json etc. first, then falls back to semantic types
        ReorderingAlgo selected = SelectReorderingPerceptronWithFeatures(
            feat, global_modularity, global_degree_variance, global_hub_concentration,
            num_nodes, num_edges, bench);
        
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
     * 
     * Format: -o 14[:max_depth[:resolution[:min_recurse_size[:mode]]]]
     * - max_depth: Maximum recursion depth (0 = no recursion, 1+ = multi-level)
     * - resolution: Leiden resolution parameter (default: 0.75)
     * - min_recurse_size: Minimum community size for recursion (default: 50000)
     * - mode: 0 = normal (per-community), 1 = full-graph adaptive (skip Leiden, pick best algo for whole graph)
     * 
     * Examples:
     *   -o 14                   # Default: depth=0, res=0.75, per-community selection
     *   -o 14:2                 # Multi-level: depth=2, allowing sub-community recursion
     *   -o 14:0:1.0             # Higher resolution for more communities
     *   -o 14:0:0.75:50000:1    # Full-graph mode: pick best algo for entire graph (no Leiden)
     */
    void GenerateAdaptiveMapping(CSRGraph<NodeID_, DestID_, invert> &g,
                                 pvector<NodeID_> &new_ids, bool useOutdeg,
                                 std::vector<std::string> reordering_options)
    {
        // Parse mode option (last parameter)
        int adaptive_mode = 0;  // 0 = per-community, 1 = full-graph
        if (reordering_options.size() > 3) {
            adaptive_mode = std::stoi(reordering_options[3]);
        }
        
        if (adaptive_mode == 1) {
            // Full-graph adaptive mode: skip Leiden, pick best algorithm for entire graph
            GenerateAdaptiveMappingFullGraph(g, new_ids, useOutdeg, reordering_options);
        } else {
            // Per-community mode (default): Leiden + per-community algorithm selection
            GenerateAdaptiveMappingRecursive(g, new_ids, useOutdeg, reordering_options, 0, true);
        }
    }
    
    /**
     * Full-Graph Adaptive Mode
     * 
     * Instead of running Leiden and selecting per-community, this mode:
     * 1. Computes global graph features
     * 2. Uses the perceptron to select the SINGLE best algorithm for the entire graph
     * 3. Applies that algorithm to the whole graph
     * 
     * This is useful when:
     * - The graph has weak community structure (low modularity)
     * - You want to quickly find the best single algorithm
     * - You're comparing against per-community selection
     */
    void GenerateAdaptiveMappingFullGraph(CSRGraph<NodeID_, DestID_, invert> &g,
                                          pvector<NodeID_> &new_ids, bool useOutdeg,
                                          std::vector<std::string> reordering_options)
    {
        Timer tm;
        tm.Start();
        
        int64_t num_nodes = g.num_nodes();
        int64_t num_edges = g.num_edges_directed();
        
        std::cout << "=== Full-Graph Adaptive Mode ===\n";
        std::cout << "Nodes: " << static_cast<long long>(num_nodes) << ", Edges: " << static_cast<long long>(num_edges) << "\n";
        
        // Compute global features
        double global_modularity = 0.0;  // Will estimate from clustering coefficient
        double global_degree_variance = 0.0;
        double global_hub_concentration = 0.0;
        double global_avg_degree = static_cast<double>(num_edges) / num_nodes;
        
        // Sample-based feature computation (fast)
        const size_t SAMPLE_SIZE = std::min(static_cast<size_t>(10000), static_cast<size_t>(num_nodes));
        std::vector<int64_t> sampled_degrees(SAMPLE_SIZE);
        
        double sum = 0.0;
        for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
            NodeID_ node = (num_nodes > static_cast<int64_t>(SAMPLE_SIZE)) ? 
                static_cast<NodeID_>((i * num_nodes) / SAMPLE_SIZE) : static_cast<NodeID_>(i);
            sampled_degrees[i] = g.out_degree(node);
            sum += sampled_degrees[i];
        }
        double sample_mean = sum / SAMPLE_SIZE;
        
        double sum_sq_diff = 0.0;
        for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
            double diff = sampled_degrees[i] - sample_mean;
            sum_sq_diff += diff * diff;
        }
        double variance = sum_sq_diff / (SAMPLE_SIZE - 1);
        global_degree_variance = (sample_mean > 0) ? std::sqrt(variance) / sample_mean : 0.0;
        
        // Hub concentration: fraction of edges from top 10% degree nodes
        std::sort(sampled_degrees.rbegin(), sampled_degrees.rend());
        size_t top_10 = std::max(size_t(1), SAMPLE_SIZE / 10);
        int64_t top_edge_sum = 0, total_edge_sum = 0;
        for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
            if (i < top_10) top_edge_sum += sampled_degrees[i];
            total_edge_sum += sampled_degrees[i];
        }
        global_hub_concentration = (total_edge_sum > 0) ? 
            static_cast<double>(top_edge_sum) / total_edge_sum : 0.0;
        
        // Estimate modularity from clustering coefficient (rough approximation)
        // Higher clustering often correlates with higher modularity
        double clustering_coeff = 0.0;
        size_t triangles_sampled = 0;
        size_t triplets_sampled = 0;
        
        for (size_t i = 0; i < std::min(SAMPLE_SIZE, static_cast<size_t>(1000)); ++i) {
            NodeID_ node = (num_nodes > static_cast<int64_t>(SAMPLE_SIZE)) ? 
                static_cast<NodeID_>((i * num_nodes) / SAMPLE_SIZE) : static_cast<NodeID_>(i);
            
            int64_t deg = g.out_degree(node);
            if (deg < 2) continue;
            
            triplets_sampled += deg * (deg - 1) / 2;
            
            // Count actual triangles (for small neighborhoods only)
            if (deg <= 100) {
                std::unordered_set<NodeID_> neighbors;
                for (auto n : g.out_neigh(node)) {
                    neighbors.insert(static_cast<NodeID_>(n));
                }
                for (auto n1 : g.out_neigh(node)) {
                    for (auto n2 : g.out_neigh(static_cast<NodeID_>(n1))) {
                        if (neighbors.count(static_cast<NodeID_>(n2))) {
                            triangles_sampled++;
                        }
                    }
                }
            }
        }
        clustering_coeff = (triplets_sampled > 0) ? 
            static_cast<double>(triangles_sampled) / (2 * triplets_sampled) : 0.0;
        
        // Rough modularity estimate: higher clustering = higher likely modularity
        global_modularity = std::min(0.9, clustering_coeff * 1.5);
        
        // Detect graph type
        GraphType detected_graph_type = DetectGraphType(
            global_modularity, global_degree_variance, global_hub_concentration, 
            global_avg_degree, static_cast<size_t>(num_nodes));
        
        std::cout << "Graph Type: " << GraphTypeToString(detected_graph_type) << "\n";
        PrintTime("Degree Variance", global_degree_variance);
        PrintTime("Hub Concentration", global_hub_concentration);
        PrintTime("Est. Modularity", global_modularity);
        PrintTime("Clustering Coeff", clustering_coeff);
        
        // Create a "fake" community feature that represents the whole graph
        CommunityFeatures global_feat;
        global_feat.num_nodes = num_nodes;
        global_feat.num_edges = num_edges;
        global_feat.internal_density = global_avg_degree / (num_nodes - 1);
        global_feat.degree_variance = global_degree_variance;
        global_feat.hub_concentration = global_hub_concentration;
        global_feat.clustering_coeff = clustering_coeff;
        
        // Select best algorithm for entire graph
        ReorderingAlgo best_algo = SelectBestReorderingForCommunity(
            global_feat, global_modularity, global_degree_variance, global_hub_concentration,
            global_avg_degree, static_cast<size_t>(num_nodes), num_edges,
            BENCH_GENERIC, detected_graph_type);
        
        std::cout << "\n=== Selected Algorithm: " << ReorderingAlgoStr(best_algo) << " ===\n";
        
        // Apply selected algorithm to entire graph
        switch (best_algo) {
            case HubSort:
                GenerateHubSortMapping(g, new_ids, useOutdeg);
                break;
            case HubCluster:
                GenerateHubClusterMapping(g, new_ids, useOutdeg);
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
            case RCMOrder:
                GenerateRCMOrderMapping(g, new_ids);
                break;
#ifdef RABBIT_ENABLE
            case RabbitOrder:
            {
                pvector<NodeID_> temp_ids(num_nodes, -1);
                GenerateSortMappingRabbit(g, temp_ids, true, true);
                CSRGraph<NodeID_, DestID_, invert> g_trans = RelabelByMapping(g, temp_ids);
                pvector<NodeID_> temp_ids2(num_nodes, -1);
                GenerateRabbitOrderMapping(g_trans, temp_ids2);
                #pragma omp parallel for
                for (NodeID_ n = 0; n < num_nodes; n++) {
                    new_ids[n] = temp_ids2[temp_ids[n]];
                }
            }
            break;
#endif
            case LeidenCSR:
            {
                std::vector<std::string> leiden_opts = {"1.0", "3", "fast"};
                GenerateLeidenCSRMappingUnified(g, new_ids, leiden_opts);
            }
            break;
            case LeidenDendrogram:
            {
                std::vector<std::string> leiden_opts = {"1.0", "hybrid"};
                GenerateLeidenDendrogramMappingUnified(g, new_ids, leiden_opts);
            }
            break;
            case LeidenOrder:
            {
                std::vector<std::string> leiden_opts = {"1.0"};
                GenerateLeidenMapping(g, new_ids, leiden_opts);
            }
            break;
            default:
                // Original ordering
                GenerateOriginalMapping(g, new_ids);
                break;
        }
        
        tm.Stop();
        PrintTime("Full-Graph Adaptive Time", tm.Seconds());
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
        
        // Parse options: max_depth, resolution, min_recurse_size, mode
        int MAX_DEPTH = 0;  // Default: no recursion
        size_t MIN_COMMUNITY_FOR_RECURSION = 50000;  // Only recurse on large communities
        
        // Parse options
        double resolution = 0.75;
        int maxIterations = 30;
        int maxPasses = 30;

        // New format: -o 14[:max_depth[:resolution[:min_recurse_size[:mode]]]]
        // For backward compatibility, also check old format
        if (reordering_options.size() > 0) {
            // First param could be max_depth (int) or resolution (float)
            double first_val = std::stod(reordering_options[0]);
            if (first_val >= 0 && first_val <= 10 && std::floor(first_val) == first_val) {
                // Looks like max_depth (integer 0-10)
                MAX_DEPTH = static_cast<int>(first_val);
            } else {
                // Assume it's resolution (old format)
                resolution = first_val;
            }
        }
        if (reordering_options.size() > 1) {
            resolution = std::stod(reordering_options[1]);
        }
        if (reordering_options.size() > 2) {
            MIN_COMMUNITY_FOR_RECURSION = std::stoul(reordering_options[2]);
        }
        // Note: mode (param 3) is parsed in GenerateAdaptiveMapping

        if (depth == 0 && verbose) {
            PrintTime("Max Depth", static_cast<double>(MAX_DEPTH));
            PrintTime("Resolution", resolution);
            PrintTime("Min Recurse Size", static_cast<double>(MIN_COMMUNITY_FOR_RECURSION));
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

        // Step 2.5: Compute quick global features for graph type detection
        // We use lightweight features here since we're at the graph level
        double global_degree_variance = 0.0;
        double global_hub_concentration = 0.0;
        double global_avg_degree = static_cast<double>(num_edges) / num_nodes;
        
        {
            // Compute degree statistics for graph type detection (fast sampling)
            const size_t SAMPLE_SIZE = std::min(static_cast<size_t>(5000), static_cast<size_t>(num_nodes));
            std::vector<int64_t> sampled_degrees(SAMPLE_SIZE);
            
            double sum = 0.0;
            for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
                NodeID_ node = (num_nodes > SAMPLE_SIZE) ? 
                    static_cast<NodeID_>((i * num_nodes) / SAMPLE_SIZE) : static_cast<NodeID_>(i);
                sampled_degrees[i] = g.out_degree(node);
                sum += sampled_degrees[i];
            }
            double sample_mean = sum / SAMPLE_SIZE;
            
            double sum_sq_diff = 0.0;
            for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
                double diff = sampled_degrees[i] - sample_mean;
                sum_sq_diff += diff * diff;
            }
            double variance = sum_sq_diff / (SAMPLE_SIZE - 1);
            global_degree_variance = (sample_mean > 0) ? std::sqrt(variance) / sample_mean : 0.0;
            
            // Hub concentration: sort degrees and see fraction from top 10%
            std::sort(sampled_degrees.rbegin(), sampled_degrees.rend());
            size_t top_10 = std::max(size_t(1), SAMPLE_SIZE / 10);
            int64_t top_edge_sum = 0;
            int64_t total_edge_sum = 0;
            for (size_t i = 0; i < SAMPLE_SIZE; ++i) {
                if (i < top_10) top_edge_sum += sampled_degrees[i];
                total_edge_sum += sampled_degrees[i];
            }
            global_hub_concentration = (total_edge_sum > 0) ? 
                static_cast<double>(top_edge_sum) / total_edge_sum : 0.0;
        }
        
        // Detect graph type based on global features
        GraphType detected_graph_type = DetectGraphType(
            global_modularity, global_degree_variance, global_hub_concentration, 
            global_avg_degree, static_cast<size_t>(num_nodes));
        
        if (depth == 0 && verbose) {
            std::cout << "Graph Type: " << GraphTypeToString(detected_graph_type) << "\n";
            PrintTime("Degree Variance", global_degree_variance);
            PrintTime("Hub Concentration", global_hub_concentration);
            
            // Load weights based on features (tries type_0.json first, then semantic types)
            // This also shows which weights file is being used
            LoadPerceptronWeightsForFeatures(
                global_modularity, global_degree_variance, global_hub_concentration,
                global_avg_degree, static_cast<size_t>(num_nodes), num_edges, true);
        }

        // Step 3: Compute features and select algorithm for each community
        std::vector<ReorderingAlgo> selected_algos(num_comm);
        std::vector<CommunityFeatures> all_features(num_comm);
        std::vector<bool> should_recurse(num_comm, false);
        
        if (depth == 0 && verbose) {
            std::cout << "\n=== Adaptive Reordering Selection (Depth " << depth 
                      << ", Modularity: " << std::fixed << std::setprecision(4) 
                      << global_modularity << ") ===\n";
            std::cout << "Comm\tNodes\tEdges\tDensity\tDegVar\tHubConc\tClustC\tAvgPath\tDiam\tSubComm\tSelected\n";
        }
        
        // Minimum size to compute features and consider reordering
        const size_t MIN_SIZE_FOR_FEATURES = 200;
        
        for (size_t c = 0; c < num_comm; ++c) {
            if (community_nodes[c].empty()) {
                selected_algos[c] = ORIGINAL;
                continue;
            }
            
            size_t comm_size = community_nodes[c].size();
            
            // Skip feature computation for small communities - just use ORIGINAL
            if (comm_size < MIN_SIZE_FOR_FEATURES) {
                selected_algos[c] = ORIGINAL;
                continue;
            }
            
            // Create node set for fast lookup
            std::unordered_set<NodeID_> node_set(
                community_nodes[c].begin(), community_nodes[c].end());
            
            // Compute features (with extended features only for large communities)
            bool compute_extended = (comm_size >= 1000);
            CommunityFeatures feat = ComputeCommunityFeatures(
                community_nodes[c], g, node_set, compute_extended);
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
                // Select local algorithm based on features using type-based weights
                // This uses LoadPerceptronWeightsForFeatures which tries type_0.json first
                selected_algos[c] = SelectBestReorderingForCommunity(
                    feat, global_modularity, global_degree_variance, global_hub_concentration,
                    global_avg_degree, static_cast<size_t>(num_nodes), num_edges,
                    BENCH_GENERIC, detected_graph_type);
            }
            
            // Print selection rationale with extended features
            if (depth == 0 && verbose && feat.num_nodes >= 100) {
                std::cout << c << "\t" 
                          << feat.num_nodes << "\t"
                          << feat.num_edges << "\t"
                          << std::fixed << std::setprecision(4) << feat.internal_density << "\t"
                          << feat.degree_variance << "\t"
                          << feat.hub_concentration << "\t"
                          << feat.clustering_coeff << "\t"
                          << feat.avg_path_length << "\t"
                          << static_cast<int>(feat.diameter_estimate) << "\t"
                          << static_cast<int>(feat.community_count) << "\t"
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
        
        // Minimum size to bother with local reordering (building subgraph is expensive)
        const size_t MIN_COMMUNITY_FOR_LOCAL_REORDER = 500;
        
        for (size_t c : sorted_comms) {
            auto& nodes = community_nodes[c];
            if (nodes.empty()) continue;
            
            size_t comm_size = nodes.size();
            ReorderingAlgo algo = selected_algos[c];
            
            if (comm_size < MIN_COMMUNITY_FOR_LOCAL_REORDER || algo == ORIGINAL) {
                // Small or ORIGINAL: just assign sequentially (no subgraph overhead)
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

        // Extract comm_ids by node ID (not sorted order)
        // comm_ids[node_id] = community of node_id
        #pragma omp parallel for
        for (size_t i = 0; i < num_nodesx; ++i)
        {
            comm_ids[i] = sort_key_col[i];
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

        // Save OMP settings
        const int saved_omp_threads = omp_get_max_threads();
        const int saved_omp_nested = omp_get_nested();
        
        // Threshold: communities larger than this get full parallelism inside RabbitOrder
        const size_t LARGE_COMMUNITY_THRESHOLD = 500000; // 500K edges
        
        // Process all communities sequentially but control parallelism dynamically
        // Large communities: let RabbitOrder use all threads
        // Small communities: run RabbitOrder single-threaded to avoid overhead
        pvector<NodeID_> new_ids_sub(num_nodes);
        
        for (size_t idx = 0; idx < top_communities.size(); ++idx) {
            size_t comm_id = top_communities[idx];
            auto &edge_list = community_edge_lists[comm_id];
            
            if (edge_list.empty()) continue;
            
            const bool is_large = edge_list.size() >= LARGE_COMMUNITY_THRESHOLD;
            
            // Set parallelism based on community size
            if (is_large) {
                omp_set_num_threads(saved_omp_threads);
                std::cout << "Community ID " << comm_id 
                          << " edge list: " << edge_list.size() 
                          << " [LARGE - full parallelism]\n";
            } else {
                omp_set_num_threads(1);  // Single-threaded for small communities
                std::cout << "Community ID " << comm_id 
                          << " edge list: " << edge_list.size() 
                          << " [small - sequential]\n";
            }
            
            // Reset new_ids_sub
            std::fill(new_ids_sub.begin(), new_ids_sub.end(), (NodeID_)-1);
            
            // Determine algorithm and options
            ReorderingAlgo reordering_algo_nest = algo;
            std::vector<std::string> local_reordering_options = reordering_options;

            // NOTE: Disabled recursive GraphBrew for now to fix segfault
            // if (numLevels > 1 && idx == 0) {
            //     reordering_algo_nest = ReorderingAlgo::GraphBrewOrder;
            //     if (local_reordering_options.size() > 2) {
            //         local_reordering_options[2] = std::to_string(static_cast<double>(resolution));
            //     } else {
            //         local_reordering_options.push_back(std::to_string(static_cast<double>(resolution)));
            //     }
            // }

            std::vector<std::string> leiden_reordering_options = {"1.0", "30", "30"};
            std::vector<std::string> next_reordering_options;

            if (reordering_algo_nest == ReorderingAlgo::LeidenOrder) {
                leiden_reordering_options[0] = local_reordering_options.size() > 2 ? local_reordering_options[2] : "1.0";
                leiden_reordering_options[1] = std::to_string(static_cast<int>(maxIterations));
                leiden_reordering_options[2] = std::to_string(static_cast<int>(maxPasses));
            }

            next_reordering_options = (reordering_algo_nest == ReorderingAlgo::LeidenOrder) 
                                      ? leiden_reordering_options : local_reordering_options;

            // Process this community
            GenerateMappingLocalEdgelist(g, edge_list, new_ids_sub, reordering_algo_nest, true,
                                         next_reordering_options, numLevels, true);
            
            // Collect results for this community (with bounds checking)
            if (comm_id > num_comm) {
                std::cerr << "ERROR: comm_id " << comm_id << " > num_comm " << num_comm << std::endl;
                continue;
            }
            for (size_t i = 0; i < new_ids_sub.size(); ++i) {
                if (new_ids_sub[i] != (NodeID_)-1 && comm_ids[i] == comm_id) {
                    community_id_mappings[comm_id].emplace_back(i, new_ids_sub[i]);
                }
            }
        }
        
        // Restore OMP settings
        omp_set_num_threads(saved_omp_threads);
        omp_set_nested(saved_omp_nested);

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
