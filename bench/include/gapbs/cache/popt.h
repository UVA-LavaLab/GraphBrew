// ============================================================================
// P-OPT: Practical Optimal Cache Replacement for Graph Analytics
// ============================================================================
// This header implements cache optimization techniques from the P-OPT paper:
//
// @inproceedings{popt-hpca21,
//   title={P-OPT: Practical Optimal Cache Replacement for Graph Analytics},
//   author={Balaji, Vignesh and Crago, Neal and Jaleel, Aamer and Lucia, Brandon},
//   booktitle={2021 IEEE International Symposium on High-Performance Computer
//              Architecture (HPCA)},
//   pages={668--681},
//   year={2021},
//   organization={IEEE}
// }
//
// Key Functions:
//   - graphSlicer: CSR-segmenting for graph partitioning (Cagra paper)
//   - quantizeGraph: Quantize graph into tiles for cache analysis
//   - makeOffsetMatrix: Build compressed rereference matrix for cache simulation
//
// These functions are used for cache simulation and optimization experiments,
// not for the core GraphBrew reordering algorithms.
//
// Author: GraphBrew Team (extracted from builder.h)
// License: See LICENSE.txt
// ============================================================================

#ifndef POPT_CACHE_ALIAS_H_
#define POPT_CACHE_ALIAS_H_
// Alias shim: prefer gapbs/partitioning/popt.h
#if defined(__GNUC__)
#warning "Include <gapbs/partitioning/popt.h> instead of <gapbs/cache/popt.h>"
#elif defined(_MSC_VER)
#pragma message("Include <gapbs/partitioning/popt.h> instead of <gapbs/cache/popt.h>")
#endif
#include "../partitioning/popt.h"
#include "../timer.h"

// ============================================================================
// GRAPH SLICING (CSR-Segmenting from Cagra paper)
// ============================================================================

/**
 * @brief Slice a graph to create a partition within a vertex range
 * 
 * Creates a sub-graph containing only edges where the destination vertex
 * falls within [startID, stopID). Used for cache-aware graph partitioning.
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @tparam invert Whether graph has inverse edges
 * @param g Input graph
 * @param startID Start of vertex range (inclusive)
 * @param stopID End of vertex range (exclusive)
 * @param outDegree If true, filter out-edges; else filter in-edges
 * @param modifyBothDestlists If true, build both CSR and CSC
 * @return Sliced graph partition
 * 
 * @note Pull implementations should use outDegree=false
 * @note Push implementations should use outDegree=true
 */
template <typename NodeID_, typename DestID_, bool invert>
CSRGraph<NodeID_, DestID_, invert>
graphSlicer(const CSRGraph<NodeID_, DestID_, invert> &g, 
            NodeID_ startID, NodeID_ stopID, 
            bool outDegree = false,
            bool modifyBothDestlists = false)
{
    Timer t;
    t.Start();

    if (g.directed() == true)
    {
        // Step I: Calculate reduced degree per vertex for the partition
        pvector<NodeID_> degrees(g.num_nodes());
        
        #pragma omp parallel for schedule(dynamic, 1024)
        for (NodeID_ n = 0; n < g.num_nodes(); ++n)
        {
            NodeID_ newDegree = 0;
            if (outDegree)
            {
                for (NodeID_ m : g.out_neigh(n))
                {
                    if (m >= startID && m < stopID) ++newDegree;
                }
            }
            else
            {
                for (NodeID_ m : g.in_neigh(n))
                {
                    if (m >= startID && m < stopID) ++newDegree;
                }
            }
            degrees[n] = newDegree;
        }

        // Step II: Build trimmed CSR
        pvector<SGOffset> offsets(g.num_nodes() + 1);
        offsets[0] = 0;
        for (NodeID_ n = 0; n < g.num_nodes(); ++n)
        {
            offsets[n + 1] = offsets[n] + degrees[n];
        }
        
        DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
        DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
        
        #pragma omp parallel for schedule(dynamic, 1024)
        for (NodeID_ u = 0; u < g.num_nodes(); ++u)
        {
            SGOffset write_pos = offsets[u];
            if (outDegree)
            {
                for (NodeID_ v : g.out_neigh(u))
                {
                    if (v >= startID && v < stopID)
                    {
                        neighs[write_pos++] = v;
                    }
                }
            }
            else
            {
                for (NodeID_ v : g.in_neigh(u))
                {
                    if (v >= startID && v < stopID)
                    {
                        neighs[write_pos++] = v;
                    }
                }
            }
        }

        // Step III: Build inverse list if needed
        DestID_ *inv_neighs = nullptr;
        DestID_ **inv_index = nullptr;
        
        if (modifyBothDestlists)
        {
            pvector<NodeID_> inv_degrees(g.num_nodes());
            
            #pragma omp parallel for
            for (NodeID_ u = 0; u < g.num_nodes(); ++u)
            {
                inv_degrees[u] = outDegree ? g.in_degree(u) : g.out_degree(u);
            }
            
            pvector<SGOffset> inv_offsets(g.num_nodes() + 1);
            inv_offsets[0] = 0;
            for (NodeID_ n = 0; n < g.num_nodes(); ++n)
            {
                inv_offsets[n + 1] = inv_offsets[n] + inv_degrees[n];
            }
            
            inv_neighs = new DestID_[inv_offsets[g.num_nodes()]];
            inv_index = CSRGraph<NodeID_, DestID_>::GenIndex(inv_offsets, inv_neighs);
            
            #pragma omp parallel for schedule(dynamic, 1024)
            for (NodeID_ u = 0; u < g.num_nodes(); ++u)
            {
                SGOffset write_pos = inv_offsets[u];
                if (outDegree)
                {
                    for (NodeID_ v : g.in_neigh(u))
                    {
                        inv_neighs[write_pos++] = v;
                    }
                }
                else
                {
                    for (NodeID_ v : g.out_neigh(u))
                    {
                        inv_neighs[write_pos++] = v;
                    }
                }
            }
        }

        t.Stop();
        PrintTime("Slice-time", t.Seconds());
        
        if (outDegree)
        {
            return CSRGraph<NodeID_, DestID_, invert>(
                g.num_nodes(), index, neighs, inv_index, inv_neighs);
        }
        else
        {
            return CSRGraph<NodeID_, DestID_, invert>(
                g.num_nodes(), inv_index, inv_neighs, index, neighs);
        }
    }
    else
    {
        // Undirected graph - simpler case
        pvector<NodeID_> degrees(g.num_nodes());
        
        #pragma omp parallel for schedule(dynamic, 1024)
        for (NodeID_ n = 0; n < g.num_nodes(); ++n)
        {
            NodeID_ newDegree = 0;
            for (NodeID_ m : g.out_neigh(n))
            {
                if (m >= startID && m < stopID) ++newDegree;
            }
            degrees[n] = newDegree;
        }

        pvector<SGOffset> offsets(g.num_nodes() + 1);
        offsets[0] = 0;
        for (NodeID_ n = 0; n < g.num_nodes(); ++n)
        {
            offsets[n + 1] = offsets[n] + degrees[n];
        }
        
        DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
        DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
        
        #pragma omp parallel for schedule(dynamic, 1024)
        for (NodeID_ u = 0; u < g.num_nodes(); ++u)
        {
            SGOffset write_pos = offsets[u];
            for (NodeID_ v : g.out_neigh(u))
            {
                if (v >= startID && v < stopID)
                {
                    neighs[write_pos++] = v;
                }
            }
        }

        t.Stop();
        PrintTime("Slice-time", t.Seconds());
        return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
    }
}

// ============================================================================
// CAGRA PARTITIONING (CSR-segmenting based partitioning)
// ============================================================================

/**
 * @brief Partition a graph using Cagra/GraphIT style CSR-segmenting
 * 
 * Divides graph into p_n x p_m partitions using graphSlicer.
 * Each partition contains edges to vertices in a contiguous ID range.
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @tparam invert Whether graph has inverse edges
 * @param g Input graph
 * @param p_n Number of row partitions
 * @param p_m Number of column partitions
 * @param outDegree If true, partition by out-edges; else by in-edges
 * @return Vector of partitioned graphs (p_n x p_m partitions)
 */
template <typename NodeID_, typename DestID_, bool invert>
std::vector<CSRGraph<NodeID_, DestID_, invert>>
MakeCagraPartitionedGraph(const CSRGraph<NodeID_, DestID_, invert> &g,
                          int p_n = 1, int p_m = 1, bool outDegree = false)
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
            graphSlicer<NodeID_, DestID_, invert>(g, startID, stopID, outDegree);
        partitions.emplace_back(std::move(partition));
        startID = stopID;
    }

    return partitions;
}

// ============================================================================
// GRAPH QUANTIZATION (For cache analysis)
// ============================================================================

/**
 * @brief Quantize graph neighbors into tiles for cache analysis
 * 
 * Creates a graph where neighbors are replaced by their tile IDs.
 * Used for analyzing cache access patterns at a coarser granularity.
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @tparam invert Whether graph has inverse edges
 * @param g Input graph
 * @param numTiles Number of tiles to divide vertices into
 * @return Quantized graph where neighbors are tile IDs
 */
template <typename NodeID_, typename DestID_, bool invert>
CSRGraph<NodeID_, DestID_, invert>
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

    pvector<SGOffset> offsets(g.num_nodes() + 1);
    offsets[0] = 0;
    for (NodeID_ n = 0; n < g.num_nodes(); ++n)
    {
        offsets[n + 1] = offsets[n] + degrees[n];
    }
    
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

        SGOffset write_pos = offsets[u];
        for (NodeID_ tile : uniqNghs)
        {
            neighs[write_pos++] = tile;
        }
    }
    
    return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
}

// ============================================================================
// REREFERENCE MATRIX (For P-OPT cache simulation)
// ============================================================================

/**
 * @brief Build compressed rereference matrix for cache simulation
 * 
 * Creates a matrix tracking when each cache line will next be accessed.
 * Used by P-OPT cache replacement policy to make optimal decisions.
 * 
 * The rereference matrix is compressed using:
 *   - MSB: 1 = rereference value, 0 = switch point
 *   - Lower 7 bits: intra-epoch information
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @tparam invert Whether graph has inverse edges
 * @param g Input graph
 * @param offsetMatrix Output compressed rereference matrix
 * @param numVtxPerLine Vertices per cache line
 * @param numEpochs Number of epochs (must be 256)
 * @param traverseCSR If true, traverse CSR (out-edges); else CSC (in-edges)
 */
template <typename NodeID_, typename DestID_, bool invert>
void makeOffsetMatrix(const CSRGraph<NodeID_, DestID_, invert> &g,
                      pvector<uint8_t> &offsetMatrix,
                      int numVtxPerLine, int numEpochs,
                      bool traverseCSR = true)
{
    if (g.directed() == false)
        traverseCSR = true;

    Timer tm;

    // Step I: Collect quantized edges & compact vertices into "super vertices"
    tm.Start();
    NodeID_ numCacheLines = (g.num_nodes() + numVtxPerLine - 1) / numVtxPerLine;
    NodeID_ epochSz = (g.num_nodes() + numEpochs - 1) / numEpochs;
    pvector<NodeID_> lastRef(numCacheLines * numEpochs, -1);
    NodeID_ chunkSz = 64 / numVtxPerLine;
    if (chunkSz == 0) chunkSz = 1;

    #pragma omp parallel for schedule(dynamic, chunkSz)
    for (NodeID_ c = 0; c < numCacheLines; ++c)
    {
        NodeID_ startVtx = c * numVtxPerLine;
        NodeID_ endVtx = (c + 1) * numVtxPerLine;
        if (c == numCacheLines - 1)
            endVtx = g.num_nodes();

        for (NodeID_ v = startVtx; v < endVtx; ++v)
        {
            if (traverseCSR)
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
    std::cout << "[P-OPT] Time to quantize neighbors and compact vertices = "
              << tm.Seconds() << std::endl;
    
    assert(numEpochs == 256);

    // Step II: Convert adjacency matrix into offsets
    tm.Start();
    uint8_t maxReref = 127;  // MSB reserved for type identification
    NodeID_ subEpochSz = (epochSz + 127) / 128;  // 7 bits for intra-epoch info
    pvector<uint8_t> compressedOffsets(numCacheLines * numEpochs);
    uint8_t mask = 1;
    mask <<= 7;

    #pragma omp parallel for schedule(dynamic, chunkSz)
    for (NodeID_ c = 0; c < numCacheLines; ++c)
    {
        for (int e = 0; e < numEpochs; ++e)
        {
            int idx = (c * numEpochs) + e;
            NodeID_ lastRefVal = lastRef[idx];
            
            if (lastRefVal != static_cast<NodeID_>(-1))
            {
                // Calculate intra-epoch position
                NodeID_ subPos = (lastRefVal % epochSz) / subEpochSz;
                compressedOffsets[idx] = static_cast<uint8_t>(subPos) | mask;
            }
            else
            {
                compressedOffsets[idx] = 0;  // No reference in this epoch
            }
        }
    }
    tm.Stop();
    std::cout << "[P-OPT] Time to convert to offsets matrix = "
              << tm.Seconds() << std::endl;

    // Step III: Transpose for cache-friendly access
    tm.Start();
    offsetMatrix.resize(numCacheLines * numEpochs);
    
    #pragma omp parallel for schedule(static)
    for (NodeID_ c = 0; c < numCacheLines; ++c)
    {
        for (int e = 0; e < numEpochs; ++e)
        {
            offsetMatrix[(e * numCacheLines) + c] = 
                compressedOffsets[(c * numEpochs) + e];
        }
    }
    tm.Stop();
    std::cout << "[P-OPT] Time to transpose offsets matrix = "
              << tm.Seconds() << std::endl;
}

// ============================================================================
// DEGREE SORTING (For cache optimization experiments)
// ============================================================================

/**
 * @brief Sort vertices by degree for cache optimization
 * 
 * Creates a new graph with vertices sorted by degree (descending).
 * Used for cache locality experiments.
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @tparam invert Whether graph has inverse edges
 * @param g Input graph
 * @param outDegree If true, sort by out-degree; else by in-degree
 * @param new_ids Output mapping from old to new IDs
 * @param createOnlyDegList If true, only create degree list
 * @param createBothCSRs If true, create both CSR and CSC
 * @return Reordered graph
 */
template <typename NodeID_, typename DestID_, bool invert>
CSRGraph<NodeID_, DestID_, invert>
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
        // Step I: Create a list of degrees
        #pragma omp parallel for
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
        {
            if (outDegree == true)
                degree_id_pairs[n] = std::make_pair(g.out_degree(n), n);
            else
                degree_id_pairs[n] = std::make_pair(g.in_degree(n), n);
        }

        // Step II: Sort based on degree order
        __gnu_parallel::sort(degree_id_pairs.begin(), degree_id_pairs.end(),
                             std::greater<degree_node_p>());

        // Step III: Assign remap for the hub vertices
        #pragma omp parallel for
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
        {
            new_ids[degree_id_pairs[n].second] = n;
        }

        // Step IV: Generate degree to build a new graph
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

        // Graph building phase
        pvector<SGOffset> offsets(g.num_nodes() + 1);
        offsets[0] = 0;
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
            offsets[n + 1] = offsets[n] + inv_degrees[n];
            
        DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
        DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
        
        #pragma omp parallel for schedule(dynamic, 1024)
        for (NodeID_ u = 0; u < g.num_nodes(); u++)
        {
            SGOffset pos = offsets[new_ids[u]];
            if (outDegree == true)
            {
                for (NodeID_ v : g.in_neigh(u))
                    neighs[pos++] = new_ids[v];
            }
            else
            {
                for (NodeID_ v : g.out_neigh(u))
                    neighs[pos++] = new_ids[v];
            }
            std::sort(index[new_ids[u]], index[new_ids[u] + 1]);
        }
        
        DestID_ *inv_neighs = nullptr;
        DestID_ **inv_index = nullptr;
        if (createOnlyDegList == true || createBothCSRs == true)
        {
            pvector<SGOffset> inv_offsets(g.num_nodes() + 1);
            inv_offsets[0] = 0;
            for (NodeID_ n = 0; n < g.num_nodes(); n++)
                inv_offsets[n + 1] = inv_offsets[n] + degrees[n];
                
            inv_neighs = new DestID_[inv_offsets[g.num_nodes()]];
            inv_index = CSRGraph<NodeID_, DestID_>::GenIndex(inv_offsets, inv_neighs);
            
            if (createBothCSRs == true)
            {
                #pragma omp parallel for schedule(dynamic, 1024)
                for (NodeID_ u = 0; u < g.num_nodes(); u++)
                {
                    SGOffset pos = inv_offsets[new_ids[u]];
                    if (outDegree == true)
                    {
                        for (NodeID_ v : g.out_neigh(u))
                            inv_neighs[pos++] = new_ids[v];
                    }
                    else
                    {
                        for (NodeID_ v : g.in_neigh(u))
                            inv_neighs[pos++] = new_ids[v];
                    }
                    std::sort(inv_index[new_ids[u]], inv_index[new_ids[u] + 1]);
                }
            }
        }
        
        t.Stop();
        PrintTime("DegSort Time", t.Seconds());
        
        if (outDegree == true)
            return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), inv_index,
                    inv_neighs, index, neighs);
        else
            return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs,
                    inv_index, inv_neighs);
    }
    else
    {
        // Undirected graphs
        #pragma omp parallel for
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
        {
            degree_id_pairs[n] = std::make_pair(g.out_degree(n), n);
        }

        __gnu_parallel::sort(degree_id_pairs.begin(), degree_id_pairs.end(),
                             std::greater<degree_node_p>());

        #pragma omp parallel for
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
        {
            new_ids[degree_id_pairs[n].second] = n;
        }

        pvector<NodeID_> degrees(g.num_nodes());
        #pragma omp parallel for
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
        {
            degrees[new_ids[n]] = g.out_degree(n);
        }

        pvector<SGOffset> offsets(g.num_nodes() + 1);
        offsets[0] = 0;
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
            offsets[n + 1] = offsets[n] + degrees[n];
            
        DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
        DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
        
        #pragma omp parallel for schedule(dynamic, 1024)
        for (NodeID_ u = 0; u < g.num_nodes(); u++)
        {
            SGOffset pos = offsets[new_ids[u]];
            for (NodeID_ v : g.out_neigh(u))
                neighs[pos++] = new_ids[v];
            std::sort(index[new_ids[u]], index[new_ids[u] + 1]);
        }
        
        t.Stop();
        PrintTime("DegSort Time", t.Seconds());
        return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
    }
}

// ============================================================================
// RANDOM ORDERING (For baseline comparison)
// ============================================================================

/**
 * @brief Create random vertex ordering
 * 
 * Creates a graph with randomly permuted vertex IDs.
 * Used as a baseline for cache optimization experiments.
 * 
 * @tparam NodeID_ Node ID type
 * @tparam DestID_ Destination ID type
 * @tparam invert Whether graph has inverse edges
 * @param g Input graph
 * @param new_ids Output mapping from old to new IDs
 * @param createOnlyDegList If true, only create degree list
 * @param createBothCSRs If true, create both CSR and CSC
 * @return Randomly reordered graph
 */
template <typename NodeID_, typename DestID_, bool invert>
CSRGraph<NodeID_, DestID_, invert>
RandOrder(const CSRGraph<NodeID_, DestID_, invert> &g,
          pvector<NodeID_> &new_ids, bool createOnlyDegList,
          bool createBothCSRs)
{
    Timer t;
    t.Start();
    std::srand(0);  // Fixed seed for reproducibility
    bool outDegree = true;

    if (g.directed() == true)
    {
        // Create a random permutation
        pvector<NodeID_> claimedVtxs(g.num_nodes(), 0);

        for (NodeID_ v = 0; v < g.num_nodes(); ++v)
        {
            while (true)
            {
                NodeID_ randID = std::rand() % g.num_nodes();
                if (claimedVtxs[randID] != 1)
                {
                    if (__sync_bool_compare_and_swap(&claimedVtxs[randID], 0, 1))
                    {
                        new_ids[v] = randID;
                        break;
                    }
                }
            }
        }

        #pragma omp parallel for
        for (NodeID_ v = 0; v < g.num_nodes(); ++v)
            assert(new_ids[v] != -1);

        // Generate degree lists
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

        // Graph building phase
        pvector<SGOffset> offsets(g.num_nodes() + 1);
        offsets[0] = 0;
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
            offsets[n + 1] = offsets[n] + inv_degrees[n];
            
        DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
        DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
        
        #pragma omp parallel for schedule(dynamic, 1024)
        for (NodeID_ u = 0; u < g.num_nodes(); u++)
        {
            SGOffset pos = offsets[new_ids[u]];
            if (outDegree == true)
            {
                for (NodeID_ v : g.in_neigh(u))
                    neighs[pos++] = new_ids[v];
            }
            else
            {
                for (NodeID_ v : g.out_neigh(u))
                    neighs[pos++] = new_ids[v];
            }
            std::sort(index[new_ids[u]], index[new_ids[u] + 1]);
        }
        
        DestID_ *inv_neighs = nullptr;
        DestID_ **inv_index = nullptr;
        if (createOnlyDegList == true || createBothCSRs == true)
        {
            pvector<SGOffset> inv_offsets(g.num_nodes() + 1);
            inv_offsets[0] = 0;
            for (NodeID_ n = 0; n < g.num_nodes(); n++)
                inv_offsets[n + 1] = inv_offsets[n] + degrees[n];
                
            inv_neighs = new DestID_[inv_offsets[g.num_nodes()]];
            inv_index = CSRGraph<NodeID_, DestID_>::GenIndex(inv_offsets, inv_neighs);
            
            if (createBothCSRs == true)
            {
                #pragma omp parallel for schedule(dynamic, 1024)
                for (NodeID_ u = 0; u < g.num_nodes(); u++)
                {
                    SGOffset pos = inv_offsets[new_ids[u]];
                    if (outDegree == true)
                    {
                        for (NodeID_ v : g.out_neigh(u))
                            inv_neighs[pos++] = new_ids[v];
                    }
                    else
                    {
                        for (NodeID_ v : g.in_neigh(u))
                            inv_neighs[pos++] = new_ids[v];
                    }
                    std::sort(inv_index[new_ids[u]], inv_index[new_ids[u] + 1]);
                }
            }
        }
        
        t.Stop();
        PrintTime("RandOrder Time", t.Seconds());
        
        if (outDegree == true)
            return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), inv_index,
                    inv_neighs, index, neighs);
        else
            return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs,
                    inv_index, inv_neighs);
    }
    else
    {
        // Undirected graphs
        pvector<NodeID_> claimedVtxs(g.num_nodes(), 0);

        for (NodeID_ v = 0; v < g.num_nodes(); ++v)
        {
            while (true)
            {
                NodeID_ randID = std::rand() % g.num_nodes();
                if (claimedVtxs[randID] != 1)
                {
                    if (__sync_bool_compare_and_swap(&claimedVtxs[randID], 0, 1))
                    {
                        new_ids[v] = randID;
                        break;
                    }
                }
            }
        }

        #pragma omp parallel for
        for (NodeID_ v = 0; v < g.num_nodes(); ++v)
            assert(new_ids[v] != -1);

        pvector<NodeID_> degrees(g.num_nodes());
        #pragma omp parallel for
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
        {
            degrees[new_ids[n]] = g.out_degree(n);
        }

        pvector<SGOffset> offsets(g.num_nodes() + 1);
        offsets[0] = 0;
        for (NodeID_ n = 0; n < g.num_nodes(); n++)
            offsets[n + 1] = offsets[n] + degrees[n];
            
        DestID_ *neighs = new DestID_[offsets[g.num_nodes()]];
        DestID_ **index = CSRGraph<NodeID_, DestID_>::GenIndex(offsets, neighs);
        
        #pragma omp parallel for schedule(dynamic, 1024)
        for (NodeID_ u = 0; u < g.num_nodes(); u++)
        {
            SGOffset pos = offsets[new_ids[u]];
            for (NodeID_ v : g.out_neigh(u))
                neighs[pos++] = new_ids[v];
            std::sort(index[new_ids[u]], index[new_ids[u] + 1]);
        }
        
        t.Stop();
        PrintTime("RandOrder Time", t.Seconds());
        return CSRGraph<NodeID_, DestID_, invert>(g.num_nodes(), index, neighs);
    }
}

#endif  // POPT_CACHE_ALIAS_H_
