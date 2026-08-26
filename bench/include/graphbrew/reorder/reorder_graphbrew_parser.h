#ifndef GRAPHBREW_REORDER_GRAPHBREW_PARSER_H_
#define GRAPHBREW_REORDER_GRAPHBREW_PARSER_H_

// Internal GraphBrew detail header. Include only from reorder_graphbrew.h.

// SECTION 23: CONFIGURATION PARSING
//=============================================================================

/**
 * Parse GraphBrew option tokens after builder-level preset expansion.
 * Unknown tokens fail closed when strict parsing is requested.
 */
inline GraphBrewConfig parseGraphBrewConfig(
    const std::vector<std::string>& options,
    bool strict = false,
    bool validateComplete = true) {
    GraphBrewConfig config;
    auto reject = [&](const std::string& opt) {
        if (strict) {
            throw std::invalid_argument(
                "Unknown or invalid GraphBrew option: " + opt);
        }
    };
    auto parseExactInt = [](const std::string& text, int& value) {
        try {
            size_t parsed = 0;
            value = std::stoi(text, &parsed);
            return parsed == text.size();
        } catch (...) {
            return false;
        }
    };
    auto parseExactDouble = [](const std::string& text, double& value) {
        try {
            size_t parsed = 0;
            value = std::stod(text, &parsed);
            return parsed == text.size() && std::isfinite(value);
        } catch (...) {
            return false;
        }
    };
    for (size_t i = 0; i < options.size(); ++i) {
        const std::string& opt = options[i];
        if (opt.empty()) continue;

        if (opt == "cd_parallel" || opt == "cd:parallel" ||
            opt == "community_parallel") {
            config.deterministicCommunityDetection = false;
            continue;
        }
        if (opt == "cd_serial" || opt == "cd:serial" ||
            opt == "community_serial") {
            config.deterministicCommunityDetection = true;
            continue;
        }
        if (
            opt.size() > 4
            && opt.substr(0, 4) == "sgmb"
        ) {
            int batch = 0;
            if (
                !parseExactInt(opt.substr(4), batch)
                || batch <= 0
            ) {
                reject(opt);
                continue;
            }
            config.superGraphMoveBatch = batch;
            continue;
        }
        // Check for main algorithm selection (RabbitOrder from paper)
        if (opt == "rabbit" || opt == "rabbitorder" ||
            opt == "cd_rabbit" || opt == "cd:rabbit" || opt == "cdrabbit") {
            config.algorithm = GraphBrewAlgorithm::RABBIT_ORDER;
            continue;
        }
        if (opt == "cd_leiden" || opt == "cd:leiden" || opt == "cdleiden") {
            config.algorithm = GraphBrewAlgorithm::LEIDEN;
            config.communityMode = CommunityMode::FULL_LEIDEN;
            config.useRefinement = true;
            config.refinementDepth = -1;
            continue;
        }

        // Check for ordering strategy
        if (opt == "hierarchical" || opt == "hier") {
            config.ordering = OrderingStrategy::HIERARCHICAL;
            config.hasExplicitOrdering = true;
        } else if (opt == "connectivity" || opt == "conn" || opt == "connbfs") {
            config.ordering = OrderingStrategy::CONNECTIVITY_BFS;
            config.hasExplicitOrdering = true;
        } else if (opt == "dfs") {
            config.ordering = OrderingStrategy::DENDROGRAM_DFS;
            config.hasExplicitOrdering = true;
        } else if (opt == "bfs") {
            config.ordering = OrderingStrategy::DENDROGRAM_BFS;
            config.hasExplicitOrdering = true;
        } else if (opt == "community" || opt == "comm") {
            config.ordering = OrderingStrategy::COMMUNITY_SORT;
            config.hasExplicitOrdering = true;
        } else if (opt == "hubcluster" || opt == "hub") {
            config.ordering = OrderingStrategy::HUB_CLUSTER;
            config.hasExplicitOrdering = true;
        } else if (opt == "dbg") {
            config.ordering = OrderingStrategy::DBG;
            config.hasExplicitOrdering = true;
        } else if (opt == "corder") {
            config.ordering = OrderingStrategy::CORDER;
            config.hasExplicitOrdering = true;
        } else if (opt == "dbg-global" || opt == "dbgglobal") {
            config.ordering = OrderingStrategy::DBG_GLOBAL;
            config.hasExplicitOrdering = true;
        } else if (opt == "corder-global" || opt == "corderglobal") {
            config.ordering = OrderingStrategy::CORDER_GLOBAL;
            config.hasExplicitOrdering = true;
        } else if (opt == "hcache" || opt == "hiercache" || opt == "hierarchical-cache") {
            config.ordering = OrderingStrategy::HIERARCHICAL_CACHE_AWARE;
            config.hasExplicitOrdering = true;
        } else if (opt == "hrab" || opt == "hybrid-rabbit" || opt == "leidenrabbit") {
            config.ordering = OrderingStrategy::HYBRID_LEIDEN_RABBIT;
            config.hasExplicitOrdering = true;
            // HRAB defaults to RCM intra-community ordering: empirically
            // measured 30-50% memory-access reduction vs the BFS default,
            // at zero reorder-time cost.  Verified on PR with L3=1MB on
            // soc-pokec, cit-Patents, hollywood-2009, com-Orkut.
            // Use ":bfs_intra" to opt out (kept for the BFS-baseline ablation).
            config.useRCMIntra = true;
        } else if (opt == "tqr" || opt == "tile-quantized" || opt == "tilequantized" || opt == "tilerabbit") {
            config.ordering = OrderingStrategy::TILE_QUANTIZED_RABBIT;
            config.hasExplicitOrdering = true;
        } else if (opt == "hlr" || opt == "hier-rabbit" || opt == "hierarchical-rabbit" || opt == "hierleidenrabbit") {
            // HLR: run RabbitOrder at every Leiden dendrogram level
            // (multi-level super-graph ordering -- generalises HRAB which
            // only uses the finest level).
            config.ordering = OrderingStrategy::HIERARCHICAL_LEIDEN_RABBIT;
            config.hasExplicitOrdering = true;
        }
        // COMPOSE strategy + Stage 2/Stage 3 picks (SECTION 16-COMPOSE)
        else if (opt == "compose" || opt == "pluggable") {
            config.ordering = OrderingStrategy::COMPOSE;
            config.hasExplicitOrdering = true;
        } else if (opt == "s2_size" || opt == "s2:size" || opt == "s2size" ||
                   opt == "comm_size" || opt == "comm:size" || opt == "commsize" ||
                   opt == "comm_size_desc" || opt == "comm:size_desc" || opt == "commsizedesc") {
            config.communityOrder = CommunityOrder::SizeDesc;
        } else if (opt == "s2_size_asc" || opt == "s2:size_asc" || opt == "s2sizeasc" ||
                   opt == "comm_size_asc" || opt == "comm:size_asc" || opt == "commsizeasc") {
            config.communityOrder = CommunityOrder::SizeAsc;
        } else if (opt == "s2_degree" || opt == "s2:degree" || opt == "s2degree" ||
                   opt == "comm_degree" || opt == "comm:degree" || opt == "commdegree" ||
                   opt == "comm_degree_desc" || opt == "comm:degree_desc" || opt == "commdegreedesc") {
            config.communityOrder = CommunityOrder::DegreeDesc;
        } else if (opt == "s2_degree_asc" || opt == "s2:degree_asc" || opt == "s2degreeasc" ||
                   opt == "comm_degree_asc" || opt == "comm:degree_asc" || opt == "commdegreeasc") {
            config.communityOrder = CommunityOrder::DegreeAsc;
        } else if (opt == "s2_cut_min" || opt == "s2:cut_min" || opt == "s2cutmin" ||
                   opt == "comm_cut_min" || opt == "comm:cut_min" || opt == "commcutmin" ||
                   opt == "cut_min" || opt == "cutmin") {
            // Mt-METIS-style: greedy NN-TSP over inter-community crossing edges.
            // Cost O(|E|+C^2).  Falls back to DegreeDesc if C>4096.
            config.communityOrder = CommunityOrder::CutMin;
        } else if (opt == "s3_bfs" || opt == "s3:bfs" || opt == "s3bfs" ||
                   opt == "intra_bfs" || opt == "intra:bfs" || opt == "intrabfs") {
            config.intraCommunityOrder = IntraCommunityOrder::BFSFromHub;
        } else if (
            opt == "s3_bfs_direct"
            || opt == "s3:bfs_direct"
            || opt == "s3bfsdirect"
            || opt == "intra_bfs_direct"
            || opt == "intra:bfs_direct"
            || opt == "intrabfsdirect"
            || opt == "bfs_direct"
        ) {
            config.intraCommunityOrder = IntraCommunityOrder::BFSDirect;
        } else if (
            opt == "intra_bfs_compact"
            || opt == "intra:bfs_compact"
            || opt == "intrabfscompact"
            || opt == "bfs_compact"
        ) {
            config.intraCommunityOrder = IntraCommunityOrder::BFSCompact;
        } else if (
            opt == "intra_bfs_compact_direct"
            || opt == "intra:bfs_compact_direct"
            || opt == "intrabfscompactdirect"
            || opt == "bfs_compact_direct"
        ) {
            config.intraCommunityOrder =
                IntraCommunityOrder::BFSCompactDirect;
        } else if (opt == "s3_rcm" || opt == "s3:rcm" || opt == "s3rcm" ||
                   opt == "intra_rcm" || opt == "intra:rcm" || opt == "intrarcm") {
            config.intraCommunityOrder = IntraCommunityOrder::RCM;
        } else if (opt == "s3_rcmpp" || opt == "s3:rcmpp" || opt == "s3rcmpp" ||
                   opt == "intra_rcmpp" || opt == "intra:rcmpp" || opt == "intrarcmpp" ||
                   opt == "rcmpp" || opt == "rcm++") {
            config.intraCommunityOrder = IntraCommunityOrder::RCMpp;
        } else if (opt == "s3_dendrogram" || opt == "s3:dendrogram" || opt == "s3dendrogram" ||
                   opt == "intra_dendrogram" || opt == "intra:dendrogram" || opt == "intradendrogram" ||
                   opt == "s3_dend" || opt == "intra_dend") {
            // Only valid with CD=Rabbit (algorithm=RABBIT_ORDER).  When
            // selected with Leiden, orderCompose<>() falls back to BFS with
            // a printed warning (see SECTION 16-COMPOSE).
            config.intraCommunityOrder = IntraCommunityOrder::Dendrogram;
        } else if (opt == "s3_gorder" || opt == "s3:gorder" || opt == "s3gorder" ||
                   opt == "intra_gorder" || opt == "intra:gorder" || opt == "intragorder" ||
                   opt == "s3_gord" || opt == "intra_gord") {
            // Per-community Gorder-greedy (UnitHeap).  Works with any CD
            // (no Rabbit dendrogram dependency).  Window size honors
            // config.gorderWindow (default 5; tune via gw<n>).
            config.intraCommunityOrder = IntraCommunityOrder::Gorder;
        } else if (opt == "s3_hubsort" || opt == "s3:hubsort" || opt == "s3hubsort" ||
                   opt == "intra_hubsort" || opt == "intra:hubsort" || opt == "intrahubsort" ||
                   opt == "intra_hub" || opt == "intra:hub") {
            // Per-community degree-descending sort.  Cheapest non-trivial
            // intra primitive; no graph traversal.
            config.intraCommunityOrder = IntraCommunityOrder::HubSort;
        } else if (opt == "s3_deg_asc" || opt == "s3:deg_asc" || opt == "s3degasc" ||
                   opt == "intra_deg_asc" || opt == "intra:deg_asc" || opt == "intra_degasc" ||
                   opt == "intra_degree_asc" || opt == "deg_asc" || opt == "degasc") {
            // Per-community degree-ascending sort.  Inverse of HubSort;
            // ablation control.  Same O(sz log sz) cost.
            config.intraCommunityOrder = IntraCommunityOrder::DegreeAsc;
        } else if (opt == "s3_hub2" || opt == "s3:hub2" || opt == "s3hub2" ||
                   opt == "intra_hub2" || opt == "intra:hub2" || opt == "intrahub2" ||
                   opt == "hub2") {
            // Second-moment degree (DRO/Lakhotia IISWC'19): per-community sort
            // by sum-of-neighbor-degree descending.  Cost O(|E_local|) per community.
            config.intraCommunityOrder = IntraCommunityOrder::Hub2;
        } else if (opt == "s3_alternate" || opt == "s3:alternate" || opt == "s3alt" ||
                   opt == "intra_alternate" || opt == "intra:alternate" || opt == "intra_alt" ||
                   opt == "alternate" || opt == "alt") {
            // Per-community hub/leaf interleave: front-back zip after sort-by-degree-desc.
            // Combines prefetch locality (hubs early) with false-sharing dispersion
            // (hubs across the ID range).
            config.intraCommunityOrder = IntraCommunityOrder::Alternate;
        } else if (opt == "s3_random" || opt == "s3:random" || opt == "s3rand" ||
                   opt == "intra_random" || opt == "intra:random" || opt == "intra_rand" ||
                   opt == "random" || opt == "rand") {
            // Per-community deterministic Fisher-Yates shuffle.
            // Worst-case control / sanity check; measures the value
            // of "any ordering at all" vs the chosen primitive.
            config.intraCommunityOrder = IntraCommunityOrder::Random;
        } else if (opt == "s3_boundary_last" || opt == "s3:boundary_last" || opt == "s3bndlast" ||
                   opt == "intra_boundary_last" || opt == "intra:boundary_last" || opt == "intra_bndlast" ||
                   opt == "boundary_last" || opt == "bndlast" || opt == "boundarylast") {
            // Per-community structure-aware sort: external degree ascending
            // (interior nodes first, boundary nodes last); ties by degree desc.
            // Tests "boundary-last" theory for BFS/CC locality.
            config.intraCommunityOrder = IntraCommunityOrder::BoundaryLast;
        } else if (opt == "s3_core" || opt == "s3:core" || opt == "s3core" ||
                   opt == "intra_core" || opt == "intra:core" || opt == "intracore" ||
                   opt == "core_order" || opt == "coreorder" || opt == "core") {
            // Per-community k-core decomposition; sort by core-number descending
            // (deep core nodes first, periphery last); ties by degree desc.
            // Tests "deep-core-first" theory: dense reuse target gets best cache lines.
            config.intraCommunityOrder = IntraCommunityOrder::CoreOrder;
        } else if (opt == "s1_none" || opt == "s1:none" || opt == "s1none" ||
                   opt == "sg_none" || opt == "sg:none" || opt == "sgnone") {
            config.superGraphOrder = SuperGraphOrder::None;
        } else if (opt == "s1_super_rabbit" || opt == "s1:super_rabbit" || opt == "s1srabbit" ||
                   opt == "sg_super_rabbit" || opt == "sg:super_rabbit" || opt == "sgsrabbit") {
            config.superGraphOrder = SuperGraphOrder::SuperRabbit;
        } else if (opt == "s1_super_rcm" || opt == "s1:super_rcm" || opt == "s1srcm" ||
                   opt == "sg_super_rcm" || opt == "sg:super_rcm" || opt == "sgsrcm") {
            config.superGraphOrder = SuperGraphOrder::SuperRCM;
        } else if (opt == "s1_tile_rabbit" || opt == "s1:tile_rabbit" || opt == "s1tilerabbit" ||
                   opt == "sg_tile_rabbit" || opt == "sg:tile_rabbit" || opt == "sgtilerabbit") {
            config.superGraphOrder = SuperGraphOrder::TileRabbit;
        } else if (opt == "s1_hilbert" || opt == "s1:hilbert" || opt == "s1hilbert" ||
                   opt == "sg_hilbert" || opt == "sg:hilbert" || opt == "sghilbert" ||
                   opt == "hilbert") {
            // Mosaic-style: 2-D Hilbert curve over (community size, avg degree).
            // Generates a super-graph permutation without building the super-graph.
            config.superGraphOrder = SuperGraphOrder::Hilbert;
        } else if (opt == "s2_identity" || opt == "s2:identity" || opt == "s2identity" ||
                   opt == "comm_identity" || opt == "comm:identity" || opt == "commidentity") {
            config.communityOrder = CommunityOrder::Identity;
        } else if (opt == "refine_2swap" || opt == "refine:2swap" || opt == "refine2swap" ||
                   opt == "r2swap" || opt == "twoswap" || opt == "2swap") {
            // FM-style adjacent-pair swap refinement (per-community).
            // See refineTwoSwap<>() in SECTION 16-PRIMITIVES.  Cost
            // O(|E_local| * refineMaxPasses) per community; trivially
            // parallel.  Use rmaxN to tune max passes (default 3).
            config.refinementPass = RefinementPass::TwoSwap;
        } else if (opt == "refine_none" || opt == "refine:none" || opt == "refinenone") {
            config.refinementPass = RefinementPass::None;
        }
        // GraphBrew mode: per-community external algorithm dispatch
        // "graphbrew" or "gb" activates LAYER ordering (default final algo = RabbitOrder 8)
        // "final:N" or "finalN" sets the final algo ID (0-11)
        else if (opt == "graphbrew" || opt == "gb") {
            config.ordering = OrderingStrategy::LAYER;
            config.hasExplicitOrdering = true;
            config.useSmallCommunityMerging = true;
            if (config.finalAlgoId < 0) config.finalAlgoId = 8;  // Default: RabbitOrder
        }
        // Final algorithm for GraphBrew: "final:8" or "final8" or just the algo number when in graphbrew mode
        else if (opt.size() > 5 && opt.substr(0, 5) == "final") {
            std::string numStr = opt.substr(5);
            if (!numStr.empty() && numStr[0] == ':') numStr = numStr.substr(1);
            int algoId = -1;
            if (!parseExactInt(numStr, algoId) ||
                algoId < 0 || algoId > 11) {
                reject(opt);
                continue;
            }
            config.finalAlgoId = algoId;
            config.ordering = OrderingStrategy::LAYER;
            config.useSmallCommunityMerging = true;
        }
        // Recursive depth for GraphBrew: "depth:2" or "depth2" or "recursive" or "flat"
        else if (opt == "flat" || opt == "norecurse") {
            config.recursiveDepth = 0;  // force flat (no recursive sub-division)
            config.ordering = OrderingStrategy::LAYER;
            config.useSmallCommunityMerging = true;
            if (config.finalAlgoId < 0) config.finalAlgoId = 8;
        }
        else if (opt == "recursive" || opt == "recurse") {
            config.recursiveDepth = std::max(config.recursiveDepth, 1);
            config.ordering = OrderingStrategy::LAYER;
            config.useSmallCommunityMerging = true;
            if (config.finalAlgoId < 0) config.finalAlgoId = 8;
        } else if (opt.size() > 5 && opt.substr(0, 5) == "depth") {
            std::string numStr = opt.substr(5);
            if (!numStr.empty() && numStr[0] == ':') numStr = numStr.substr(1);
            int depth = -1;
            if (!parseExactInt(numStr, depth) ||
                depth < 0 || depth > 10) {
                reject(opt);
                continue;
            }
            config.recursiveDepth = depth;
            config.ordering = OrderingStrategy::LAYER;
            config.useSmallCommunityMerging = true;
            if (config.finalAlgoId < 0) config.finalAlgoId = 8;
        }
        // Sub-community algorithm for recursive GraphBrew: "sub:auto" or "sub:3" or "subauto"
        else if (opt == "subauto" || opt == "sub:auto") {
            config.subAlgoId = -1;  // adaptive per-sub-community selection
        } else if (opt.size() > 3 && opt.substr(0, 3) == "sub") {
            std::string numStr = opt.substr(3);
            if (!numStr.empty() && numStr[0] == ':') numStr = numStr.substr(1);
            if (numStr == "auto") {
                config.subAlgoId = -1;
            } else {
                int algorithm = -1;
                if (!parseExactInt(numStr, algorithm) ||
                    algorithm < 0 || algorithm > 11) {
                    reject(opt);
                    continue;
                }
                config.subAlgoId = algorithm;
            }
        }
        // Check for aggregation strategy (for Leiden variant)
        else if (opt == "leiden") {
            config.aggregation = AggregationStrategy::LEIDEN_CSR;
        } else if (opt == "streaming" || opt == "lazy") {
            config.aggregation = AggregationStrategy::RABBIT_LAZY;
        } else if (opt == "hybrid") {
            config.aggregation = AggregationStrategy::HYBRID;
        }
        // Check for community merging (key for cache locality!)
        else if (opt == "merge" || opt == "coarsen") {
            config.useCommunityMerging = true;
        }
        // Check for hub extraction (extract high-degree hubs before ordering)
        else if (opt == "hubx" || opt == "hub-extract" || opt == "hubextract") {
            config.useHubExtraction = true;
        }
        // Check for hub extraction with custom percentage: hubx0.5 = top 0.5%
        else if (opt.size() > 4 && opt.substr(0, 4) == "hubx") {
            config.useHubExtraction = true;
            double pct = 0.0;
            if (!parseExactDouble(opt.substr(4), pct) ||
                pct <= 0 || pct >= 100) {
                reject(opt);
                continue;
            }
            config.hubExtractionPct = pct / 100.0;
        }
        // Gorder-inspired improvements
        else if (opt == "gord" || opt == "gorder") {
            config.useGorderIntra = true;
        }
        // gord with custom window: gord8 = window of 8
        else if (opt.size() > 4 && opt.substr(0, 4) == "gord" && std::isdigit(opt[4])) {
            config.useGorderIntra = true;
            int window = 0;
            if (!parseExactInt(opt.substr(4), window) ||
                window <= 0 || window > 100) {
                reject(opt);
                continue;
            }
            config.gorderWindow = window;
        }
        // gord fallback threshold: gordf5000 = BFS fallback for communities > 5000
        else if (opt.size() > 5 && opt.substr(0, 5) == "gordf") {
            config.useGorderIntra = true;
            int fallback = 0;
            if (!parseExactInt(opt.substr(5), fallback) ||
                fallback <= 0) {
                reject(opt);
                continue;
            }
            config.gorderFallback = fallback;
        }
        // gw<N>: set gorderWindow without forcing legacy useGorderIntra path
        // (works for compose:intra_gorder).  Default 5.  Range 1..100.
        else if (opt.size() > 2 && opt.substr(0, 2) == "gw" && std::isdigit(opt[2])) {
            int window = 0;
            if (!parseExactInt(opt.substr(2), window) ||
                window <= 0 || window > 100) {
                reject(opt);
                continue;
            }
            config.gorderWindow = window;
        }
        // Super-graph modularity resolution for HRAB / TQR community merge
        // (γ in ΔQ = w_uv − γ·str(u)·str(v)/(2M_super)).  Default 0.25.
        // Token forms:  sgres0.5  /  sgres1  /  gamma0.1  (alias)
        else if ((opt.size() > 5 && opt.substr(0, 5) == "sgres") ||
                 (opt.size() > 5 && opt.substr(0, 5) == "gamma")) {
            double resolution = 0.0;
            if (!parseExactDouble(opt.substr(5), resolution) ||
                resolution <= 0.0 || resolution > 10.0) {
                reject(opt);
                continue;
            }
            config.superGraphResolution = resolution;
        }
        else if (opt == "hsort" || opt == "hubsort") {
            config.useHubSort = true;
        }
        else if (opt == "rcm") {
            config.useRCMSuper = true;
            config.useRCMIntra = true;
        }
        else if (opt == "rcm_super" || opt == "rcmsuper") {
            config.useRCMSuper = true;
        }
        else if (opt == "rcm_intra" || opt == "rcmintra") {
            config.useRCMIntra = true;
        }
        // Opt-out of RCM intra-community ordering (HRAB defaults to RCM intra
        // since the rcm_intra-on-by-default change; this lets the BFS-baseline
        // ablation still be exercised via "12:hrab:bfs_intra").
        else if (opt == "bfs_intra" || opt == "bfsintra" || opt == "no_rcm_intra") {
            config.useRCMIntra = false;
        }
        // Check for refinement
        else if (opt == "norefine") {
            config.useRefinement = false;
        }
        // CD-mode composability tokens (expose existing CommunityMode + flags).
        // These let composition recipes select the community-detection
        // algorithm independently of the rest of the pipeline.
        //   cd_full     = full Leiden + refinement (default reference)
        //   cd_louvain  = Leiden without refinement (≈ Louvain semantics)
        //   cd_lp_only  = single-level label propagation (no aggregation hierarchy)
        //   cd_hybrid   = LP first pass + Leiden refinement of survivors
        //   cd_gve      = alias for `refine0` (refine pass 0 only, GVE-Leiden)
        //   cd_leiden   = alias for cd_full
        else if (opt == "cd_full" || opt == "cd_leiden") {
            config.communityMode = CommunityMode::FULL_LEIDEN;
            config.useRefinement = true;
            config.refinementDepth = -1;
        }
        else if (opt == "cd_louvain") {
            config.communityMode = CommunityMode::FULL_LEIDEN;
            config.useRefinement = false;
        }
        else if (opt == "cd_lp_only" || opt == "cd_lp") {
            config.communityMode = CommunityMode::FAST_LP;
            config.useRefinement = false;
        }
        else if (opt == "cd_hybrid") {
            config.communityMode = CommunityMode::HYBRID;
            config.useRefinement = true;
            config.refinementDepth = 0;
        }
        else if (opt == "cd_gve") {
            config.communityMode = CommunityMode::FULL_LEIDEN;
            config.useRefinement = true;
            config.refinementDepth = 0;
        }
        // Refinement depth control: refine0 = pass 0 only (GVE), refine2 = passes 0-2
        else if (opt.size() > 6 && opt.substr(0, 6) == "refine" && std::isdigit(opt[6])) {
            int depth = -1;
            if (!parseExactInt(opt.substr(6), depth) || depth < 0) {
                reject(opt);
                continue;
            }
            config.refinementDepth = depth;
        }
        // M computation mode
        else if (opt == "totalm" || opt == "gvem") {
            config.mComputation = MComputation::TOTAL_EDGES;
        } else if (opt == "halfm") {
            config.mComputation = MComputation::HALF_EDGES;
        }
        // GVE-style aggregation
        else if (opt == "gvecsr" || opt == "gve-csr") {
            config.aggregation = AggregationStrategy::GVE_CSR;
        }
        // Quality preset: GVE detection quality in GraphBrew pipeline
        // Sets: GVE_CSR aggregation, TOTAL_EDGES M, refinement on pass 0 only,
        //        hierarchical ordering (top-3 pass multi-level sort)
        // Note: maxIterations left at DEFAULT_MAX_ITERATIONS (10)
        else if (opt == "quality") {
            config.aggregation = AggregationStrategy::GVE_CSR;
            config.mComputation = MComputation::TOTAL_EDGES;
            config.refinementDepth = 0;
            config.ordering = OrderingStrategy::HIERARCHICAL;
        }
        // Check for lazy community weight updates
        else if (opt == "lazyupdate" || opt == "lazyupdates") {
            config.useLazyUpdates = true;
        }
        // Check for verification
        else if (opt == "verify") {
            config.verifyTopology = true;
        }
        // Check for auto resolution (computed once from graph properties)
        else if (opt == "auto" || opt == "0") {
            config.resolution = reorder::DEFAULT_RESOLUTION;  // Signal to use auto-resolution
        }
        // Check for dynamic resolution (adjusted per-pass based on runtime metrics)
        else if (opt == "dynamic") {
            config.resolution = reorder::DEFAULT_RESOLUTION;  // Initial, will be adjusted
            config.useDynamicResolution = true;  // Enable per-pass adjustment
        }
        else if (opt.rfind("dynamic_", 0) == 0) {
            double initial = 0.0;
            if (!parseExactDouble(opt.substr(8), initial) ||
                initial <= 0.0 || initial > 3.0) {
                reject(opt);
                continue;
            }
            config.resolution = initial;
            config.useDynamicResolution = true;
        }
        // Check for numeric (resolution, iterations, passes)
        else {
            try {
                size_t parsed = 0;
                double val = std::stod(opt, &parsed);
                if (parsed != opt.size() || !std::isfinite(val)) {
                    reject(opt);
                    continue;
                }
                // Resolution: fractional value (0.0, 3.0], or if > 0 and <= 3
                // iterations/passes: integer value >= 1
                if (val > 0 && val <= 3 && (opt.find('.') != std::string::npos || val < 1)) {
                    // Contains decimal point or is fractional - likely resolution
                    config.resolution = val;
                } else if (val >= 1 && val <= 100) {
                    // Integer value - could be iterations or passes
                    int intVal = static_cast<int>(val);
                    if (config.maxIterations == reorder::DEFAULT_MAX_ITERATIONS) {
                        config.maxIterations = intVal;
                    } else {
                        config.maxPasses = intVal;
                    }
                } else if (val > 0 && val <= 3) {
                    // Small integer without decimal - likely resolution
                    config.resolution = val;
                } else {
                    reject(opt);
                }
            } catch (...) {
                reject(opt);
            }
        }
    }

    if (strict && config.ordering == OrderingStrategy::COMPOSE) {
        bool explicitSuperGraph = false;
        bool explicitCommunityOrder = false;
        for (const std::string& opt : options) {
            explicitSuperGraph =
                explicitSuperGraph ||
                opt.rfind("sg_", 0) == 0 ||
                opt.rfind("sg:", 0) == 0 ||
                opt.rfind("s1_", 0) == 0 ||
                opt.rfind("s1:", 0) == 0;
            explicitCommunityOrder =
                explicitCommunityOrder ||
                opt.rfind("comm_", 0) == 0 ||
                opt.rfind("comm:", 0) == 0 ||
                opt.rfind("s2_", 0) == 0 ||
                opt.rfind("s2:", 0) == 0;
        }
        if (explicitSuperGraph && !explicitCommunityOrder) {
            throw std::invalid_argument(
                "COMPOSE super-graph order requires an explicit "
                "community-order token");
        }
    }
    const bool directEmission =
        config.intraCommunityOrder == IntraCommunityOrder::BFSDirect
        || config.intraCommunityOrder
            == IntraCommunityOrder::BFSCompact
        || config.intraCommunityOrder
            == IntraCommunityOrder::BFSCompactDirect;
    if (
        directEmission
        && config.ordering != OrderingStrategy::COMPOSE
    ) {
        throw std::invalid_argument(
            "direct/compact BFS requires compose ordering");
    }
    if (
        directEmission
        && config.refinementPass != RefinementPass::None
    ) {
        throw std::invalid_argument(
            "direct/compact BFS requires refine_none");
    }
    const bool compactEmission =
        config.intraCommunityOrder == IntraCommunityOrder::BFSCompact
        || config.intraCommunityOrder
            == IntraCommunityOrder::BFSCompactDirect;
    if (
        compactEmission
        && (
            config.superGraphOrder != SuperGraphOrder::None
            || config.communityOrder == CommunityOrder::CutMin
        )
    ) {
        throw std::invalid_argument(
            "compact BFS emission requires sg_none and non-cut-min "
            "community order");
    }
    if (
        validateComplete
        && compactEmission
        && config.maxPasses != 1
    ) {
        throw std::invalid_argument(
            "compact BFS emission requires maxPasses=1");
    }
    return config;
}

/**
 * Parse the public GraphBrew CLI grammar, including named presets and legacy
 * positional overrides. All runtime entry points must use this wrapper rather
 * than feeding CLI options directly to the token parser above.
 */
inline GraphBrewConfig parseGraphBrewCliConfig(
    const std::vector<std::string>& options,
    double auto_resolution) {
    if (options.empty() || options[0].empty()) {
        GraphBrewConfig config = parseGraphBrewConfig(
            {"gvecsr", "totalm", "refine0", "graphbrew"}, true);
        config.resolution = auto_resolution;
        return config;
    }

    struct PresetDef {
        std::vector<std::string> tokens;
    };
    static const std::map<std::string, PresetDef> presets = {
        {"leiden", {{"gvecsr", "totalm", "refine0", "graphbrew"}}},
        {"rabbit", {{"rabbitorder", "0.5"}}},
        {"hubcluster", {{"hubcluster"}}},
    };

    auto is_numeric = [](const std::string& token) {
        try {
            size_t parsed = 0;
            const double value = std::stod(token, &parsed);
            return parsed == token.size() && std::isfinite(value);
        } catch (...) {
            return false;
        }
    };
    auto is_integer = [](const std::string& token) {
        try {
            size_t parsed = 0;
            std::stoi(token, &parsed);
            return parsed == token.size();
        } catch (...) {
            return false;
        }
    };
    auto dynamic_initial = [&](const std::string& token, double& initial) {
        if (token == "dynamic") {
            initial = auto_resolution;
            return true;
        }
        if (
            token.rfind("dynamic_", 0) != 0
            || !is_numeric(token.substr(8))
        ) {
            return false;
        }
        initial = std::stod(token.substr(8));
        return initial > 0.0 && initial <= 3.0;
    };

    const auto preset = presets.find(options[0]);
    GraphBrewConfig config;
    if (preset != presets.end()) {
        std::vector<std::string> tokens = preset->second.tokens;
        for (size_t i = 1; i < options.size(); ++i) {
            const std::string& token = options[i];
            if (token.empty()) {
                continue;
            }
            bool positional = false;
            const bool numeric = is_numeric(token);
            double initial = 0.0;
            if (
                (i == 1 || i == 3 || i == 4 || i == 5)
                && numeric
                && !is_integer(token)
            ) {
                throw std::invalid_argument(
                    "GraphBrew positional integer is malformed: " + token);
            }
            if (i == 1 && is_integer(token)) {
                positional = true;
            }
            if (
                i == 2
                && (
                    numeric
                    || token == "auto"
                    || dynamic_initial(token, initial)
                )
            ) {
                positional = true;
            }
            if ((i == 3 || i == 4) && is_integer(token)) {
                positional = true;
            }
            if (
                i == 5
                && (
                    is_integer(token)
                    || token == "auto"
                    || token == "adaptive"
                )
            ) {
                positional = true;
            }
            if (!positional) {
                tokens.push_back(token);
            }
        }
        config = parseGraphBrewConfig(tokens, true, false);

        if (
            config.algorithm != GraphBrewAlgorithm::RABBIT_ORDER
            && config.ordering == OrderingStrategy::CONNECTIVITY_BFS
        ) {
            config.ordering = OrderingStrategy::LAYER;
        }
        if (config.ordering == OrderingStrategy::LAYER) {
            config.useSmallCommunityMerging = true;
            if (config.finalAlgoId < 0) {
                config.finalAlgoId = 8;
            }
        }
        if (config.resolution == reorder::DEFAULT_RESOLUTION) {
            config.resolution = auto_resolution;
        }

        if (options.size() > 1 && is_integer(options[1])) {
            const int final_algo = std::stoi(options[1]);
            if (final_algo < 0 || final_algo > 11) {
                throw std::invalid_argument(
                    "Invalid GraphBrew final algorithm: " + options[1]);
            }
            config.finalAlgoId = final_algo;
        }
        if (options.size() > 2 && !options[2].empty()) {
            const std::string& resolution = options[2];
            if (resolution != "auto" && resolution != "0") {
                double initial = 0.0;
                if (dynamic_initial(resolution, initial)) {
                    config.resolution = initial;
                    config.useDynamicResolution = true;
                } else if (is_numeric(resolution)) {
                    const double value = std::stod(resolution);
                    if (value <= 0.0 || value > 3.0) {
                        throw std::invalid_argument(
                            "Invalid GraphBrew resolution: " + resolution);
                    }
                    config.resolution = value;
                }
            }
        }
        if (options.size() > 3 && is_integer(options[3])) {
            const int passes = std::stoi(options[3]);
            if (passes <= 0 || passes > 50) {
                throw std::invalid_argument(
                    "Invalid GraphBrew pass count: " + options[3]);
            }
            config.maxPasses = passes;
        }
        if (options.size() > 4 && !options[4].empty()) {
            const std::string& depth = options[4];
            if (depth == "recursive" || depth == "recurse") {
                config.recursiveDepth = std::max(config.recursiveDepth, 1);
            } else if (is_integer(depth)) {
                const int value = std::stoi(depth);
                if (value < 0 || value > 10) {
                    throw std::invalid_argument(
                        "Invalid GraphBrew recursion depth: " + depth);
                }
                config.recursiveDepth = value;
            }
        }
        if (options.size() > 5 && !options[5].empty()) {
            const std::string& sub_algo = options[5];
            if (sub_algo == "auto" || sub_algo == "adaptive") {
                config.subAlgoId = -1;
            } else if (is_integer(sub_algo)) {
                const int value = std::stoi(sub_algo);
                if (value < 0 || value > 11) {
                    throw std::invalid_argument(
                        "Invalid GraphBrew sub-algorithm: " + sub_algo);
                }
                config.subAlgoId = value;
            }
        }
    } else {
        config = parseGraphBrewConfig(options, true);
        if (
            config.algorithm != GraphBrewAlgorithm::RABBIT_ORDER
            && config.ordering == OrderingStrategy::CONNECTIVITY_BFS
        ) {
            config.ordering = OrderingStrategy::LAYER;
        }
        if (config.ordering == OrderingStrategy::LAYER) {
            config.useSmallCommunityMerging = true;
            if (config.finalAlgoId < 0) {
                config.finalAlgoId = 8;
            }
        }
        if (config.resolution == reorder::DEFAULT_RESOLUTION) {
            config.resolution = auto_resolution;
        }
    }
    const bool directOrCompact =
        config.intraCommunityOrder == IntraCommunityOrder::BFSDirect
        || config.intraCommunityOrder
            == IntraCommunityOrder::BFSCompact
        || config.intraCommunityOrder
            == IntraCommunityOrder::BFSCompactDirect;
    const bool compact =
        config.intraCommunityOrder == IntraCommunityOrder::BFSCompact
        || config.intraCommunityOrder
            == IntraCommunityOrder::BFSCompactDirect;
    if (
        directOrCompact
        && config.ordering != OrderingStrategy::COMPOSE
    ) {
        throw std::invalid_argument(
            "direct/compact BFS requires compose ordering");
    }
    if (
        directOrCompact
        && config.refinementPass != RefinementPass::None
    ) {
        throw std::invalid_argument(
            "direct/compact BFS requires refine_none");
    }
    if (compact && config.maxPasses != 1) {
        throw std::invalid_argument(
            "compact BFS emission requires maxPasses=1");
    }
    if (
        compact
        && (
            config.superGraphOrder != SuperGraphOrder::None
            || config.communityOrder == CommunityOrder::CutMin
        )
    ) {
        throw std::invalid_argument(
            "compact BFS emission requires sg_none and non-cut-min "
            "community order");
    }
    return config;
}

//=============================================================================

#endif  // GRAPHBREW_REORDER_GRAPHBREW_PARSER_H_
