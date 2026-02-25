// ============================================================================
// reorder_database.h — Database-Driven Algorithm Selection (MODE_DATABASE)
//
// Loads the centralized benchmark database (results/data/benchmarks.json) and
// graph properties (results/data/graph_properties.json) at runtime. Selects
// the best reordering algorithm using:
//   1. Oracle lookup: if the graph name matches a known graph, return the
//      algorithm family with the best (lowest) benchmark time.
//   2. kNN fallback: if the graph is unknown, compute its 12 features,
//      find the k nearest known graphs by Euclidean distance, and vote
//      on the best algorithm family weighted by inverse distance.
//
// This replaces pre-trained models (perceptron, decision tree) with a
// "streaming equation": the database IS the model. When new benchmark
// data is appended, the selection automatically improves.
//
// Usage:
//   ./pr -f graph.sg -a adaptive=mode:database,bench:pr
//
// Files:
//   results/data/benchmarks.json       — append-only benchmark records
//   results/data/graph_properties.json  — graph feature vectors
//
// NOTE: This header is included from within reorder_types.h AFTER
//       CommunityFeatures, BenchmarkType, and PerceptronSelection are
//       defined. It must NOT include reorder_types.h to avoid circular deps.
// ============================================================================

#pragma once

#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <mutex>
#include <limits>
#include <cstdio>

// nlohmann/json for robust JSON parsing of the 140K-line database
#include "../../external/nlohmann_json.hpp"

// Types CommunityFeatures, BenchmarkType, BenchmarkTypeToString() are
// expected to be already defined by reorder_types.h before this point.

namespace graphbrew {
namespace database {

// ============================================================================
// Constants
// ============================================================================

/// Directory containing the centralized data files
inline constexpr const char* DATA_DIR = "results/data/";

/// Benchmark database file (append-only, deduplicated)
inline constexpr const char* BENCHMARKS_FILE = "results/data/benchmarks.json";

/// Graph properties file (feature vectors for all known graphs)
inline constexpr const char* GRAPH_PROPS_FILE = "results/data/graph_properties.json";

/// Number of nearest neighbors for kNN selection
inline constexpr int KNN_K = 5;

/// Number of features used for kNN distance computation
inline constexpr int N_FEATURES = 12;

/// Algorithm family names (must match Python ALGO_FAMILY values)
inline const std::vector<std::string> FAMILY_NAMES = {
    "ORIGINAL", "SORT", "RCM", "HUBSORT", "GORDER", "RABBIT", "LEIDEN"
};

// ============================================================================
// Data Structures
// ============================================================================

/// A single benchmark record from benchmarks.json
struct BenchRecord {
    std::string graph;
    std::string algorithm;
    int algorithm_id = 0;
    std::string benchmark;
    double time_seconds = 0.0;
    double reorder_time = 0.0;
    int trials = 1;
    int nodes = 0;
    int edges = 0;
    bool success = true;
};

/// Feature vector for a graph (12 features, same order as Python)
struct GraphFeatureVec {
    double modularity = 0.0;
    double hub_concentration = 0.0;
    double log_nodes = 0.0;
    double log_edges = 0.0;
    double density = 0.0;
    double avg_degree_100 = 0.0;      // avg_degree / 100
    double clustering_coeff = 0.0;
    double packing_factor = 0.0;
    double forward_edge_fraction = 0.0;
    double log2_wsr = 0.0;            // log2(working_set_ratio + 1)
    double log10_cc = 0.0;            // log10(community_count + 1)
    double diameter_50 = 0.0;         // diameter / 50

    double& operator[](int i) { return reinterpret_cast<double*>(&modularity)[i]; }
    double operator[](int i) const { return reinterpret_cast<const double*>(&modularity)[i]; }
};

/// Algorithm-to-family mapping (same as Python ALGO_FAMILY)
inline std::string AlgoToFamily(const std::string& algo) {
    // Basic
    if (algo == "ORIGINAL") return "ORIGINAL";
    if (algo == "RANDOM" || algo == "SORT") return "SORT";
    // RCM
    if (algo.find("RCM") != std::string::npos) return "RCM";
    // HubSort
    if (algo.find("HUB") != std::string::npos || algo == "DBG") return "HUBSORT";
    // Gorder
    if (algo.find("GORDER") != std::string::npos || algo == "CORDER") return "GORDER";
    // Rabbit
    if (algo.find("RABBIT") != std::string::npos) return "RABBIT";
    // Leiden / GraphBrew
    if (algo.find("Leiden") != std::string::npos || algo.find("GraphBrew") != std::string::npos) return "LEIDEN";
    // Compound: check suffix
    if (algo.find("+") != std::string::npos) {
        // Use the second component as the family
        auto pos = algo.find("+");
        return AlgoToFamily(algo.substr(pos + 1));
    }
    return "ORIGINAL";
}

// ============================================================================
// BenchmarkDatabase — Singleton
// ============================================================================

/**
 * @brief Centralized benchmark database loaded from JSON.
 *
 * Thread-safe singleton. Loads once on first access, caches in memory.
 * Provides:
 *   - Oracle lookup: best_family(graph_name, benchmark)
 *   - kNN selection: best_family(features, benchmark, k)
 *   - Append: add new records and save back to disk
 */
class BenchmarkDatabase {
public:
    /// Get the singleton instance (loads from disk on first call)
    static BenchmarkDatabase& Get() {
        static BenchmarkDatabase instance;
        return instance;
    }

    /// Check if the database was successfully loaded
    bool loaded() const { return loaded_; }

    /// Number of benchmark records
    size_t num_records() const { return records_.size(); }

    /// Number of known graphs
    size_t num_graphs() const { return graph_features_.size(); }

    // ========================================================================
    // Oracle Lookup (known graph)
    // ========================================================================

    /**
     * @brief Get the best algorithm family for a known graph + benchmark.
     *
     * Returns the family that achieved the lowest time_seconds in the
     * database. Returns empty string if the graph is not in the database.
     */
    std::string best_family_oracle(const std::string& graph_name,
                                    const std::string& benchmark) const {
        auto key = graph_name + "|" + benchmark;
        auto it = oracle_cache_.find(key);
        if (it != oracle_cache_.end()) {
            return it->second;
        }
        return "";
    }

    /**
     * @brief Check if a graph name is known in the database.
     */
    bool has_graph(const std::string& graph_name) const {
        return graph_features_.count(graph_name) > 0;
    }

    // ========================================================================
    // kNN Selection (unknown graph)
    // ========================================================================

    /**
     * @brief Select the best algorithm family using k-nearest neighbors.
     *
     * Computes Euclidean distance between the query feature vector and all
     * known graphs. The k closest graphs vote for the best family, where
     * each vote is weighted by 1/distance. For each voter, the vote goes
     * to the family that performed best on that graph for the given benchmark.
     *
     * @param feat CommunityFeatures of the query graph
     * @param benchmark Benchmark name (e.g., "pr", "bfs")
     * @param k Number of nearest neighbors (default: KNN_K)
     * @param verbose Print kNN details
     * @return Best algorithm family name
     */
    std::string best_family_knn(const CommunityFeatures& feat,
                                 const std::string& benchmark,
                                 int k = KNN_K,
                                 bool verbose = false) const {
        if (graph_features_.empty()) return "ORIGINAL";

        // Extract query features (same transform as ModelTree::extract_features)
        GraphFeatureVec query;
        query.modularity = feat.modularity;
        query.hub_concentration = feat.hub_concentration;
        query.log_nodes = std::log10(static_cast<double>(feat.num_nodes) + 1.0);
        query.log_edges = std::log10(static_cast<double>(feat.num_edges) + 1.0);
        query.density = feat.internal_density;
        query.avg_degree_100 = feat.avg_degree / 100.0;
        query.clustering_coeff = feat.clustering_coeff;
        query.packing_factor = feat.packing_factor;
        query.forward_edge_fraction = feat.forward_edge_fraction;
        query.log2_wsr = std::log2(feat.working_set_ratio + 1.0);
        query.log10_cc = std::log10(feat.community_count + 1.0);
        query.diameter_50 = feat.diameter_estimate / 50.0;

        return best_family_knn_from_features(query, benchmark, k, verbose);
    }

    /**
     * @brief kNN selection from a pre-extracted feature vector.
     */
    std::string best_family_knn_from_features(const GraphFeatureVec& query,
                                               const std::string& benchmark,
                                               int k = KNN_K,
                                               bool verbose = false) const {
        // Compute distances to all known graphs
        struct Neighbor {
            std::string name;
            double distance;
        };
        std::vector<Neighbor> neighbors;
        neighbors.reserve(graph_features_.size());

        for (const auto& [name, fv] : graph_features_) {
            double dist = euclidean_distance(query, fv);
            neighbors.push_back({name, dist});
        }

        // Sort by distance
        std::sort(neighbors.begin(), neighbors.end(),
                  [](const Neighbor& a, const Neighbor& b) {
                      return a.distance < b.distance;
                  });

        // Take top-k
        int actual_k = std::min(k, static_cast<int>(neighbors.size()));

        // Weighted voting: each neighbor votes for the family that performed
        // best on it for this benchmark, weighted by 1/(distance + eps)
        std::map<std::string, double> family_votes;
        const double eps = 1e-8;

        if (verbose) {
            std::cout << "  kNN neighbors (k=" << actual_k << ", bench=" << benchmark << "):\n";
        }

        for (int i = 0; i < actual_k; ++i) {
            const auto& nb = neighbors[i];
            std::string nb_best = best_family_oracle(nb.name, benchmark);
            if (nb_best.empty()) {
                // Try "generic" benchmark
                nb_best = best_family_oracle(nb.name, "generic");
            }
            if (nb_best.empty()) nb_best = "ORIGINAL";

            double weight = 1.0 / (nb.distance + eps);
            family_votes[nb_best] += weight;

            if (verbose) {
                std::cout << "    " << (i+1) << ". " << nb.name
                          << " (dist=" << nb.distance
                          << ") → " << nb_best
                          << " (weight=" << weight << ")\n";
            }
        }

        // Find family with highest total vote weight
        std::string best = "ORIGINAL";
        double best_weight = -1.0;
        for (const auto& [fam, w] : family_votes) {
            if (w > best_weight) {
                best_weight = w;
                best = fam;
            }
        }

        if (verbose) {
            std::cout << "  kNN result: " << best << " (total_weight=" << best_weight << ")\n";
        }

        return best;
    }

    // ========================================================================
    // Unified Selection (oracle if known, kNN if unknown)
    // ========================================================================

    /**
     * @brief Select the best algorithm family for a graph.
     *
     * If graph_name is known in the database → oracle lookup.
     * Otherwise → kNN on the 12-feature vector.
     *
     * @param feat CommunityFeatures of the graph
     * @param graph_name Name of the graph (for oracle lookup)
     * @param benchmark Benchmark name (e.g., "pr")
     * @param verbose Print selection details
     * @return Best algorithm family name
     */
    std::string select(const CommunityFeatures& feat,
                       const std::string& graph_name,
                       const std::string& benchmark,
                       bool verbose = false) const {
        // Try oracle first (direct lookup from the database)
        if (!graph_name.empty() && has_graph(graph_name)) {
            std::string family = best_family_oracle(graph_name, benchmark);
            if (!family.empty()) {
                if (verbose) {
                    std::cout << "  Database: oracle lookup → " << family
                              << " (known graph: " << graph_name << ")\n";
                }
                return family;
            }
        }

        // Fall back to kNN
        if (verbose) {
            std::cout << "  Database: kNN lookup (unknown graph: "
                      << (graph_name.empty() ? "<unnamed>" : graph_name) << ")\n";
        }
        return best_family_knn(feat, benchmark, KNN_K, verbose);
    }

    // ========================================================================
    // Append New Records
    // ========================================================================

    /**
     * @brief Append a new benchmark record to the database and save to disk.
     *
     * Deduplicates by (graph, algorithm, benchmark) key, keeping the
     * record with the lowest time_seconds.
     *
     * Thread-safe.
     */
    void append(const BenchRecord& rec) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto key = rec.graph + "|" + rec.algorithm + "|" + rec.benchmark;

        auto it = dedup_.find(key);
        if (it != dedup_.end()) {
            // Keep the one with lower time
            auto& existing = records_[it->second];
            if (rec.time_seconds < existing.time_seconds) {
                existing = rec;
                rebuild_oracle_entry(rec);
            }
        } else {
            dedup_[key] = records_.size();
            records_.push_back(rec);
            rebuild_oracle_entry(rec);
        }
    }

    /**
     * @brief Append a new benchmark record and update graph properties.
     *
     * Also updates the graph feature vector from CommunityFeatures.
     * Use this when the C++ runtime has just benchmarked a new graph.
     */
    void append_with_features(const BenchRecord& rec,
                               const CommunityFeatures& feat) {
        append(rec);

        // Update graph properties
        std::lock_guard<std::mutex> lock(mutex_);
        auto& fv = graph_features_[rec.graph];
        fv.modularity = feat.modularity;
        fv.hub_concentration = feat.hub_concentration;
        fv.log_nodes = std::log10(static_cast<double>(feat.num_nodes) + 1.0);
        fv.log_edges = std::log10(static_cast<double>(feat.num_edges) + 1.0);
        fv.density = feat.internal_density;
        fv.avg_degree_100 = feat.avg_degree / 100.0;
        fv.clustering_coeff = feat.clustering_coeff;
        fv.packing_factor = feat.packing_factor;
        fv.forward_edge_fraction = feat.forward_edge_fraction;
        fv.log2_wsr = std::log2(feat.working_set_ratio + 1.0);
        fv.log10_cc = std::log10(feat.community_count + 1.0);
        fv.diameter_50 = feat.diameter_estimate / 50.0;
    }

    /**
     * @brief Save the current database to disk (atomic write).
     *
     * Writes to benchmarks.json and graph_properties.json.
     */
    bool save() {
        std::lock_guard<std::mutex> lock(mutex_);
        return save_benchmarks() && save_graph_props();
    }

    /**
     * @brief Force reload from disk.
     */
    void reload() {
        std::lock_guard<std::mutex> lock(mutex_);
        records_.clear();
        dedup_.clear();
        oracle_cache_.clear();
        graph_features_.clear();
        loaded_ = false;
        load();
    }

    /**
     * @brief Print database statistics.
     */
    void print_stats() const {
        std::cout << "BenchmarkDatabase: "
                  << records_.size() << " records, "
                  << graph_features_.size() << " graphs";

        // Count unique benchmarks
        std::set<std::string> benchmarks;
        for (const auto& r : records_) benchmarks.insert(r.benchmark);
        std::cout << ", " << benchmarks.size() << " benchmarks";

        // Count unique algorithms
        std::set<std::string> algos;
        for (const auto& r : records_) algos.insert(r.algorithm);
        std::cout << ", " << algos.size() << " algorithms\n";
    }

private:
    BenchmarkDatabase() { load(); }
    BenchmarkDatabase(const BenchmarkDatabase&) = delete;
    BenchmarkDatabase& operator=(const BenchmarkDatabase&) = delete;

    // ========================================================================
    // Loading
    // ========================================================================

    void load() {
        load_benchmarks();
        load_graph_props();
        build_oracle_cache();
        loaded_ = !records_.empty();

        if (loaded_) {
            std::cout << "[DATABASE] Loaded " << records_.size()
                      << " records, " << graph_features_.size()
                      << " graph feature vectors\n";
        }
    }

    void load_benchmarks() {
        std::ifstream ifs(BENCHMARKS_FILE);
        if (!ifs.is_open()) {
            std::cerr << "[DATABASE] Warning: cannot open " << BENCHMARKS_FILE << "\n";
            return;
        }

        try {
            nlohmann::json j;
            ifs >> j;

            if (!j.is_array()) {
                std::cerr << "[DATABASE] Error: benchmarks.json is not an array\n";
                return;
            }

            records_.reserve(j.size());
            for (const auto& entry : j) {
                BenchRecord rec;
                rec.graph = entry.value("graph", "");
                rec.algorithm = entry.value("algorithm", "");
                rec.algorithm_id = entry.value("algorithm_id", 0);
                rec.benchmark = entry.value("benchmark", "");
                rec.time_seconds = entry.value("time_seconds", 0.0);
                rec.reorder_time = entry.value("reorder_time", 0.0);
                rec.trials = entry.value("trials", 1);
                rec.nodes = entry.value("nodes", 0);
                rec.edges = entry.value("edges", 0);
                rec.success = entry.value("success", true);

                if (!rec.success || rec.graph.empty() || rec.algorithm.empty())
                    continue;

                // Dedup by (graph, algorithm, benchmark), keep lowest time
                auto key = rec.graph + "|" + rec.algorithm + "|" + rec.benchmark;
                auto it = dedup_.find(key);
                if (it != dedup_.end()) {
                    if (rec.time_seconds < records_[it->second].time_seconds) {
                        records_[it->second] = rec;
                    }
                } else {
                    dedup_[key] = records_.size();
                    records_.push_back(rec);
                }
            }
        } catch (const std::exception& e) {
            std::cerr << "[DATABASE] Error parsing " << BENCHMARKS_FILE
                      << ": " << e.what() << "\n";
        }
    }

    void load_graph_props() {
        std::ifstream ifs(GRAPH_PROPS_FILE);
        if (!ifs.is_open()) {
            std::cerr << "[DATABASE] Warning: cannot open " << GRAPH_PROPS_FILE << "\n";
            return;
        }

        try {
            nlohmann::json j;
            ifs >> j;

            if (!j.is_object()) {
                std::cerr << "[DATABASE] Error: graph_properties.json is not an object\n";
                return;
            }

            for (auto it = j.begin(); it != j.end(); ++it) {
                const std::string& name = it.key();
                const auto& props = it.value();

                GraphFeatureVec fv;
                double raw_nodes = props.value("nodes", 0.0);
                double raw_edges = props.value("edges", 0.0);
                double raw_avg_degree = props.value("avg_degree", 0.0);
                double raw_wsr = props.value("working_set_ratio", 0.0);
                double raw_cc = props.value("community_count", 0.0);
                double raw_diameter = props.value("diameter", 0.0);

                // Apply the same transforms as ModelTree::extract_features
                fv.modularity = props.value("modularity", 0.0);
                fv.hub_concentration = props.value("hub_concentration", 0.0);
                fv.log_nodes = std::log10(raw_nodes + 1.0);
                fv.log_edges = std::log10(raw_edges + 1.0);
                fv.density = props.value("density", 0.0);
                fv.avg_degree_100 = raw_avg_degree / 100.0;
                fv.clustering_coeff = props.value("clustering_coefficient", 0.0);
                fv.packing_factor = props.value("packing_factor", 0.0);
                fv.forward_edge_fraction = props.value("forward_edge_fraction", 0.0);
                fv.log2_wsr = std::log2(raw_wsr + 1.0);
                fv.log10_cc = std::log10(raw_cc + 1.0);
                fv.diameter_50 = raw_diameter / 50.0;

                graph_features_[name] = fv;
            }
        } catch (const std::exception& e) {
            std::cerr << "[DATABASE] Error parsing " << GRAPH_PROPS_FILE
                      << ": " << e.what() << "\n";
        }
    }

    // ========================================================================
    // Oracle Cache
    // ========================================================================

    /**
     * Build the oracle cache: for each (graph, benchmark) → best family.
     *
     * Strategy: group records by (graph, benchmark), find the family
     * with the lowest average time_seconds across all algorithms in that family.
     */
    void build_oracle_cache() {
        // Collect: (graph, benchmark) → family → [times]
        std::map<std::string, std::map<std::string, std::vector<double>>> grouped;

        for (const auto& r : records_) {
            std::string family = AlgoToFamily(r.algorithm);
            std::string key = r.graph + "|" + r.benchmark;
            grouped[key][family].push_back(r.time_seconds);
        }

        // For each (graph, benchmark), pick the family with the lowest min time
        for (const auto& [key, families] : grouped) {
            std::string best_fam;
            double best_time = std::numeric_limits<double>::infinity();

            for (const auto& [fam, times] : families) {
                // Use the minimum time across all algorithms in this family
                double min_t = *std::min_element(times.begin(), times.end());
                if (min_t < best_time) {
                    best_time = min_t;
                    best_fam = fam;
                }
            }

            oracle_cache_[key] = best_fam;
        }
    }

    void rebuild_oracle_entry(const BenchRecord& rec) {
        // Rebuild oracle cache for this (graph, benchmark) pair
        std::string bench_key = rec.graph + "|" + rec.benchmark;
        std::map<std::string, double> family_best;

        for (const auto& r : records_) {
            if (r.graph == rec.graph && r.benchmark == rec.benchmark) {
                std::string fam = AlgoToFamily(r.algorithm);
                auto it = family_best.find(fam);
                if (it == family_best.end() || r.time_seconds < it->second) {
                    family_best[fam] = r.time_seconds;
                }
            }
        }

        std::string best_fam = "ORIGINAL";
        double best_time = std::numeric_limits<double>::infinity();
        for (const auto& [fam, t] : family_best) {
            if (t < best_time) {
                best_time = t;
                best_fam = fam;
            }
        }
        oracle_cache_[bench_key] = best_fam;
    }

    // ========================================================================
    // Saving
    // ========================================================================

    bool save_benchmarks() {
        nlohmann::json j = nlohmann::json::array();

        // Sort by (graph, benchmark, algorithm) for deterministic output
        auto sorted = records_;
        std::sort(sorted.begin(), sorted.end(),
                  [](const BenchRecord& a, const BenchRecord& b) {
                      if (a.graph != b.graph) return a.graph < b.graph;
                      if (a.benchmark != b.benchmark) return a.benchmark < b.benchmark;
                      return a.algorithm < b.algorithm;
                  });

        for (const auto& r : sorted) {
            nlohmann::json entry;
            entry["graph"] = r.graph;
            entry["algorithm"] = r.algorithm;
            entry["algorithm_id"] = r.algorithm_id;
            entry["benchmark"] = r.benchmark;
            entry["time_seconds"] = r.time_seconds;
            entry["reorder_time"] = r.reorder_time;
            entry["trials"] = r.trials;
            entry["nodes"] = r.nodes;
            entry["edges"] = r.edges;
            entry["success"] = r.success;
            entry["error"] = "";
            entry["extra"] = nlohmann::json::object();
            j.push_back(entry);
        }

        // Atomic write via temp file
        std::string tmp = std::string(BENCHMARKS_FILE) + ".tmp";
        {
            std::ofstream ofs(tmp);
            if (!ofs.is_open()) {
                std::cerr << "[DATABASE] Cannot write to " << tmp << "\n";
                return false;
            }
            ofs << j.dump(2) << "\n";
        }
        if (std::rename(tmp.c_str(), BENCHMARKS_FILE) != 0) {
            std::cerr << "[DATABASE] Cannot rename " << tmp << " → "
                      << BENCHMARKS_FILE << "\n";
            return false;
        }
        return true;
    }

    bool save_graph_props() {
        nlohmann::json j;

        for (const auto& [name, fv] : graph_features_) {
            nlohmann::json props;
            // Reverse-transform features back to raw values for storage
            props["modularity"] = fv.modularity;
            props["hub_concentration"] = fv.hub_concentration;
            props["nodes"] = static_cast<int>(std::pow(10.0, fv.log_nodes) - 1.0);
            props["edges"] = static_cast<int>(std::pow(10.0, fv.log_edges) - 1.0);
            props["density"] = fv.density;
            props["avg_degree"] = fv.avg_degree_100 * 100.0;
            props["clustering_coefficient"] = fv.clustering_coeff;
            props["packing_factor"] = fv.packing_factor;
            props["forward_edge_fraction"] = fv.forward_edge_fraction;
            props["working_set_ratio"] = std::pow(2.0, fv.log2_wsr) - 1.0;
            props["community_count"] = std::pow(10.0, fv.log10_cc) - 1.0;
            props["diameter"] = fv.diameter_50 * 50.0;
            j[name] = props;
        }

        std::string tmp = std::string(GRAPH_PROPS_FILE) + ".tmp";
        {
            std::ofstream ofs(tmp);
            if (!ofs.is_open()) return false;
            ofs << j.dump(2) << "\n";
        }
        return std::rename(tmp.c_str(), GRAPH_PROPS_FILE) == 0;
    }

    // ========================================================================
    // Distance Computation
    // ========================================================================

    static double euclidean_distance(const GraphFeatureVec& a,
                                      const GraphFeatureVec& b) {
        double sum = 0.0;
        for (int i = 0; i < N_FEATURES; ++i) {
            double d = a[i] - b[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    }

    // ========================================================================
    // Data Members
    // ========================================================================

    std::vector<BenchRecord> records_;
    std::unordered_map<std::string, size_t> dedup_;           // key → index
    std::unordered_map<std::string, std::string> oracle_cache_;  // "graph|bench" → family
    std::map<std::string, GraphFeatureVec> graph_features_;   // graph_name → features
    bool loaded_ = false;
    std::mutex mutex_;
};

// ============================================================================
// Convenience Functions (called from reorder_types.h)
// ============================================================================

/**
 * @brief Select algorithm family using the database.
 *
 * Called from SelectReorderingWithMode() for MODE_DATABASE.
 */
inline std::string SelectAlgorithmDatabase(
    const CommunityFeatures& feat,
    BenchmarkType bench,
    const std::string& graph_name = "",
    bool verbose = false) {

    auto& db = BenchmarkDatabase::Get();

    if (!db.loaded()) {
        if (verbose) {
            std::cout << "  Database: not loaded, falling back to ORIGINAL\n";
        }
        return "";
    }

    // Convert BenchmarkType enum to string
    static const char* bench_names[] = {
        "generic", "pr", "bfs", "cc", "sssp", "bc", "tc", "pr_spmv", "cc_sv"
    };
    std::string bench_str = "generic";
    if (static_cast<int>(bench) >= 0 &&
        static_cast<int>(bench) < static_cast<int>(sizeof(bench_names)/sizeof(bench_names[0]))) {
        bench_str = bench_names[static_cast<int>(bench)];
    }

    std::string family = db.select(feat, graph_name, bench_str, verbose);

    if (verbose) {
        std::cout << "  Database selection: " << family
                  << " (graph=" << (graph_name.empty() ? "<unnamed>" : graph_name)
                  << ", bench=" << bench_str << ")\n";
    }

    return family;
}

/**
 * @brief Append a benchmark result to the centralized database.
 *
 * Called after running a benchmark to update the database in real-time.
 */
inline void AppendBenchmarkResult(
    const std::string& graph_name,
    const std::string& algorithm,
    int algorithm_id,
    const std::string& benchmark,
    double time_seconds,
    double reorder_time,
    int trials,
    int nodes,
    int edges,
    const CommunityFeatures& feat) {

    BenchRecord rec;
    rec.graph = graph_name;
    rec.algorithm = algorithm;
    rec.algorithm_id = algorithm_id;
    rec.benchmark = benchmark;
    rec.time_seconds = time_seconds;
    rec.reorder_time = reorder_time;
    rec.trials = trials;
    rec.nodes = nodes;
    rec.edges = edges;
    rec.success = true;

    auto& db = BenchmarkDatabase::Get();
    db.append_with_features(rec, feat);
    db.save();
}

}  // namespace database
}  // namespace graphbrew
