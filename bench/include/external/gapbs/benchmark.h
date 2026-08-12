// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef BENCHMARK_H_
#define BENCHMARK_H_

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <parallel/algorithm>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "builder.h"
#include "graph.h"
#include "timer.h"
#include "util.h"
#include "writer.h"
#include "graphbrew/reorder/reorder_database.h"

/*
   GAP Benchmark Suite
   File:   Benchmark
   Author: Scott Beamer

   Various helper functions to ease writing of kernels
 */

// Default type signatures for commonly used types
typedef int32_t NodeID;
typedef int32_t WeightT;
typedef NodeWeight<NodeID, WeightT> WNode;

typedef CSRGraph<NodeID> Graph;
typedef CSRGraph<NodeID, WNode> WGraph;

typedef BuilderBase<NodeID, NodeID, WeightT> Builder;
typedef BuilderBase<NodeID, WNode, WeightT> WeightedBuilder;

typedef WriterBase<NodeID, NodeID, WeightT> Writer;
typedef WriterBase<NodeID, WNode, WeightT> WeightedWriter;

// New type definitions for PartitionedGraph
typedef std::vector<Graph> PGraph;
typedef std::vector<WGraph> PWGraph;

typedef CSRGraphFlat<NodeID> FlatGraph;
typedef std::vector<FlatGraph> PFlatGraph;

[[noreturn]] inline void FailBenchmarkVerification(
    const std::string& benchmark = "") {
    PrintLabel(
        "Verification Failure",
        benchmark.empty() ? "semantic" : benchmark);
    std::fflush(nullptr);
    std::exit(3);
}

// Used to pick random non-zero degree starting points for search algorithms
template <typename GraphT_> class SourcePicker
{
public:
    struct PickedSource
    {
        NodeID original = -1;
        NodeID internal = -1;
        int64_t out_degree = 0;
    };

    explicit SourcePicker(
        const GraphT_ &g,
        NodeID given_source = -1,
        int num_sources = 0,
        int source_repeats = 1)
        : SourcePicker(
            g,
            given_source == -1
                ? std::vector<int64_t>{}
                : std::vector<int64_t>{given_source},
            num_sources,
            source_repeats)
    {
    }

    explicit SourcePicker(
        const GraphT_ &g,
        const std::vector<int64_t> &given_sources,
        int num_sources = 0,
        int source_repeats = 1)
        : rng_(kRandSeed),
          udist_(g.num_nodes() - 1, rng_), g_(g), source_index_(0)
    {
        BuildInverseOriginalIds();

        if (!given_sources.empty())
        {
            SetExplicitSources(
                given_sources, source_repeats, num_sources);
        }
        else if (num_sources > 0)
        {
            GenerateConsistentSources(num_sources, source_repeats);
        }
    }

    // Generate a list of ORIGINAL vertex IDs to use as sources
    // This ensures all orderings explore the same vertices
    void GenerateConsistentSources(int count, int repeats = 1)
    {
        if (repeats <= 0)
            throw std::invalid_argument(
                "Source repeat count must be positive");
        consistent_sources_.clear();
        consistent_sources_.reserve(count);
        
        // Use a fresh RNG with fixed seed for reproducibility.
        std::mt19937_64 src_rng(kRandSeed);
        const int unique_count = (count + repeats - 1) / repeats;
        std::vector<NodeID> active_sources;
        active_sources.reserve(g_.num_nodes());
        for (NodeID original = 0; original < g_.num_nodes(); ++original)
        {
            if (g_.out_degree(ResolveOriginalSource(original)) > 0)
                active_sources.push_back(original);
        }
        if (active_sources.empty())
            throw std::runtime_error(
                "Unable to select a non-isolated source vertex");
        std::shuffle(active_sources.begin(), active_sources.end(), src_rng);

        std::vector<NodeID> grouped_sources;
        grouped_sources.reserve(unique_count);
        for (int index = 0; index < unique_count; ++index)
            grouped_sources.push_back(
                active_sources[index % active_sources.size()]);

        for (NodeID source : grouped_sources)
        {
            for (
                int repeat = 0;
                repeat < repeats &&
                static_cast<int>(consistent_sources_.size()) < count;
                ++repeat)
            {
                consistent_sources_.push_back(source);
            }
        }
        ResolveAllConsistentSources();
    }

    NodeID PickNextRand()
    {
        if (!consistent_sources_.empty())
        {
            if (source_index_ >= consistent_sources_.size())
                throw std::out_of_range(
                    "Source list exhausted before benchmark trials ended");
            NodeID original_id = consistent_sources_[source_index_++];
            NodeID internal_id =
                consistent_internal_sources_[source_index_ - 1];
            RecordPickedSource(original_id, internal_id);
            return internal_id;
        }
        
        // Fallback to random (for backward compatibility)
        NodeID original_id;
        NodeID internal_id;
        do
        {
            original_id = udist_();
            internal_id = ResolveOriginalSource(original_id);
        }
        while (g_.out_degree(internal_id) == 0);
        RecordPickedSource(original_id, internal_id);
        return internal_id;
    }

    NodeID PickNextTop()
    {
        return PickNextRand();
    }

    NodeID PickNext()
    {
        return PickNextRand();
    }

    NodeID last_original_source() const
    {
        return last_original_source_;
    }

    NodeID last_internal_source() const
    {
        return last_internal_source_;
    }

    int64_t last_source_out_degree() const
    {
        return last_source_out_degree_;
    }

    std::vector<PickedSource> TakePickedSources()
    {
        auto sources = std::move(picked_sources_);
        picked_sources_.clear();
        return sources;
    }

private:
    void SetExplicitSources(
        const std::vector<int64_t>& sources,
        int repeats,
        int expected_count)
    {
        if (repeats <= 0)
            throw std::invalid_argument(
                "Source repeat count must be positive");
        consistent_sources_.clear();
        consistent_sources_.reserve(sources.size() * repeats);
        if (sources.size() == 1 && expected_count > 0)
        {
            const int64_t source_value = sources.front();
            if (
                source_value < std::numeric_limits<NodeID>::min()
                || source_value > std::numeric_limits<NodeID>::max()
            ) {
                throw std::out_of_range(
                    "Source original ID is outside the NodeID domain");
            }
            consistent_sources_.assign(
                static_cast<size_t>(expected_count),
                static_cast<NodeID>(source_value));
            ResolveAllConsistentSources();
            return;
        }
        for (int64_t source_value : sources)
        {
            if (
                source_value < std::numeric_limits<NodeID>::min()
                || source_value > std::numeric_limits<NodeID>::max()
            ) {
                throw std::out_of_range(
                    "Source original ID is outside the NodeID domain");
            }
            const NodeID source = static_cast<NodeID>(source_value);
            for (int repeat = 0; repeat < repeats; ++repeat)
                consistent_sources_.push_back(source);
        }
        if (
            expected_count > 0
            && expected_count
                != static_cast<int>(consistent_sources_.size())
        ) {
            throw std::invalid_argument(
                "Benchmark trial count does not match explicit source list");
        }
        ResolveAllConsistentSources();
    }

    void ResolveAllConsistentSources()
    {
        consistent_internal_sources_.clear();
        consistent_internal_sources_.reserve(consistent_sources_.size());
        for (NodeID original : consistent_sources_)
        {
            const NodeID internal = ResolveOriginalSource(original);
            if (g_.out_degree(internal) == 0)
                throw std::invalid_argument(
                    "Explicit source resolves to an isolated vertex");
            consistent_internal_sources_.push_back(internal);
        }
    }

    void RecordPickedSource(NodeID original, NodeID internal)
    {
        last_original_source_ = original;
        last_internal_source_ = internal;
        last_source_out_degree_ = g_.out_degree(internal);
        picked_sources_.push_back({
            last_original_source_,
            last_internal_source_,
            last_source_out_degree_,
        });
    }

    void BuildInverseOriginalIds()
    {
        if (g_.get_org_ids() == nullptr)
            throw std::runtime_error(
                "Graph original-ID mapping is unavailable");
    }

    NodeID ResolveOriginalSource(NodeID original) const
    {
        if (original < 0 || original >= g_.num_nodes())
            throw std::out_of_range("Source original ID is outside the graph");
        NodeID internal = g_.get_internal_id(original);
        if (internal < 0 || internal >= g_.num_nodes())
            throw std::runtime_error("Source original ID is not mapped");
        return internal;
    }

    NodeID last_original_source_ = -1;
    NodeID last_internal_source_ = -1;
    int64_t last_source_out_degree_ = 0;
    std::mt19937_64 rng_;
    UniDist<NodeID, std::mt19937_64> udist_;
    const GraphT_ &g_;
    std::vector<NodeID> consistent_sources_; // Pre-generated ORIGINAL source IDs
    std::vector<NodeID> consistent_internal_sources_;
    std::vector<PickedSource> picked_sources_;
    size_t source_index_;           // Current index in consistent_sources_
};

// Returns k pairs with the largest values from list of key-value pairs
template <typename KeyT, typename ValT>
std::vector<std::pair<ValT, KeyT>>
                                TopK(const std::vector<std::pair<KeyT, ValT>> &to_sort, size_t k)
{
    std::vector<std::pair<ValT, KeyT>> top_k;
    ValT min_so_far = 0;
    for (auto kvp : to_sort)
    {
        if ((top_k.size() < k) || (kvp.second > min_so_far))
        {
            top_k.push_back(std::make_pair(kvp.second, kvp.first));
            __gnu_parallel::stable_sort(top_k.begin(), top_k.end(),
                                        std::greater<std::pair<ValT, KeyT>>());
            if (top_k.size() > k)
                top_k.resize(k);
            min_so_far = top_k.back().first;
        }
    }
    return top_k;
}

bool VerifyUnimplemented(...)
{
    std::cout << "** verify unimplemented **" << std::endl;
    return false;
}

// Calls (and times) kernel according to command line arguments
template <typename GraphT_, typename GraphFunc, typename AnalysisFunc,
          typename VerifierFunc>
void BenchmarkKernel(const CLApp &cli, const GraphT_ &g, GraphFunc kernel,
                     AnalysisFunc stats, VerifierFunc verify)
{
    g.PrintStats();
    double total_seconds = 0;
    Timer trial_timer;
    for (int iter = 0; iter < cli.num_trials(); iter++)
    {
        trial_timer.Start();
        auto result = kernel(g);
        trial_timer.Stop();
        PrintTime("Trial Time", trial_timer.Seconds());
        total_seconds += trial_timer.Seconds();
        if (cli.do_analysis() && (iter == (cli.num_trials() - 1)))
            stats(g, result);
        if (cli.do_verify())
        {
            trial_timer.Start();
            bool passed = verify(std::ref(g), std::ref(result));
            PrintLabel("Verification", passed ? "PASS" : "FAIL");
            trial_timer.Stop();
            PrintTime("Verification Time", trial_timer.Seconds());
            if (!passed)
                FailBenchmarkVerification();
        }
    }
    PrintTime("Average Time", total_seconds / cli.num_trials());
}

// ============================================================================
// Self-Recording BenchmarkKernel (v2.1)
// ============================================================================

/// Extract a clean graph name from a file path.
/// "/path/to/graphs/amazon/amazon.sg" → "amazon"
/// "/data/web-Google.sg" → "web-Google"
inline std::string ExtractGraphName(const std::string& filepath) {
    // Find the last path separator
    std::string base = filepath;
    auto slash = base.rfind('/');
    if (slash != std::string::npos)
        base = base.substr(slash + 1);
    // Strip extension (.sg, .el, .wel, .mtx, etc.)
    auto dot = base.rfind('.');
    if (dot != std::string::npos)
        base = base.substr(0, dot);
    return base;
}

/// Self-recording BenchmarkKernel overload.
///
/// In addition to timing and verifying, this overload:
///  1. Captures per-trial TrialResult (time, verification, answer data)
///  2. Constructs a RunReport after the trial loop
///  3. Auto-saves to benchmarks.json if self-recording is enabled
///
/// @param benchmark_name    Short name: "pr", "bfs", "cc", "sssp", "bc", "tc"
/// @param result_extractor  Lambda (graph, result) → json with answer fields
template <typename GraphT_, typename GraphFunc, typename AnalysisFunc,
          typename VerifierFunc, typename ResultExtractor>
void BenchmarkKernel(const CLApp &cli, const GraphT_ &g, GraphFunc kernel,
                     AnalysisFunc stats, VerifierFunc verify,
                     const std::string& benchmark_name,
                     ResultExtractor result_extractor)
{
    using namespace graphbrew::database;

    g.PrintStats();
    double total_seconds = 0;
    Timer trial_timer;

    // Collect per-trial results
    std::vector<TrialResult> trial_results;
    trial_results.reserve(cli.num_trials());

    for (int iter = 0; iter < cli.num_trials(); iter++)
    {
        ClearBenchmarkIterationLog();   // reset per-iteration log for this trial
        trial_timer.Start();
        auto result = kernel(g);
        trial_timer.Stop();
        PrintTime("Trial Time", trial_timer.Seconds());
        total_seconds += trial_timer.Seconds();

        // Build TrialResult
        TrialResult tr;
        tr.trial_id = iter;
        tr.time_seconds = trial_timer.Seconds();

        // Answer extraction is part of the correctness contract; fail closed.
        tr.answer = result_extractor(g, result);

        // Attach per-iteration granular data collected by the kernel
        auto& iter_log = GetBenchmarkIterationLog();
        if (!iter_log.empty()) {
            tr.answer["iterations"] = iter_log;
        }

        if (cli.do_analysis() && (iter == (cli.num_trials() - 1)))
            stats(g, result);
        if (cli.do_verify())
        {
            trial_timer.Start();
            bool passed = verify(std::ref(g), std::ref(result));
            PrintLabel("Verification", passed ? "PASS" : "FAIL");
            trial_timer.Stop();
            PrintTime("Verification Time", trial_timer.Seconds());
            tr.verified = passed;
            if (!passed)
                FailBenchmarkVerification(benchmark_name);
        }

        trial_results.push_back(std::move(tr));
    }

    double avg_time = total_seconds / cli.num_trials();
    PrintTime("Average Time", avg_time);

    // ---- Self-recording: build and save RunReport ----
    if (SelfRecordingEnabled()) {
        RunReport report;
        report.graph_name   = ExtractGraphName(cli.filename());
        report.algorithm    = GetReorderAlgoHint();
        report.algorithm_id = GetReorderAlgoIdHint();
        report.benchmark    = benchmark_name;
        report.avg_time     = avg_time;
        report.reorder_time = GetReorderTimeHint();
        const auto& preprocessing = GetPreprocessingTimingHint();
        report.representation_build_time =
            preprocessing.representation_build_time;
        report.reorder_core_time =
            preprocessing.reorder_core_time;
        report.reorder_validation_time =
            preprocessing.reorder_validation_time;
        report.reorder_apply_time =
            preprocessing.reorder_apply_time;
        report.total_preprocessing_time =
            preprocessing.total_preprocessing_time;
        report.num_trials   = cli.num_trials();
        report.trials       = std::move(trial_results);
        report.reorder_metas = GetReorderMetaHints();  // accumulated reorder metadata
        report.nodes        = g.num_nodes();
        report.edges        = g.num_edges_directed();
        report.success      = true;

        // Observation condition for this raw self-recorded run.  A unique
        // run-id (with GRAPHBREW_RUN_ID override) keeps repeated direct runs
        // from colliding; the thread policy is the OpenMP default in effect.
        report.run_id       = GenerateRunId();
#ifdef _OPENMP
        report.omp_threads  = omp_get_max_threads();
#endif

        // If no algorithm hint was set, default to "Original"
        if (report.algorithm.empty()) {
            report.algorithm = "Original";
        }

        BenchmarkDatabase::Get().append_run(report);
    }
}

#endif // BENCHMARK_H_
