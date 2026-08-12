// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef COMMAND_LINE_H_
#define COMMAND_LINE_H_

#include <getopt.h>

#include "util.h"
#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

/*
   GAP Benchmark Suite
   Class:  CLBase
   Author: Scott Beamer

   Handles command line argument parsing
   - Through inheritance, can add more options to object
   - For example, most kernels will use CLApp
 */

class CLBase
{
protected:
    int argc_;
    char **argv_;
    std::string name_;
    std::string get_args_ = "f:g:hk:su:m:o:zj:SlD:W:";
    std::vector<std::string> help_strings_;
    std::vector<std::pair<ReorderingAlgo, std::vector<std::string> > >
    reorder_options_;

    int scale_ = -1;
    int degree_ = 16;
    std::string filename_ = "";
    bool symmetrize_ = false;
    bool uniform_ = false;
    bool keep_self_ = false;
    bool in_place_ = false;
    bool use_out_degree_ = true;
    std::vector<int> segments_{0, 1, 1};   // Default to one segment ir
    bool enable_logging_ = false;
    std::string db_dir_ = "";              // --db-dir: JSON database output directory
    std::string weight_scheme_ = "unit";   // generated weights for .sg -> WGraph

    void AddHelpLine(char opt, std::string opt_arg, std::string text,
                     std::string def = "")
    {
        const int kBufLen = 200;
        char buf[kBufLen];
        if (opt_arg != "")
            opt_arg = "<" + opt_arg + ">";
        if (def != "")
            def = "[" + def + "]";
        snprintf(buf, kBufLen, " -%c %-9s: %-54s%10s", opt, opt_arg.c_str(),
                 text.c_str(), def.c_str());
        help_strings_.push_back(buf);
    }

public:
    CLBase(int argc, char **argv, std::string name = "")
        : argc_(argc), argv_(argv), name_(name)
    {
        AddHelpLine('h', "", "print this help message");
        AddHelpLine('f', "file", "load graph from file");
        AddHelpLine('s', "", "symmetrize input edge list", "false");
        AddHelpLine('g', "scale", "generate 2^scale kronecker graph");
        AddHelpLine('u', "scale", "generate 2^scale uniform-random graph");
        AddHelpLine('k', "degree", "average degree for synthetic graph",
                    std::to_string(degree_));
        AddHelpLine('m', "", "reduces memory usage during graph building", "false");
        AddHelpLine('o', "order",
                    "apply reordering strategy, optionally with a parameter \n     "
                    "          [example]-o <12|13>:option1:option2 "
                    "-o 2 -o 14:mapping.label",
                    "optional");
        AddHelpLine('z', "indegree",
                    "use indegree for ordering [Degree Based Orderings]", "false");
        AddHelpLine('j', "segments", "number of segments for the graph \n     "
                    "          [type:n:m] <0:GRAPHIT/Cagra> <1:TRUST>", "0:1:1");
        AddHelpLine('l', "", "log performance within each trial", "false");
        AddHelpLine('S', "", "keep self loops", "false");
        AddHelpLine('D', "dir", "database directory for JSON output", "results/data");
        AddHelpLine('W', "scheme", "generated edge weights for .sg: unit or hash",
                    weight_scheme_);
    }

    bool ParseArgs()
    {
        signed char c_opt;
        extern char *optarg; // from and for getopt
        while ((c_opt = getopt(argc_, argv_, get_args_.c_str())) != -1)
        {
            HandleArg(c_opt, optarg);
        }
        if ((filename_ == "") && (scale_ == -1))
        {
            std::cout << "No graph input specified. (Use -h for help)" << std::endl;
            return false;
        }
        if (scale_ != -1)
            symmetrize_ = true;
        return true;
    }

    void virtual HandleArg(signed char opt, char *opt_arg)
    {
        switch (opt)
        {
        case 'f':
            filename_ = std::string(opt_arg);
            break;
        case 'g':
            scale_ = atoi(opt_arg);
            break;
        case 'h':
            PrintUsage();
            break;
        case 'k':
            degree_ = atoi(opt_arg);
            break;
        case 'z':
            use_out_degree_ = false;
            break;
        case 's':
            symmetrize_ = true;
            break;
        case 'u':
            uniform_ = true;
            scale_ = atoi(opt_arg);
            break;
        case 'm':
            in_place_ = true;
            break;
        case 'l':
            enable_logging_ = true;
            break;
        case 'S':
            keep_self_ = true;
            break;
        case 'D':
            db_dir_ = std::string(opt_arg);
            break;
        case 'W':
            weight_scheme_ = std::string(opt_arg);
            if (weight_scheme_ != "unit" && weight_scheme_ != "hash")
                throw std::invalid_argument(
                    "Weight scheme must be unit or hash");
            break;
        case 'o':
        {
            std::string arg(opt_arg);
            size_t pos = arg.find(':');
            ReorderingAlgo algo =
                static_cast<ReorderingAlgo>(std::stoi(arg.substr(0, pos)));
            std::vector<std::string> params;
            if (pos != std::string::npos)
            {
                size_t start = pos + 1;
                size_t end;
                while ((end = arg.find(':', start)) != std::string::npos)
                {
                    params.push_back(arg.substr(start, end - start));
                    start = end + 1;
                }
                params.push_back(arg.substr(start));
            }
            reorder_options_.emplace_back(algo, params);
        }
        break;
        case 'j':
        {
            std::string arg(opt_arg);
            size_t start = 0;
            size_t end = arg.find(':');
            size_t index = 0;

            while (end != std::string::npos)
            {
                if (index < segments_.size())
                {
                    segments_[index] = std::stoi(arg.substr(start, end - start));
                }
                else
                {
                    segments_.push_back(std::stoi(arg.substr(start, end - start)));
                }
                start = end + 1;
                end = arg.find(':', start);
                ++index;
            }

            // Handle the last segment
            if (index < segments_.size())
            {
                segments_[index] = std::stoi(arg.substr(start));
            }
            else
            {
                segments_.push_back(std::stoi(arg.substr(start)));
            }
        }
        break;
        }
    }

    void PrintUsage()
    {
        std::cout << name_ << std::endl;
        // std::sort(help_strings_.begin(), help_strings_.end());
        for (std::string h : help_strings_)
            std::cout << h << std::endl;
        std::exit(0);
    }

    int scale() const
    {
        return scale_;
    }
    int degree() const
    {
        return degree_;
    }
    std::string filename() const
    {
        return filename_;
    }
    const std::string &weight_scheme() const
    {
        return weight_scheme_;
    }
    bool symmetrize() const
    {
        return symmetrize_;
    }
    bool keep_self() const
    {
        return keep_self_;
    }
    void set_keep_self(bool keep_self_loop = false)
    {
        keep_self_ = keep_self_loop;
    }
    bool uniform() const
    {
        return uniform_;
    }
    bool in_place() const
    {
        return in_place_;
    }
    bool use_out_degree() const
    {
        return use_out_degree_;
    }
    bool logging_en() const
    {
        return enable_logging_;
    }
    const std::vector<std::pair<ReorderingAlgo, std::vector<std::string> > > &
    reorder_options() const
    {
        return reorder_options_;
    }
    const std::vector<int> &segments() const
    {
        return segments_;
    }
    std::string db_dir() const
    {
        return db_dir_;
    }
};

class CLApp : public CLBase
{
    bool do_analysis_ = false;
    int num_trials_ = 16;
    bool num_trials_explicit_ = false;
    int source_repeats_ = 1;
    std::vector<int64_t> start_vertices_;
    bool do_verify_ = false;

    static std::vector<int64_t> ParseSourceList(
        const std::string& text)
    {
        if (text.empty())
            throw std::invalid_argument("Source list cannot be empty");
        std::vector<int64_t> sources;
        std::set<int64_t> seen;
        size_t start = 0;
        while (start <= text.size())
        {
            const size_t comma = text.find(',', start);
            const std::string token = text.substr(
                start,
                comma == std::string::npos
                    ? std::string::npos : comma - start);
            if (token.empty())
                throw std::invalid_argument(
                    "Source list contains an empty field");
            size_t parsed = 0;
            const long long source = std::stoll(token, &parsed);
            if (parsed != token.size() || source < 0)
                throw std::invalid_argument(
                    "Source IDs must be non-negative integers");
            if (!seen.insert(source).second)
                throw std::invalid_argument(
                    "Source list contains a duplicate ID");
            sources.push_back(source);
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        return sources;
    }

public:
    CLApp(int argc, char **argv, std::string name) : CLBase(argc, argv, name)
    {
        get_args_ += "an:r:R:v";
        AddHelpLine('a', "", "output analysis of last run", "false");
        AddHelpLine('n', "n", "perform n trials", std::to_string(num_trials_));
        AddHelpLine(
            'r', "id[,id...]",
            "original source ID or comma-separated original-ID list",
            "rand");
        AddHelpLine(
            'R', "repeats",
            "repeat each deterministic source this many consecutive trials",
            std::to_string(source_repeats_));
        AddHelpLine('v', "", "verify the output of each run", "false");
    }

    void HandleArg(signed char opt, char *opt_arg) override
    {
        switch (opt)
        {
        case 'a':
            do_analysis_ = true;
            break;
        case 'n':
            num_trials_ = atoi(opt_arg);
            num_trials_explicit_ = true;
            if (num_trials_ <= 0)
                throw std::invalid_argument(
                    "Trial count must be positive");
            break;
        case 'r':
            start_vertices_ = ParseSourceList(opt_arg);
            break;
        case 'R':
            source_repeats_ = std::stoi(opt_arg);
            if (source_repeats_ <= 0)
                throw std::invalid_argument(
                    "Source repeat count must be positive");
            break;
        case 'v':
            do_verify_ = true;
            break;
        default:
            CLBase::HandleArg(opt, opt_arg);
        }
    }

    bool ParseArgs()
    {
        const bool parsed = CLBase::ParseArgs();
        if (!parsed) return false;
        if (!start_vertices_.empty())
        {
            if (
                start_vertices_.size() > 1
                || SourceRepeatMultiplier() > 1
            )
            {
                const int expected = static_cast<int>(
                    start_vertices_.size()) * SourceRepeatMultiplier();
                if (num_trials_explicit_ && num_trials_ != expected)
                    throw std::invalid_argument(
                        "Explicit source list requires -n to equal "
                        "source_count * source_repeats");
                num_trials_ = expected;
            }
        }
        return true;
    }

    int SourceRepeatMultiplier() const
    {
        return source_repeats_;
    }
    int source_repeats() const
    {
        return source_repeats_;
    }
    bool do_analysis() const
    {
        return do_analysis_;
    }
    int num_trials() const
    {
        return num_trials_;
    }
    int64_t start_vertex() const
    {
        return start_vertices_.size() == 1
            ? start_vertices_.front() : -1;
    }
    const std::vector<int64_t>& start_vertices() const
    {
        return start_vertices_;
    }
    bool do_verify() const
    {
        return do_verify_;
    }
};

class CLPartitionApp : public CLApp
{
    int num_partitions_ = 2;
    std::string partition_balance_ = "total";
    std::string partition_export_dir_;

public:
    CLPartitionApp(int argc, char **argv, std::string name)
        : CLApp(argc, argv, name)
    {
        get_args_ += "P:B:E:";
        AddHelpLine(
            'P', "partitions", "build this many compact graph shards",
            std::to_string(num_partitions_));
        AddHelpLine(
            'B', "balance", "partition balance: vertices, out, or total",
            partition_balance_);
        AddHelpLine(
            'E', "dir", "export a graph.shard.v1 package to this directory");
    }

    void HandleArg(signed char opt, char *opt_arg) override
    {
        switch (opt)
        {
        case 'P':
            num_partitions_ = std::stoi(opt_arg);
            if (num_partitions_ <= 0)
                throw std::invalid_argument(
                    "Graph partition count must be positive");
            break;
        case 'B':
            partition_balance_ = opt_arg;
            if (
                partition_balance_ != "vertices" &&
                partition_balance_ != "vertex" &&
                partition_balance_ != "out" &&
                partition_balance_ != "outgoing" &&
                partition_balance_ != "total" &&
                partition_balance_ != "edges")
            {
                throw std::invalid_argument(
                    "Partition balance must be vertices, out, or total");
            }
            break;
        case 'E':
            partition_export_dir_ = opt_arg;
            if (partition_export_dir_.empty())
                throw std::invalid_argument(
                    "Partition export directory must not be empty");
            break;
        default:
            CLApp::HandleArg(opt, opt_arg);
        }
    }

    int num_partitions() const
    {
        return num_partitions_;
    }

    const std::string &partition_balance() const
    {
        return partition_balance_;
    }

    const std::string &partition_export_dir() const
    {
        return partition_export_dir_;
    }
};

class CLIterApp : public CLApp
{
    int num_iters_;

public:
    CLIterApp(int argc, char **argv, std::string name, int num_iters)
        : CLApp(argc, argv, name), num_iters_(num_iters)
    {
        get_args_ += "i:";
        AddHelpLine('i', "i", "perform i iterations", std::to_string(num_iters_));
    }

    void HandleArg(signed char opt, char *opt_arg) override
    {
        switch (opt)
        {
        case 'i':
            num_iters_ = atoi(opt_arg);
            break;
        default:
            CLApp::HandleArg(opt, opt_arg);
        }
    }

    int num_iters() const
    {
        return num_iters_;
    }
};

class CLPageRank : public CLApp
{
    int max_iters_;
    double tolerance_;
    bool fixed_work_ = false;

public:
    CLPageRank(int argc, char **argv, std::string name, double tolerance,
               int max_iters)
        : CLApp(argc, argv, name), max_iters_(max_iters), tolerance_(tolerance)
    {
        get_args_ += "i:t:F";
        AddHelpLine('i', "i", "perform at most i iterations",
                    std::to_string(max_iters_));
        AddHelpLine('t', "t", "use tolerance t", std::to_string(tolerance_));
        AddHelpLine(
            'F', "", "run exactly i iterations (ignore early convergence)",
            "false");
    }

    void HandleArg(signed char opt, char *opt_arg) override
    {
        switch (opt)
        {
        case 'i':
            max_iters_ = std::stoi(opt_arg);
            if (max_iters_ <= 0)
                throw std::invalid_argument(
                    "PageRank iteration count must be positive");
            break;
        case 't':
            tolerance_ = std::stod(opt_arg);
            if (!std::isfinite(tolerance_) || tolerance_ < 0)
                throw std::invalid_argument(
                    "PageRank tolerance must be finite and non-negative");
            break;
        case 'F':
            fixed_work_ = true;
            break;
        default:
            CLApp::HandleArg(opt, opt_arg);
        }
    }

    int max_iters() const
    {
        return max_iters_;
    }
    double tolerance() const
    {
        return tolerance_;
    }
    bool fixed_work() const
    {
        return fixed_work_;
    }
};

template <typename WeightT_> class CLDelta : public CLApp
{
    WeightT_ delta_ = 1;

public:
    CLDelta(int argc, char **argv, std::string name) : CLApp(argc, argv, name)
    {
        get_args_ += "d:";
        AddHelpLine('d', "d", "delta parameter", std::to_string(delta_));
    }

    void HandleArg(signed char opt, char *opt_arg) override
    {
        switch (opt)
        {
        case 'd':
        {
            std::string value_text(opt_arg);
            size_t parsed = 0;
            if constexpr (std::is_floating_point<WeightT_>::value) {
                double value = std::stod(value_text, &parsed);
                if (
                    parsed != value_text.size() ||
                    !std::isfinite(value) || value <= 0)
                    throw std::invalid_argument(
                        "Delta must be a finite positive number");
                delta_ = static_cast<WeightT_>(value);
            } else {
                long long value = std::stoll(value_text, &parsed);
                if (
                    parsed != value_text.size() || value <= 0 ||
                    value > std::numeric_limits<WeightT_>::max())
                    throw std::invalid_argument(
                        "Delta must be a positive in-range integer");
                delta_ = static_cast<WeightT_>(value);
            }
            break;
        }
        default:
            CLApp::HandleArg(opt, opt_arg);
        }
    }

    WeightT_ delta() const
    {
        return delta_;
    }

};

class CLConvert : public CLBase
{
    std::string out_filename_ = "";
    std::string label_out_filename_ = "";
    bool out_weighted_ = false;
    bool out_el_ = false;
    bool out_mtx_ = false;
    bool out_sg_ = false;
    bool out_ligra_ = false;
    bool out_structures_ = false;
    bool out_label_so_ = false;
    bool out_label_lo_ = false;

public:
    CLConvert(int argc, char **argv, std::string name)
        : CLBase(argc, argv, name)
    {
        get_args_ += "e:b:x:q:p:y:V:w";
        AddHelpLine('b', "file", "output serialized graph to file (.sg)");
        AddHelpLine('e', "file", "output edge list to file (.el)");
        AddHelpLine('V', "file", "output separate files each has csr array content .out_degree/.out_neigh/.offset");
        AddHelpLine('p', "file",
                    "output matrix market exchange format to file (.mtx)");
        AddHelpLine('y', "file",
                    "output in Ligra adjacency graph format to file (.ligra)");
        AddHelpLine('w', "file", "make output weighted (.wel|.wsg)");
        AddHelpLine('x', "file", "output new reordered labels to file list (.so)");
        AddHelpLine('q', "file",
                    "output new reordered labels to file serialized (.lo)");
    }

    void HandleArg(signed char opt, char *opt_arg) override
    {
        switch (opt)
        {
        case 'b':
            out_sg_ = true;
            out_filename_ = std::string(opt_arg);
            break;
        case 'p':
            out_mtx_ = true;
            out_filename_ = std::string(opt_arg);
            break;
        case 'y':
            out_ligra_ = true;
            out_filename_ = std::string(opt_arg);
            break;
        case 'V':
            out_structures_ = true;
            out_filename_ = std::string(opt_arg);
            break;
        case 'x':
            out_label_so_ = true;
            label_out_filename_ = std::string(opt_arg);
            break;
        case 'q':
            out_label_lo_ = true;
            label_out_filename_ = std::string(opt_arg);
            break;
        case 'e':
            out_el_ = true;
            out_filename_ = std::string(opt_arg);
            break;
        case 'w':
            out_weighted_ = true;
            break;
        default:
            CLBase::HandleArg(opt, opt_arg);
        }
    }

    std::string out_filename() const
    {
        return out_filename_;
    }
    std::string label_out_filename() const
    {
        return label_out_filename_;
    }
    bool out_weighted() const
    {
        return out_weighted_;
    }
    bool out_structures() const
    {
        return out_structures_;
    }
    bool out_el() const
    {
        return out_el_;
    }
    bool out_label_so() const
    {
        return out_label_so_;
    }
    bool out_label_lo() const
    {
        return out_label_lo_;
    }
    bool out_sg() const
    {
        return out_sg_;
    }
    bool out_mtx() const
    {
        return out_mtx_;
    }
    bool out_ligra() const
    {
        return out_ligra_;
    }
};

#endif // COMMAND_LINE_H_
