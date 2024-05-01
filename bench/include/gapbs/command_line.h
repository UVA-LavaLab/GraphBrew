// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef COMMAND_LINE_H_
#define COMMAND_LINE_H_

#include <getopt.h>

#include "util.h"
#include <algorithm>
#include <cinttypes>
#include <cstdlib>
#include <iostream>
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

class CLBase {
protected:
  int argc_;
  char **argv_;
  std::string name_;
  std::string get_args_ = "f:g:hk:su:m:o:zj:";
  std::vector<std::string> help_strings_;
  std::vector<std::pair<ReorderingAlgo, std::string>> reorder_options_;

  int scale_ = -1;
  int degree_ = 16;
  std::string filename_ = "";
  bool symmetrize_ = false;
  bool uniform_ = false;
  bool in_place_ = false;
  bool use_out_degree_ = true;
  std::pair<std::string, int> segments_ = {
      "", 1}; // Label and number of segments as a pair

  void AddHelpLine(char opt, std::string opt_arg, std::string text,
                   std::string def = "") {
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
      : argc_(argc), argv_(argv), name_(name) {
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
                "          [example]-r 3 "
                "-r 2 -r 10:mapping.label",
                "optional");
    AddHelpLine('z', "indegree","use indegree for ordering [Degree Based Orderings]", "false");
    AddHelpLine('j', "segments", "number of segments for the graph", "1");
  }

  bool ParseArgs() {
    signed char c_opt;
    extern char *optarg; // from and for getopt
    while ((c_opt = getopt(argc_, argv_, get_args_.c_str())) != -1) {
      HandleArg(c_opt, optarg);
    }
    if ((filename_ == "") && (scale_ == -1)) {
      std::cout << "No graph input specified. (Use -h for help)" << std::endl;
      return false;
    }
    if (scale_ != -1)
      symmetrize_ = true;
    return true;
  }

  void virtual HandleArg(signed char opt, char *opt_arg) {
    switch (opt) {
    case 'f':
      filename_ = std::string(opt_arg);
      segments_.first = filename_; // Set label to filename
      break;
    case 'g':
      scale_ = atoi(opt_arg);
      segments_.first = "kronecker" + std::to_string(scale_);
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
      segments_.first = "uniform-random" + std::to_string(scale_);
      break;
    case 'm':
      in_place_ = true;
      break;
    case 'o': {
      std::string arg(opt_arg);
      size_t pos = arg.find(':');
      ReorderingAlgo algo =
          static_cast<ReorderingAlgo>(std::stoi(arg.substr(0, pos)));
      std::string param = (pos != std::string::npos) ? arg.substr(pos + 1) : "";
      reorder_options_.emplace_back(algo, param);
    } break;
    case 'j':
      segments_.second = atoi(opt_arg);
      if (segments_.first.empty()) {
        segments_.first = "default";
      }
      break;
    }
  }

  void PrintUsage() {
    std::cout << name_ << std::endl;
    // std::sort(help_strings_.begin(), help_strings_.end());
    for (std::string h : help_strings_)
      std::cout << h << std::endl;
    std::exit(0);
  }

  int scale() const { return scale_; }
  int degree() const { return degree_; }
  std::string filename() const { return filename_; }
  bool symmetrize() const { return symmetrize_; }
  bool uniform() const { return uniform_; }
  bool in_place() const { return in_place_; }
  bool use_out_degree() const { return use_out_degree_; }
  const std::vector<std::pair<ReorderingAlgo, std::string>> &
  reorder_options() const {
    return reorder_options_;
  }
  std::pair<std::string, int> const segments()  {
    return segments_;
  }
};

class CLApp : public CLBase {
  bool do_analysis_ = false;
  int num_trials_ = 16;
  int64_t start_vertex_ = -1;
  bool do_verify_ = false;
  bool enable_logging_ = false;

public:
  CLApp(int argc, char **argv, std::string name) : CLBase(argc, argv, name) {
    get_args_ += "an:r:vl";
    AddHelpLine('a', "", "output analysis of last run", "false");
    AddHelpLine('n', "n", "perform n trials", std::to_string(num_trials_));
    AddHelpLine('r', "node", "start from node r", "rand");
    AddHelpLine('v', "", "verify the output of each run", "false");
    AddHelpLine('l', "", "log performance within each trial", "false");
  }

  void HandleArg(signed char opt, char *opt_arg) override {
    switch (opt) {
    case 'a':
      do_analysis_ = true;
      break;
    case 'n':
      num_trials_ = atoi(opt_arg);
      break;
    case 'r':
      start_vertex_ = atol(opt_arg);
      break;
    case 'v':
      do_verify_ = true;
      break;
    case 'l':
      enable_logging_ = true;
      break;
    default:
      CLBase::HandleArg(opt, opt_arg);
    }
  }

  bool do_analysis() const { return do_analysis_; }
  int num_trials() const { return num_trials_; }
  int64_t start_vertex() const { return start_vertex_; }
  bool do_verify() const { return do_verify_; }
  bool logging_en() const { return enable_logging_; }
};

class CLIterApp : public CLApp {
  int num_iters_;

public:
  CLIterApp(int argc, char **argv, std::string name, int num_iters)
      : CLApp(argc, argv, name), num_iters_(num_iters) {
    get_args_ += "i:";
    AddHelpLine('i', "i", "perform i iterations", std::to_string(num_iters_));
  }

  void HandleArg(signed char opt, char *opt_arg) override {
    switch (opt) {
    case 'i':
      num_iters_ = atoi(opt_arg);
      break;
    default:
      CLApp::HandleArg(opt, opt_arg);
    }
  }

  int num_iters() const { return num_iters_; }
};

class CLPageRank : public CLApp {
  int max_iters_;
  double tolerance_;

public:
  CLPageRank(int argc, char **argv, std::string name, double tolerance,
             int max_iters)
      : CLApp(argc, argv, name), max_iters_(max_iters), tolerance_(tolerance) {
    get_args_ += "i:t:";
    AddHelpLine('i', "i", "perform at most i iterations",
                std::to_string(max_iters_));
    AddHelpLine('t', "t", "use tolerance t", std::to_string(tolerance_));
  }

  void HandleArg(signed char opt, char *opt_arg) override {
    switch (opt) {
    case 'i':
      max_iters_ = atoi(opt_arg);
      break;
    case 't':
      tolerance_ = std::stod(opt_arg);
      break;
    default:
      CLApp::HandleArg(opt, opt_arg);
    }
  }

  int max_iters() const { return max_iters_; }
  double tolerance() const { return tolerance_; }
};

template <typename WeightT_> class CLDelta : public CLApp {
  WeightT_ delta_ = 1;

public:
  CLDelta(int argc, char **argv, std::string name) : CLApp(argc, argv, name) {
    get_args_ += "d:";
    AddHelpLine('d', "d", "delta parameter", std::to_string(delta_));
  }

  void HandleArg(signed char opt, char *opt_arg) override {
    switch (opt) {
    case 'd':
      if (std::is_floating_point<WeightT_>::value)
        delta_ = static_cast<WeightT_>(atof(opt_arg));
      else
        delta_ = static_cast<WeightT_>(atol(opt_arg));
      break;
    default:
      CLApp::HandleArg(opt, opt_arg);
    }
  }

  WeightT_ delta() const { return delta_; }
};

class CLConvert : public CLBase {
  std::string out_filename_ = "";
  std::string label_out_filename_ = "";
  bool out_weighted_ = false;
  bool out_el_ = false;
  bool out_sg_ = false;
  bool out_label_so_ = false;
  bool out_label_lo_ = false;

public:
  CLConvert(int argc, char **argv, std::string name)
      : CLBase(argc, argv, name) {
    get_args_ += "e:b:x:q:w";
    AddHelpLine('b', "file", "output serialized graph to file (.sg)");
    AddHelpLine('e', "file", "output edge list to file (.el)");
    AddHelpLine('w', "file", "make output weighted (.wel|.wsg)");
    AddHelpLine('x', "file", "output new reordered labels to file list (.so)");
    AddHelpLine('q', "file", "output new reordered labels to file serialized (.lo)");
  }

  void HandleArg(signed char opt, char *opt_arg) override {
    switch (opt) {
    case 'b':
      out_sg_ = true;
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

  std::string out_filename() const { return out_filename_; }
  std::string label_out_filename() const { return label_out_filename_; }
  bool out_weighted() const { return out_weighted_; }
  bool out_el() const { return out_el_; }
  bool out_label_so() const { return out_label_so_; }
  bool out_label_lo() const { return out_label_lo_; }
  bool out_sg() const { return out_sg_; }
};

#endif // COMMAND_LINE_H_
