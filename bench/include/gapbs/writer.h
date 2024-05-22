// Copyright (c) 2015, The Regents of the University of California (Regents)
// See LICENSE.txt for license details

#ifndef WRITER_H_
#define WRITER_H_

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <type_traits>

#include "graph.h"

/*
   GAP Benchmark Suite
   Class:  Writer
   Author: Scott Beamer

   Given filename and graph, writes out the graph to storage
   - Should use WriteGraph(filename, serialized)
   - If serialized, will write out as serialized graph, otherwise, as edgelist
 */

template <typename NodeID_, typename DestID_ = NodeID_, typename WeightT_= NodeID_> class WriterBase {
public:
explicit WriterBase(CSRGraph<NodeID_, DestID_> &g) : g_(g) {
}

void WriteEL(std::fstream &out) {
  for (NodeID_ u = 0; u < g_.num_nodes(); u++) {
    for (DestID_ v : g_.out_neigh(u))
      out << u << " " << v << std::endl;
  }
}

void WriteMTX(std::fstream &out) {

  std::string field_type;
  if (g_.is_weighted()) {
    if (std::is_integral<WeightT_>::value) {
      field_type = "integer";
    } else if (std::is_floating_point<WeightT_>::value) {
      field_type = "real";
    } else {
      std::cerr << "Unsupported weight type." << std::endl;
      return;
    }
  } else {
    field_type = "pattern";
  }

  // Write the header
  out << "%%MatrixMarket matrix coordinate "
      << field_type << " general" << std::endl;
  out << "%" << std::endl;
  out << static_cast<long long>(g_.num_nodes()) << " "
      << static_cast<long long>(g_.num_nodes()) << " "
      << static_cast<long long>(g_.num_edges()) << std::endl;

  // Write the edges
  for (NodeID_ u = 0; u < g_.num_nodes(); u++) {
    for (DestID_ v : g_.out_neigh(u)) {
      if (g_.is_weighted()) {
        out << u + 1 << " " << static_cast<NodeWeight<NodeID_, WeightT_> >(v).v + 1 << " " << static_cast<NodeWeight<NodeID_, WeightT_> >(v).w << std::endl; // 1-based indexing
      } else {
        out << u + 1 << " " << v + 1 << std::endl; // 1-based indexing
      }
    }
  }
}

void WriteSerializedGraph(std::fstream &out) {
  if (!std::is_same<NodeID_, SGID>::value) {
    std::cout << "serialized graphs only allowed for 32b IDs" << std::endl;
    std::exit(-4);
  }
  if (!std::is_same<DestID_, NodeID_>::value &&
      !std::is_same<DestID_, NodeWeight<NodeID_, SGID> >::value) {
    std::cout << ".wsg only allowed for int32_t weights" << std::endl;
    std::exit(-8);
  }
  bool directed = g_.directed();
  SGOffset num_nodes = g_.num_nodes();
  SGOffset edges_to_write = g_.num_edges_directed();
  std::streamsize index_bytes = (num_nodes + 1) * sizeof(SGOffset);
  std::streamsize neigh_bytes;
  std::streamsize ids_bytes = num_nodes * sizeof(NodeID_);

  if (std::is_same<DestID_, NodeID_>::value)
    neigh_bytes = edges_to_write * sizeof(SGID);
  else
    neigh_bytes = edges_to_write * sizeof(NodeWeight<NodeID_, SGID>);
  out.write(reinterpret_cast<char *>(&directed), sizeof(bool));
  out.write(reinterpret_cast<char *>(&edges_to_write), sizeof(SGOffset));
  out.write(reinterpret_cast<char *>(&num_nodes), sizeof(SGOffset));
  pvector<SGOffset> offsets = g_.VertexOffsets(false);
  out.write(reinterpret_cast<char *>(offsets.data()), index_bytes);
  out.write(reinterpret_cast<char *>(g_.out_neigh(0).begin()), neigh_bytes);
  if (directed) {
    offsets = g_.VertexOffsets(true);
    out.write(reinterpret_cast<char *>(offsets.data()), index_bytes);
    out.write(reinterpret_cast<char *>(g_.in_neigh(0).begin()), neigh_bytes);
  }
  // Write original IDs
  NodeID_ *temp_org_ids = g_.get_org_ids();
  out.write(reinterpret_cast<char *>(temp_org_ids), ids_bytes);

  out.flush();
  out.close();
}

void WriteSerializedLabels(std::fstream &out) {
  if (!std::is_same<NodeID_, SGID>::value) {
    std::cout << "serialized graphs only allowed for 32b IDs" << std::endl;
    std::exit(-4);
  }
  SGOffset num_nodes = g_.num_nodes();
  std::streamsize ids_bytes = num_nodes * sizeof(NodeID_);
  out.write(reinterpret_cast<char *>(g_.get_org_ids()), ids_bytes);
}

void WriteListLabels(std::fstream &out) {
  NodeID_ *source_array = g_.get_org_ids();
  for (NodeID_ u = 0; u < g_.num_nodes(); u++) {
    out << source_array[u] << std::endl;
  }
}

void WriteGraph(std::string filename, bool serialized = false, bool mtxed = false, bool edgelisted = false) {
  if (filename == "") {
    std::cout << "No output filename given (Use -h for help)" << std::endl;
    std::exit(-8);
  }
  std::fstream file(filename, std::ios::out | std::ios::binary);
  if (!file) {
    std::cout << "Couldn't write to file " << filename << std::endl;
    std::exit(-5);
  }

  std::cout << "writing to file " << filename << std::endl;
  if (serialized)
    WriteSerializedGraph(file);
  if (edgelisted)
    WriteEL(file);
  if(mtxed)
    WriteMTX(file);

  file.close();
}

void WriteLabels(std::string filename, bool serialized = false, bool edgelisted = false) {
  if (filename == "") {
    std::cout << "No output filename given (Use -h for help)" << std::endl;
    std::exit(-8);
  }
  std::fstream file(filename, std::ios::out | std::ios::binary);
  if (!file) {
    std::cout << "Couldn't write to file " << filename << std::endl;
    std::exit(-5);
  }
  if (serialized)
    WriteSerializedLabels(file);
  if (edgelisted)
    WriteListLabels(file);

  file.close();
}

private:
CSRGraph<NodeID_, DestID_> &g_;
std::string filename_;
};

#endif // WRITER_H_
