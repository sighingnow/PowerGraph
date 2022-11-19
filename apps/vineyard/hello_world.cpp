/*
 * Copyright (c) 2009 Carnegie Mellon University.
 *     All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing,
 *  software distributed under the License is distributed on an "AS
 *  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 *  express or implied.  See the License for the specific language
 *  governing permissions and limitations under the License.
 *
 *
 */

#include <graphlab.hpp>

#include <vineyard/client/client.h>
#include <vineyard/client/ds/blob.h>

// The vertex data is just the pagerank value (a float)
typedef int64_t vertex_data_type;

// There is no edge data in the pagerank application
typedef double edge_data_type;

// The graph type is determined by the vertex and edge data types
typedef graphlab::distributed_graph<vertex_data_type, edge_data_type>
    graph_type;

int64_t get_micro_timestamp() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

void compute_time_diff(std::string const &tag, int64_t micro_t1,
                       int64_t micro_t2) {
  std::cout << "time elapsed: " << std::setw(10) << (micro_t2 - micro_t1)
            << " us" << std::endl;
}

void compute_time_diff(std::string const &tag, int64_t micro_t1) {
  std::cout << "time elapsed: " << std::setw(10)
            << (get_micro_timestamp() - micro_t1) << " us" << std::endl;
}

/*
 * A simple function used by graph.transform_vertices(init_vertex);
 * to initialize the vertex data.
 */
void init_vertex(graph_type::vertex_type &vertex) {
  vertex.data() = vertex.id();
}

/* We want to save the final graph so we define a write which will be
 * used in graph.save("path/prefix", concomp_writer()) to save the graph.
 */
struct vineyard_concomp_writer {
  int64_t *src_data;
  int64_t *dst_data;
  double *edge_data;
  size_t edge_index = 0;
  vineyard_concomp_writer(int64_t *src_data, int64_t *dst_data,
                          double *edge_data)
      : src_data(src_data), dst_data(dst_data), edge_data(edge_data) {}
  std::string save_vertex(graph_type::vertex_type v) { return ""; }
  std::string save_edge(graph_type::edge_type e) {
    src_data[edge_index] = e.source().id();
    dst_data[edge_index] = e.source().id();
    edge_data[edge_index] = e.data();
    edge_index += 1;
    return "";
  }
}; // end of concomp writer

struct file_concomp_writer {
  std::string save_vertex(graph_type::vertex_type v) { return ""; }
  std::string save_edge(graph_type::edge_type e) {
    return std::to_string(e.source().id()) + "," +
           std::to_string(e.target().id()) + "," + std::to_string(e.data());
  }
}; // end of concomp writer

int main(int argc, char **argv) {
  // Initialize control plain using mpi
  graphlab::mpi_tools::init(argc, argv);
  graphlab::distributed_control dc;
  global_logger().set_log_level(LOG_INFO);

  // Parse command line options -----------------------------------------------
  graphlab::command_line_options clopts("PageRank algorithm.");
  std::string vineyard_socket;
  std::string kind;
  std::string base;

  std::string graph_dir;
  std::string format = "csv";
  clopts.attach_option("graph", graph_dir, "The graph file. Required ");
  clopts.add_positional("graph");
  clopts.attach_option("vineyard_socket", vineyard_socket,
                       "The graph file format");
  clopts.attach_option("kind", kind, "The graph file format");
  clopts.attach_option("base", base, "The graph file format");
  clopts.attach_option("format", format, "The graph file format");
  if (!clopts.parse(argc, argv)) {
    dc.cout() << "Error in parsing command line arguments." << std::endl;
    return EXIT_FAILURE;
  }
  if (graph_dir == "") {
    dc.cout() << "Graph not specified. Cannot continue";
    return EXIT_FAILURE;
  }

  // connect ot vineyard
  vineyard::Client client;
  VINEYARD_CHECK_OK(client.Connect(vineyard_socket));

  // Build the graph ----------------------------------------------------------
  graph_type graph(dc, clopts);
  dc.cout() << "Loading graph in format: " << format << std::endl;

  auto start_time = get_micro_timestamp();
  graph.load_format(graph_dir, format);
  // must call finalize before querying the graph
  graph.finalize();
  compute_time_diff("loading graph from csv", start_time);

  graph.transform_vertices(init_vertex);

  dc.cout() << "#vertices: " << graph.num_vertices()
            << " #edges:" << graph.num_edges() << std::endl;

  if (kind == "csv") {
    auto start_time = get_micro_timestamp();
    graph.save("output" + graph_dir + ".txt", file_concomp_writer(),
               false,  // do not gzip
               true,   // save vertices
               false); // do not save edges
    compute_time_diff("save edges to csv", start_time);
  } else {
    std::unique_ptr<vineyard::BlobWriter> src_writer;
    std::unique_ptr<vineyard::BlobWriter> dst_writer;
    std::unique_ptr<vineyard::BlobWriter> data_writer;
    VINEYARD_CHECK_OK(
        client.CreateBlob(graph.num_edges() * sizeof(int64_t), src_writer));
    VINEYARD_CHECK_OK(
        client.CreateBlob(graph.num_edges() * sizeof(int64_t), dst_writer));
    VINEYARD_CHECK_OK(
        client.CreateBlob(graph.num_edges() * sizeof(double), data_writer));

    auto start_time = get_micro_timestamp();
    graph.save("output" + graph_dir + ".txt",
               vineyard_concomp_writer(
                   reinterpret_cast<int64_t *>(src_writer->data()),
                   reinterpret_cast<int64_t *>(dst_writer->data()),
                   reinterpret_cast<double *>(data_writer->data())),
               false,  // do not gzip
               true,   // save vertices
               false); // do not save edges
    compute_time_diff("save edges to vineyard", start_time);

    VINEYARD_CHECK_OK(src_writer->Abort(client));
    VINEYARD_CHECK_OK(dst_writer->Abort(client));
    VINEYARD_CHECK_OK(data_writer->Abort(client));
  }
}
