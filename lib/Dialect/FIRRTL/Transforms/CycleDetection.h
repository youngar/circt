//===- CycleDetection.h - Generic Cycle Detection Utility ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a generic cycle detection utility that can be reused
// across different passes. It provides an IR-agnostic engine for detecting
// strongly connected components (SCCs) and cycles in directed graphs.
//
//===----------------------------------------------------------------------===//

#ifndef LIB_DIALECT_FIRRTL_TRANSFORMS_CYCLEDETECTION_H
#define LIB_DIALECT_FIRRTL_TRANSFORMS_CYCLEDETECTION_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Support/FieldRef.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Location.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <functional>

namespace circt {

/// Map from output ports to the set of input ports they depend on.
/// This represents "summary paths" through a module - the inter-module
/// connectivity that allows hierarchical cycle detection.
///
/// Example: If a module has `output y` that depends on `input x1` and `input
/// x2`, the PortPaths would contain: `y -> {x1, x2}`.
using PortPaths = DenseMap<FieldRef, llvm::DenseSet<FieldRef>>;

using ModulePortPaths = DenseMap<FModuleLike, PortPaths>;

//===----------------------------------------------------------------------===//
// FieldRefGraph - Deterministic Graph Representation
//===----------------------------------------------------------------------===//

/// A directed graph of FieldRef nodes with indexed representation.
/// Nodes are assigned stable integer IDs in insertion order, enabling:
/// - O(1) indexed access for equivalence class operations
/// - Deterministic iteration order for reproducible cycle detection
/// - Efficient storage and traversal using integer indices
///
/// The graph stores edges as an adjacency list using node indices for
/// efficient lookups and compatibility with algorithms like EquivalenceClasses.
class FieldRefGraph {
public:
  FieldRefGraph() = default;

  /// Get or create a node index for the given FieldRef.
  /// Returns the existing index if the node already exists, or creates a new
  /// index if it doesn't. Indices are assigned in insertion order.
  unsigned getOrAddNode(FieldRef node) {
    auto it = nodeToIndex.find(node);
    if (it != nodeToIndex.end())
      return it->second;

    unsigned idx = indexToNode.size();
    nodeToIndex[node] = idx;
    indexToNode.push_back(node);
    adjacencyList.push_back({});
    return idx;
  }

  /// Add a directed edge from source to destination using node indices.
  void addEdge(unsigned srcIdx, unsigned dstIdx) {
    assert(srcIdx < adjacencyList.size() && "Invalid source index");
    assert(dstIdx < adjacencyList.size() && "Invalid destination index");
    adjacencyList[srcIdx].push_back(dstIdx);
  }

  /// Add a directed edge from src to dst.
  /// Convenience method that handles FieldRef lookup/creation.
  void addEdge(FieldRef src, FieldRef dst) {
    unsigned srcIdx = getOrAddNode(src);
    unsigned dstIdx = getOrAddNode(dst);
    addEdge(srcIdx, dstIdx);
  }

  /// Ensure a node exists in the graph, even if it has no outgoing edges.
  void ensureNodeExists(FieldRef node) { (void)getOrAddNode(node); }

  /// Get all nodes in the graph in insertion order.
  llvm::ArrayRef<FieldRef> getNodes() const { return indexToNode; }

  /// Get the FieldRef for a given node index.
  FieldRef getNode(unsigned idx) const {
    assert(idx < indexToNode.size() && "Invalid node index");
    return indexToNode[idx];
  }

  /// Get the index for a given FieldRef, or None if it doesn't exist.
  std::optional<unsigned> getNodeIndex(FieldRef node) const {
    auto it = nodeToIndex.find(node);
    if (it == nodeToIndex.end())
      return std::nullopt;
    return it->second;
  }

  /// Get the successors of a node by index (returns indices).
  llvm::ArrayRef<unsigned> getSuccessors(unsigned nodeIdx) const {
    assert(nodeIdx < adjacencyList.size() && "Invalid node index");
    return adjacencyList[nodeIdx];
  }

  /// Get the successors of a node by FieldRef (returns FieldRefs).
  llvm::SmallVector<FieldRef> getSuccessors(FieldRef node) const {
    auto idx = getNodeIndex(node);
    if (!idx)
      return {};

    llvm::SmallVector<FieldRef> result;
    for (unsigned succIdx : adjacencyList[*idx])
      result.push_back(indexToNode[succIdx]);
    return result;
  }

  /// Get the number of nodes in the graph.
  size_t getNumNodes() const { return indexToNode.size(); }

  /// Check if a node exists in the graph.
  bool hasNode(FieldRef node) const { return nodeToIndex.count(node); }

  /// Clear all nodes and edges.
  void clear() {
    nodeToIndex.clear();
    indexToNode.clear();
    adjacencyList.clear();
  }

private:
  /// Map from FieldRef to its assigned node index.
  DenseMap<FieldRef, unsigned> nodeToIndex;

  /// Map from node index to FieldRef (preserves insertion order).
  llvm::SmallVector<FieldRef> indexToNode;

  /// Adjacency list: adjacencyList[srcIdx] = {dstIdx1, dstIdx2, ...}
  /// Stores directed edges using node indices for O(1) access.
  llvm::SmallVector<llvm::SmallVector<unsigned>> adjacencyList;
};

/// Cycle detection engine for FIRRTL that operates on FieldRef nodes.
/// This class provides the core cycle detection algorithm with support for
/// hierarchical module analysis via port connectivity summaries.
///
/// Hierarchical Analysis Approach:
/// --------------------------------
/// 1. **Bottom-up traversal:** Process modules in post-order (children before
/// parents)
/// 2. **Intra-module analysis:** Build local graph and detect cycles within
/// each module
/// 3. **Port path recording:** For each module, record which output ports
/// depend on
///    which input ports using recordPortPaths()
/// 4. **Inter-module propagation:** When processing a parent module that
/// instantiates
///    a child, use addModulePortPaths() to add "bypass" edges representing
///    dataflow through the child module instance
///
/// This approach allows cycle detection across module boundaries without
/// flattening the entire design hierarchy.
///
/// All nodes in the graph are represented as FieldRef, which can precisely
/// identify specific fields within aggregate types (bundles, vectors, domains,
/// etc.).
class CycleDetector {
public:
  /// Represents a path in the graph as a sequence of FieldRef nodes.
  using Path = llvm::SmallVector<FieldRef, 16>;

  /// Callback function type for reporting cycles.
  /// The callback receives the cyclic path and a suggested location for error
  /// reporting. Returns failure if the cycle should cause the overall analysis
  /// to fail.
  using CycleCallback = std::function<mlir::LogicalResult(
      const Path &cyclicPath, mlir::Location loc)>;

  /// Constructor takes a reference to a pre-built graph.
  /// The graph must outlive the CycleDetector instance.
  explicit CycleDetector(const FieldRefGraph &g) : graph(g) {}

  /// Detect cycles in the graph using DFS.
  /// If a cycle is found and cycleCallback is provided, it will be invoked
  /// with the cycle path.
  /// Returns success if no cycles are found, or if all cycle callbacks succeed.
  mlir::LogicalResult detectCycles(
      CycleCallback cycleCallback = nullptr,
      llvm::function_ref<mlir::Location(FieldRef)> getNodeLoc = nullptr);

private:
  /// Internal DFS implementation for cycle detection.
  mlir::LogicalResult
  dfsTraverseInternal(llvm::ArrayRef<FieldRef> nodes,
                      const CycleCallback &cycleCallback,
                      llvm::function_ref<mlir::Location(FieldRef)> getNodeLoc);

  /// Reference to the externally-managed graph.
  const FieldRefGraph &graph;

  /// Mapping for indexed access (used during DFS traversal).
  DenseMap<FieldRef, unsigned> nodeToIndex;
  llvm::SmallVector<FieldRef> indexToNode;
};

//===----------------------------------------------------------------------===//
// Graph Helper Functions
//===----------------------------------------------------------------------===//

/// Perform DFS traversal on a graph starting from a specific node.
/// Records which nodes are visited and optionally invokes a callback on each
/// edge.
mlir::LogicalResult dfsFromNode(
    const FieldRefGraph &graph, FieldRef startNode,
    llvm::DenseSet<FieldRef> &visited,
    llvm::function_ref<void(FieldRef, FieldRef)> onEdgeTraversal = nullptr);

/// Record port-to-port connectivity for a FIRRTL module's graph.
/// For each output port present in the graph, performs DFS to find all input
/// ports it depends on. This creates a "summary" of the module's internal
/// connectivity that can be used by parent modules for hierarchical analysis.
///
/// This function only considers ports that are actually present as nodes in
/// the graph. For example, if a module has domain and non-domain ports, but
/// the graph only contains domain-related nodes, only domain port paths will
/// be recorded.
///
/// \param module The FIRRTL module whose ports to analyze
/// \param graph The graph representing the module's internal connectivity
/// \return A map from output ports to the set of input ports they depend on
PortPaths recordPortPaths(firrtl::FModuleOp module, const FieldRefGraph &graph);

} // namespace circt

#endif // LIB_DIALECT_FIRRTL_TRANSFORMS_CYCLEDETECTION_H
