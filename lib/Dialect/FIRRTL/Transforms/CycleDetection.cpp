//===- CycleDetection.cpp - FIRRTL Cycle Detection Utility ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements cycle detection for FIRRTL using FieldRef nodes.
//
//===----------------------------------------------------------------------===//

#include "CycleDetection.h"
#include "circt/Support/FieldRef.h"
#include "llvm/ADT/STLExtras.h"

using namespace circt;
using namespace mlir;

LogicalResult CycleDetector::detectCycles(
    CycleCallback cycleCallback,
    llvm::function_ref<Location(FieldRef)> getNodeLoc) {

  // Build indexed representation for efficient traversal.
  // FieldRefGraph already uses indexed representation with insertion order
  // preserved, so we can use it directly!
  nodeToIndex.clear();
  indexToNode.clear();

  llvm::ArrayRef<FieldRef> nodes = graph.getNodes();

  unsigned idx = 0;
  for (const auto &node : nodes) {
    nodeToIndex[node] = idx++;
    indexToNode.push_back(node);
  }

  return dfsTraverseInternal(nodes, cycleCallback, getNodeLoc);
}

LogicalResult CycleDetector::dfsTraverseInternal(
    llvm::ArrayRef<FieldRef> nodes,
    const CycleCallback &cycleCallback,
    llvm::function_ref<Location(FieldRef)> getNodeLoc) {

  auto numNodes = graph.getNumNodes();
  llvm::SmallVector<bool> onStack(numNodes, false);
  llvm::SmallVector<unsigned> dfsStack;
  llvm::DenseSet<unsigned> visited;

  auto hasCycle = [&](unsigned rootNode) -> LogicalResult {
    if (visited.contains(rootNode))
      return success();

    dfsStack.push_back(rootNode);

    while (!dfsStack.empty()) {
      auto currentNodeIdx = dfsStack.back();

      if (!visited.contains(currentNodeIdx)) {
        visited.insert(currentNodeIdx);
        onStack[currentNodeIdx] = true;
      } else {
        onStack[currentNodeIdx] = false;
        dfsStack.pop_back();
        continue;
      }

      FieldRef currentNode = indexToNode[currentNodeIdx];
      auto successors = graph.getSuccessors(currentNode);
      if (successors.empty()) {
        onStack[currentNodeIdx] = false;
        dfsStack.pop_back();
        continue;
      }

      for (auto neighbor : successors) {
        auto neighborIt = nodeToIndex.find(neighbor);
        if (neighborIt == nodeToIndex.end())
          continue;

        unsigned neighborIdx = neighborIt->second;

        if (!visited.contains(neighborIdx)) {
          dfsStack.push_back(neighborIdx);
        } else if (onStack[neighborIdx]) {
          // Cycle detected! Construct the path.
          CycleDetector::Path cyclicPath;
          auto loopNode = neighborIdx;

          // Build the cycle path by following on-stack nodes
          do {
            FieldRef loopNodeVal = indexToNode[loopNode];
            auto successors = graph.getSuccessors(loopNodeVal);
            if (successors.empty())
              break;

            // Find the next node in the cycle that's on the stack
            auto *nextIt = llvm::find_if(successors, [&](FieldRef n) {
              auto idx = nodeToIndex.find(n);
              return idx != nodeToIndex.end() && onStack[idx->second];
            });

            if (nextIt == successors.end())
              break;

            cyclicPath.push_back(loopNodeVal);
            loopNode = nodeToIndex[*nextIt];
          } while (loopNode != neighborIdx);

          // Invoke the callback if provided
          if (cycleCallback) {
            Location loc = getNodeLoc ? getNodeLoc(indexToNode[neighborIdx])
                                      : UnknownLoc::get(nullptr);
            if (failed(cycleCallback(cyclicPath, loc)))
              return failure();
          }

          return failure();
        }
      }
    }
    return success();
  };

  // Visit all nodes to ensure we find cycles in disconnected components
  for (unsigned nodeIdx = 0; nodeIdx < numNodes; ++nodeIdx) {
    if (failed(hasCycle(nodeIdx)))
      return failure();
  }

  return success();
}

LogicalResult circt::dfsFromNode(const FieldRefGraph &graph, FieldRef startNode,
                                 llvm::DenseSet<FieldRef> &visited,
                                 llvm::function_ref<void(FieldRef, FieldRef)>
                                     onEdgeTraversal) {

  llvm::SmallVector<FieldRef> stack;
  stack.push_back(startNode);

  while (!stack.empty()) {
    FieldRef current = stack.back();
    stack.pop_back();

    if (visited.contains(current))
      continue;

    visited.insert(current);

    auto successors = graph.getSuccessors(current);
    for (auto neighbor : successors) {
      if (onEdgeTraversal)
        onEdgeTraversal(current, neighbor);

      if (!visited.contains(neighbor))
        stack.push_back(neighbor);
    }
  }

  return success();
}

PortPaths circt::recordPortPaths(firrtl::FModuleOp module,
                                    const FieldRefGraph &graph) {
  using namespace firrtl;

  PortPaths portPaths;

  // Iterate over the module's ports (BlockArguments)
  for (auto port : module.getArguments()) {
    unsigned portNum = port.getArgNumber();

    // Only process output ports
    if (module.getPortDirection(portNum) != Direction::Out)
      continue;

    // Create FieldRef for this port (fieldID 0 for the base value)
    FieldRef outputPort(port, 0);

    // Only process if this port exists in the graph
    if (!graph.hasNode(outputPort))
      continue;

    // Perform DFS from this output port to find reachable input ports
    llvm::DenseSet<FieldRef> visited;
    (void)dfsFromNode(graph, outputPort, visited,
                      [&](const FieldRef &from, const FieldRef &to) {
                        // Check if 'to' is an input port of this module
                        auto toArg = dyn_cast<BlockArgument>(to.getValue());
                        if (!toArg)
                          return;

                        unsigned toPortNum = toArg.getArgNumber();
                        if (module.getPortDirection(toPortNum) != Direction::In)
                          return;

                        // Only record if this input port exists in the graph
                        if (!graph.hasNode(to))
                          return;

                        // Record that output depends on this input
                        portPaths[outputPort].insert(to);
                      });
  }

  return portPaths;
}

