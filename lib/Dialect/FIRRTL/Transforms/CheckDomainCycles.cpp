//===- CheckDomainCycles.cpp - Check FIRRTL Domain Cycles ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to detect cycles in FIRRTL domain connectivity.
// It checks for cycles in domain.define operations across the module hierarchy.
//
//===----------------------------------------------------------------------===//

#include "CycleDetection.h"
#include "CycleReporting.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/FieldRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "check-domain-cycles"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_CHECKDOMAINCYCLES
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Type Aliases
//===----------------------------------------------------------------------===//

namespace {

// Use the unified port paths type from CycleDetection.h
using DomainPathsMap = PortPaths;

/// Graph builder for domain connectivity.
/// This class constructs a directed graph representing domain flow through
/// domain.define operations and module instances.
class DomainGraphBuilder {
public:
  DomainGraphBuilder(FModuleOp module, InstanceGraph &instanceGraph,
                     const ModulePortPaths &otherModulePortPaths)
      : module(module), instanceGraph(instanceGraph),
        modulePortPaths(otherModulePortPaths) {}

  /// Build the domain connectivity graph for this module.
  void buildGraph(FieldRefGraph &graph);

  /// Get a human-readable name for a domain node (for error reporting).
  static std::string getNodeName(const FieldRef &node) {
    auto fieldName = getFieldName(node);
    return fieldName.first;
  }

  /// Check if a type is a DomainType.
  /// Note: Domains cannot be members of bundles, vectors, or other aggregates.
  /// They are a top-level type category.
  static bool isDomainType(FIRRTLBaseType type) {
    return isa<DomainType>(type);
  }

private:
  /// Process a domain.define operation to add connectivity.
  void processDomainDefine(DomainDefineOp define, FieldRefGraph &graph);

  /// Process a domain.subfield operation to track subfield access.
  void processDomainSubfield(DomainSubfieldOp subfield, FieldRefGraph &graph);

  /// Process a wire operation to add domain-typed wires as nodes.
  void processDomainWire(WireOp wire, FieldRefGraph &graph);

  /// Process an instance to add cross-module domain connectivity.
  void processInstance(InstanceOp inst, FieldRefGraph &graph);

  /// Process an instance choice operation (conservative analysis).
  void processInstanceChoice(InstanceChoiceOp inst, FieldRefGraph &graph);

  /// Add nodes for all output domain ports.
  void addOutputDomainPorts(FieldRefGraph &graph);

  FModuleOp module;
  InstanceGraph &instanceGraph;
  const llvm::DenseMap<FModuleLike, DomainPathsMap> &modulePortPaths;
};

//===----------------------------------------------------------------------===//
// DomainGraphBuilder Implementation
//===----------------------------------------------------------------------===//

void DomainGraphBuilder::addOutputDomainPorts(FieldRefGraph &graph) {
  // Add all output domain ports as initial nodes in the graph.
  for (auto port : module.getArguments()) {
    if (module.getPortDirection(port.getArgNumber()) != Direction::Out)
      continue;

    auto portType = port.getType();
    if (!isa<DomainType>(portType))
      continue;

    FieldRef node(port, 0);
    graph.ensureNodeExists(node);

    LLVM_DEBUG(llvm::dbgs()
               << "Added output domain port: " << getNodeName(node) << "\n");
  }
}

void DomainGraphBuilder::processDomainDefine(DomainDefineOp define,
                                             FieldRefGraph &graph) {
  // domain.define %dest, %src means %src drives %dest
  auto dest = define.getDest();
  auto src = define.getSrc();

  // getFieldRefFromValue will automatically walk through domain.subfield
  // operations and accumulate the fieldID, giving us the precise field
  // reference
  FieldRef destNode = getFieldRefFromValue(dest);
  FieldRef srcNode = getFieldRefFromValue(src);

  // Add edge from src to dest (src drives dest)
  graph.addEdge(srcNode, destNode);

  LLVM_DEBUG(llvm::dbgs() << "Added domain edge: " << getNodeName(srcNode)
                          << " (fieldID=" << srcNode.getFieldID() << ") -> "
                          << getNodeName(destNode)
                          << " (fieldID=" << destNode.getFieldID() << ")\n");
}

void DomainGraphBuilder::processDomainSubfield(DomainSubfieldOp subfield,
                                               FieldRefGraph &graph) {
  // Domain subfield extracts a specific field from a domain aggregate.
  // We need to create a mapping that tracks the identity of the subfield
  // result.
  //
  // For example:  %result = domain.subfield %bundle[field_index]
  // The result represents a specific field of the bundle domain.
  //
  // We create a FieldRef for the input's specific field using the DomainType's
  // field indexing, which maps to a fieldID in the aggregate.

  auto input = subfield.getInput();
  auto result = subfield.getResult();
  auto fieldIndex = subfield.getFieldIndex();

  // Get the base FieldRef for the input
  FieldRef inputBase = getFieldRefFromValue(input);

  // Calculate the fieldID for the specific field being accessed
  // DomainType uses sequential fieldIDs for its fields
  uint64_t fieldID = fieldIndex + 1; // fieldID 0 is the base, fields start at 1

  // Create FieldRef for the specific input field
  FieldRef inputField(inputBase.getValue(), inputBase.getFieldID() + fieldID);

  // The result itself is represented with fieldID 0 (it's the result value
  // itself) But we need to remember that this result logically represents the
  // input field So we create an identity edge: inputField drives result
  FieldRef resultRef(result, 0);

  // Add the identity edge showing the result is driven by the specific input
  // field
  graph.addEdge(inputField, resultRef);

  LLVM_DEBUG(llvm::dbgs() << "Added domain subfield: "
                          << getNodeName(inputField)
                          << " (fieldID=" << inputField.getFieldID() << ") -> "
                          << getNodeName(resultRef)
                          << " (fieldID=" << resultRef.getFieldID() << ")\n");
}

void DomainGraphBuilder::processDomainWire(WireOp wire, FieldRefGraph &graph) {
  // Add domain-typed wires as explicit nodes in the graph.
  // Wires are important intermediate nodes that appear in dataflow paths.
  //
  // Note: Domains are a top-level type and cannot be nested inside bundles,
  // vectors, or other aggregates. However, a DomainType itself can have fields
  // (domain bundles/rows), where each field is also a domain.

  auto wireType = wire.getResult().getType();

  if (!isDomainType(cast<FIRRTLBaseType>(wireType)))
    return; // Not a domain wire, skip

  auto domainType = cast<DomainType>(wireType);

  // Add the base wire node
  FieldRef wireRef(wire.getResult(), 0);
  graph.ensureNodeExists(wireRef);

  LLVM_DEBUG(llvm::dbgs() << "Added domain wire node: " << getNodeName(wireRef)
                          << "\n");

  // If this is a domain bundle/row (has fields), add nodes for each field
  // Each field in a domain aggregate is itself a domain
  if (domainType.getNumFields() > 0) {
    for (size_t i = 0; i < domainType.getNumFields(); ++i) {
      uint64_t fieldID = domainType.getFieldID(i);
      FieldRef fieldRef(wire.getResult(), fieldID);
      graph.ensureNodeExists(fieldRef);

      LLVM_DEBUG(llvm::dbgs()
                 << "Added domain wire field node: " << getNodeName(fieldRef)
                 << " (fieldID=" << fieldID << ")\n");
    }
  }
}

void DomainGraphBuilder::processInstance(InstanceOp inst,
                                         FieldRefGraph &graph) {
  auto refMod = inst.getReferencedModule<FModuleOp>(instanceGraph);
  if (!refMod)
    return;

  auto modulePaths = modulePortPaths.find(refMod);
  if (modulePaths == modulePortPaths.end())
    return;

  // Add bypass edges from module port paths.
  // For each output port -> {input ports} mapping in the child module,
  // add edges from the corresponding instance result ports to model
  // dataflow through the module instance.
  for (const auto &entry : modulePaths->second) {
    const FieldRef &modOutputPort = entry.first;
    const llvm::DenseSet<FieldRef> &modInputPorts = entry.second;

    // Map module output port to instance result port
    auto outArgNum =
        cast<BlockArgument>(modOutputPort.getValue()).getArgNumber();
    FieldRef instOutputPort(inst.getResult(outArgNum),
                            modOutputPort.getFieldID());

    for (const auto &modInputPort : modInputPorts) {
      // Map module input port to instance result port
      auto inArgNum =
          cast<BlockArgument>(modInputPort.getValue()).getArgNumber();
      FieldRef instInputPort(inst.getResult(inArgNum),
                             modInputPort.getFieldID());

      // Skip self-loops
      if (instInputPort != instOutputPort) {
        graph.addEdge(instInputPort, instOutputPort);
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Added instance paths through " << refMod.getName()
                          << "\n");
}

void DomainGraphBuilder::processInstanceChoice(InstanceChoiceOp inst,
                                               FieldRefGraph &graph) {
  // For InstanceChoiceOp, conservatively process all possible target modules.
  for (auto moduleName : inst.getReferencedModuleNamesAttr()) {
    auto moduleNameStr = cast<StringAttr>(moduleName);
    auto *node = instanceGraph.lookup(moduleNameStr);
    if (!node)
      continue;

    auto refMod = dyn_cast<FModuleOp>(*node->getModule());
    if (!refMod)
      continue;

    auto modulePaths = modulePortPaths.find(refMod);
    if (modulePaths == modulePortPaths.end())
      continue;

    // Add bypass edges from module port paths.
    // For each output port -> {input ports} mapping in the child module,
    // add edges from the corresponding instance result ports.
    for (const auto &entry : modulePaths->second) {
      const FieldRef &modOutputPort = entry.first;
      const llvm::DenseSet<FieldRef> &modInputPorts = entry.second;

      // Map module output port to instance result port
      auto outArgNum =
          cast<BlockArgument>(modOutputPort.getValue()).getArgNumber();
      FieldRef instOutputPort(inst.getResult(outArgNum),
                              modOutputPort.getFieldID());

      for (const auto &modInputPort : modInputPorts) {
        // Map module input port to instance result port
        auto inArgNum =
            cast<BlockArgument>(modInputPort.getValue()).getArgNumber();
        FieldRef instInputPort(inst.getResult(inArgNum),
                               modInputPort.getFieldID());

        // Skip self-loops
        if (instInputPort != instOutputPort) {
          graph.addEdge(instInputPort, instOutputPort);
        }
      }
    }
  }
}

void DomainGraphBuilder::buildGraph(FieldRefGraph &graph) {
  LLVM_DEBUG(llvm::dbgs() << "Building domain graph for module: "
                          << module.getName() << "\n");

  // Add output domain ports as initial nodes
  addOutputDomainPorts(graph);

  // Walk through all operations in the module
  module.walk([&](Operation *op) {
    if (auto define = dyn_cast<DomainDefineOp>(op)) {
      processDomainDefine(define, graph);
    } else if (auto subfield = dyn_cast<DomainSubfieldOp>(op)) {
      processDomainSubfield(subfield, graph);
    } else if (auto wire = dyn_cast<WireOp>(op)) {
      processDomainWire(wire, graph);
    } else if (auto inst = dyn_cast<InstanceOp>(op)) {
      processInstance(inst, graph);
    } else if (auto instChoice = dyn_cast<InstanceChoiceOp>(op)) {
      processInstanceChoice(instChoice, graph);
    }
  });
}

//===----------------------------------------------------------------------===//
// Domain Cycle Checker
//===----------------------------------------------------------------------===//

/// Domain cycle checker.
/// This class orchestrates domain cycle detection across a circuit by
/// building domain graphs for each module and detecting cycles.
class DomainCycleChecker {
public:
  explicit DomainCycleChecker(InstanceGraph &instanceGraph)
      : instanceGraph(instanceGraph) {}

  /// Check for domain cycles in the given module.
  /// Returns failure if a cycle is detected.
  LogicalResult checkModule(FModuleOp module);

  /// Run cycle detection across all modules in the instance graph.
  /// Processes modules bottom-up to propagate inter-module paths.
  LogicalResult checkCircuit();

private:
  InstanceGraph &instanceGraph;

  /// Maps module to domain paths between its ports.
  llvm::DenseMap<FModuleLike, DomainPathsMap> modulePortPaths;
};

LogicalResult DomainCycleChecker::checkModule(FModuleOp module) {
  // Build the domain connectivity graph
  FieldRefGraph fieldRefGraph;
  DomainGraphBuilder builder(module, instanceGraph, modulePortPaths);
  builder.buildGraph(fieldRefGraph);

  // Create cycle detector with the graph
  CycleDetector detector(fieldRefGraph);

  // Define callback for reporting cycles
  auto cycleCallback = [&](const CycleDetector::Path &cyclicPath,
                           Location loc) -> LogicalResult {
    // Path is already in FieldRef format, can use directly
    SmallVector<FieldRef, 16> path(cyclicPath.begin(), cyclicPath.end());
    firrtl::reportCycle(module.getLoc(), module.getName(), "domain", path);
    return failure();
  };

  // Detect cycles
  auto getNodeLoc = [](const FieldRef &ref) -> Location {
    return ref.getLoc();
  };

  if (failed(detector.detectCycles(cycleCallback, getNodeLoc)))
    return failure();

  // Record paths between ports for inter-module analysis
  // Since the fieldRefGraph only contains domain-related nodes, recordPortPaths
  // will automatically only record paths between domain ports (non-domain ports
  // won't have corresponding nodes in the graph).
  modulePortPaths[module] = recordPortPaths(module, fieldRefGraph);

  return success();
}

LogicalResult DomainCycleChecker::checkCircuit() {
  // Traverse modules in post-order to process callees before callers.
  // This allows us to propagate domain paths bottom-up through the hierarchy.
  for (auto *igNode : llvm::post_order<InstanceGraph *>(&instanceGraph)) {
    if (auto module = dyn_cast<FModuleOp>(*igNode->getModule())) {
      if (failed(checkModule(module)))
        return failure();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// CheckDomainCycles Pass
//===----------------------------------------------------------------------===//

class CheckDomainCyclesPass
    : public circt::firrtl::impl::CheckDomainCyclesBase<CheckDomainCyclesPass> {
public:
  void runOnOperation() override {
    auto &instanceGraph = getAnalysis<InstanceGraph>();

    // Create the domain cycle checker
    DomainCycleChecker checker(instanceGraph);

    // Run cycle detection across the circuit
    if (failed(checker.checkCircuit())) {
      return signalPassFailure();
    }

    markAllAnalysesPreserved();
  }
};

} // namespace
