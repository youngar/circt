//===- IMDeadCodeElim.cpp - Intermodule Dead Code Elimination ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-imdeadcodeelim"

using namespace circt;
using namespace firrtl;

/// Return true if this is a wire or a register or a node.
static bool isWireOrRegOrNode(Operation *op) {
  return isa<WireOp, RegResetOp, RegOp, NodeOp>(op);
}

/// Return true if this is a wire or register we're allowed to delete.
static bool isDeletableWireOrRegOrNode(Operation *op) {
  if (auto name = dyn_cast<FNamableOp>(op))
    if (!name.hasDroppableName())
      return false;
  return !hasDontTouch(op);
}

namespace {
struct IMDeadCodeElimPass : public IMDeadCodeElimBase<IMDeadCodeElimPass> {
  void runOnOperation() override;

  void rewriteModuleSignature(FModuleOp module);
  void rewriteModuleBody(FModuleOp module);

  void markAlive(Value value) {
    //  If the value is already in `liveSet`, skip it.
    if (liveSet.insert(value).second)
      worklist.push_back(value);
  }

  /// Return true if the value is known alive.
  bool isKnownAlive(Value value) const {
    assert(value && "null should not be used");
    return liveSet.count(value);
  }

  /// Return true if the value is assumed dead.
  bool isAssumedDead(Value value) const { return !isKnownAlive(value); }

  /// Return true if the block is alive.
  bool isBlockExecutable(Block *block) const {
    return executableBlocks.count(block);
  }

  void visitUser(Operation *op);
  void visitValue(Value value);
  void visitConnect(FConnectLike connect);
  void visitSubelement(Operation *op);
  void markBlockExecutable(Block *block);
  void markWireOrRegOrNode(Operation *op);
  void markMemOp(MemOp op);
  void markInstanceOp(InstanceOp instanceOp);
  void markUnknownSideEffectOp(Operation *op);

private:
  /// The set of blocks that are known to execute, or are intrinsically alive.
  DenseSet<Block *> executableBlocks;

  /// This keeps track of users the instance results that correspond to output
  /// ports.
  DenseMap<BlockArgument, llvm::TinyPtrVector<Value>>
      resultPortToInstanceResultMapping;
  InstanceGraph *instanceGraph;

  /// A worklist of values whose liveness recently changed, indicating the
  /// users need to be reprocessed.
  SmallVector<Value, 64> worklist;
  llvm::DenseSet<Value> liveSet;
};
} // namespace

void IMDeadCodeElimPass::markWireOrRegOrNode(Operation *op) {
  assert(isWireOrRegOrNode(op) && "only a wire, a reg or a node is expected");
  if (!isDeletableWireOrRegOrNode(op))
    markAlive(op->getResult(0));
}

void IMDeadCodeElimPass::markMemOp(MemOp mem) {
  for (auto result : mem.getResults())
    markAlive(result);
}

void IMDeadCodeElimPass::markUnknownSideEffectOp(Operation *op) {
  // For operations with side effects, pessimistically mark results and
  // operands as alive.
  for (auto result : op->getResults())
    markAlive(result);
  for (auto operand : op->getOperands())
    markAlive(operand);
}

void IMDeadCodeElimPass::visitUser(Operation *op) {
  LLVM_DEBUG(llvm::dbgs() << "Visit: " << *op << "\n");
  if (auto connectOp = dyn_cast<FConnectLike>(op))
    return visitConnect(connectOp);
  if (isa<SubfieldOp, SubindexOp, SubaccessOp>(op))
    return visitSubelement(op);
}

void IMDeadCodeElimPass::markInstanceOp(InstanceOp instance) {
  // Get the module being referenced.
  Operation *op = instanceGraph->getReferencedModule(instance);

  // If this is an extmodule, just remember that any inputs and inouts are
  // alive.
  if (!isa<FModuleOp>(op)) {
    auto module = dyn_cast<FModuleLike>(op);
    for (auto resultNo : llvm::seq(0u, instance.getNumResults())) {
      auto portVal = instance.getResult(resultNo);
      // If this is an output to the extmodule, we can ignore it.
      if (module.getPortDirection(resultNo) == Direction::Out)
        continue;

      // Otherwise this is an inuput from it or an inout, mark it as alive.
      markAlive(portVal);
    }
    return;
  }

  // Otherwise this is a defined module.
  auto fModule = cast<FModuleOp>(op);
  markBlockExecutable(fModule.getBody());

  // Ok, it is a normal internal module reference so populate
  // resultPortToInstanceResultMapping.
  for (size_t resultNo = 0, e = instance.getNumResults(); resultNo != e;
       ++resultNo) {
    auto instancePortVal = instance.getResult(resultNo);

    // Otherwise we have a result from the instance.  We need to forward results
    // from the body to this instance result's SSA value, so remember it.
    BlockArgument modulePortVal = fModule.getArgument(resultNo);

    resultPortToInstanceResultMapping[modulePortVal].push_back(instancePortVal);
  }
}

void IMDeadCodeElimPass::markBlockExecutable(Block *block) {
  if (!executableBlocks.insert(block).second)
    return; // Already executable.

  // Mark ports with don't touch as alive.
  for (auto blockArg : block->getArguments())
    if (hasDontTouch(blockArg))
      markAlive(blockArg);

  for (auto &op : *block) {
    if (isWireOrRegOrNode(&op))
      markWireOrRegOrNode(&op);
    else if (auto instance = dyn_cast<InstanceOp>(op))
      markInstanceOp(instance);
    else if (isa<FConnectLike>(op))
      // Skip connect op.
      continue;
    else if (auto mem = dyn_cast<MemOp>(op))
      markMemOp(mem);
    else if (!mlir::MemoryEffectOpInterface::hasNoEffect(&op))
      markUnknownSideEffectOp(&op);

    // TODO: Handle attach etc.
  }
}

void IMDeadCodeElimPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "===----- Remove unused ports -----==="
                          << "\n");
  auto circuit = getOperation();
  instanceGraph = &getAnalysis<InstanceGraph>();
  for (auto module : circuit.getBody()->getOps<FModuleOp>()) {
    // Mark the ports of public modules as alive.
    if (module.isPublic()) {
      markBlockExecutable(module.getBody());
      for (auto port : module.getBody()->getArguments())
        markAlive(port);
    }
  }

  // If a value changed liveness then propagate liveness through its users and
  // definition.
  while (!worklist.empty())
    visitValue(worklist.pop_back_val());

  // Rewrite module signatures.
  for (auto module : circuit.getBody()->getOps<FModuleOp>())
    rewriteModuleSignature(module);

  // Rewrite module bodies parallelly.
  mlir::parallelForEach(circuit.getContext(),
                        circuit.getBody()->getOps<FModuleOp>(),
                        [&](auto op) { rewriteModuleBody(op); });
}

void IMDeadCodeElimPass::visitValue(Value value) {
  assert(isKnownAlive(value) && "only alive values reach here");

  // Propagate liveness through users.
  for (Operation *user : value.getUsers())
    visitUser(user);

  // Requiring an input port propagates the liveness to each instance.
  if (auto blockArg = value.dyn_cast<BlockArgument>()) {
    auto module = cast<FModuleOp>(blockArg.getParentBlock()->getParentOp());
    auto portDirection = module.getPortDirection(blockArg.getArgNumber());
    // If the port is input, it's necessary to mark corresponding input ports of
    // instances as alive. We don't have to propagate the liveness of output
    // ports.
    if (portDirection == Direction::In)
      for (auto userOfResultPort : resultPortToInstanceResultMapping[blockArg])
        markAlive(userOfResultPort);
    return;
  }

  // Marking an instance port as alive propagates to the corresponding port of
  // the module.
  if (auto instance = value.getDefiningOp<InstanceOp>()) {
    auto instanceResult = value.cast<mlir::OpResult>();
    // Update the src, when it's an instance op.
    auto module =
        dyn_cast<FModuleOp>(*instanceGraph->getReferencedModule(instance));
    if (!module)
      return;

    BlockArgument modulePortVal =
        module.getArgument(instanceResult.getResultNumber());
    return markAlive(modulePortVal);
  }

  // If op is defined by an operation, mark its operands as alive.
  if (auto op = value.getDefiningOp())
    for (auto operand : op->getOperands())
      markAlive(operand);
}

void IMDeadCodeElimPass::visitConnect(FConnectLike connect) {
  // If the dest is alive, mark the source value as alive.
  if (isKnownAlive(connect.dest()))
    markAlive(connect.src());
}

void IMDeadCodeElimPass::visitSubelement(Operation *op) {
  if (isKnownAlive(op->getOperand(0)))
    markAlive(op->getResult(0));
}

void IMDeadCodeElimPass::rewriteModuleBody(FModuleOp module) {
  auto *body = module.getBody();
  // If the module is unreachable, just ignore it.
  // TODO: Erase this module from circuit op.
  if (!isBlockExecutable(body))
    return;

  // Walk the IR bottom-up when deleting operations.
  for (auto &op : llvm::make_early_inc_range(llvm::reverse(*body))) {
    // Connects to values that we found to be dead can be dropped.
    if (auto connect = dyn_cast<FConnectLike>(op)) {
      if (isAssumedDead(connect.dest())) {
        LLVM_DEBUG(llvm::dbgs() << "DEAD: " << connect << "\n";);
        connect.erase();
      }
      continue;
    }

    // Delete dead wires, regs and nodes.
    if (isWireOrRegOrNode(&op) && isAssumedDead(op.getResult(0))) {
      LLVM_DEBUG(llvm::dbgs() << "DEAD: " << op << "\n";);
      // Users should be already removed.
      assert(op.use_empty() && "no user");
      op.erase();
      continue;
    }

    // Remove non-sideeffect op using `isOpTriviallyDead`.
    if (mlir::isOpTriviallyDead(&op))
      op.erase();
  }
}

void IMDeadCodeElimPass::rewriteModuleSignature(FModuleOp module) {
  // If the module is unreachable, just ignore it.
  // TODO: Erase this module from circuit op.
  if (!isBlockExecutable(module.getBody()))
    return;

  // Ports of public modules cannot be modified.
  if (module.isPublic())
    return;

  InstanceGraphNode *instanceGraphNode =
      instanceGraph->lookup(module.moduleNameAttr());
  LLVM_DEBUG(llvm::dbgs() << "Prune ports of module: " << module.getName()
                          << "\n");
  SmallVector<unsigned> deadPortIndexes;
  unsigned numOldPorts = module.getNumPorts();

  ImplicitLocOpBuilder builder(module.getLoc(), module.getContext());
  builder.setInsertionPointToStart(module.getBody());

  for (auto index : llvm::seq(0u, numOldPorts)) {
    auto argument = module.getArgument(index);
    assert((!hasDontTouch(argument) || isKnownAlive(argument)) &&
           "If the port has don't touch, it should be known alive");

    // If the port has dontTouch, skip.
    if (hasDontTouch(argument))
      continue;

    // If the port is known alive, then we can't delete it except for write-only
    // output ports.
    if (isKnownAlive(argument)) {
      bool deadOutputPortAtAnyInstantiation =
          module.getPortDirection(index) == Direction::Out &&
          llvm::all_of(resultPortToInstanceResultMapping[argument],
                       [&](Value result) { return isAssumedDead(result); });

      if (!deadOutputPortAtAnyInstantiation)
        continue;

      // Ok, this port is used only within its defined module. So we can replace
      // the port with a wire.
      WireOp wire = builder.create<WireOp>(argument.getType());

      // Since `liveSet` contains the port, we have to erase it from the set.
      liveSet.erase(argument);
      liveSet.insert(wire);
      argument.replaceAllUsesWith(wire);
      deadPortIndexes.push_back(index);
      continue;
    }

    // Replace the port with a dummy wire. This wire should be erased within
    // `rewriteModuleBody`.
    WireOp wire = builder.create<WireOp>(argument.getType());
    argument.replaceAllUsesWith(wire);
    assert(isAssumedDead(wire) && "dummy wire must be dead");
    deadPortIndexes.push_back(index);
  }

  // If there is nothing to remove, abort.
  if (deadPortIndexes.empty())
    return;

  // Erase arguments of the old module from liveSet to prevent from creating
  // dangling pointers.
  for (auto arg : module.getArguments())
    liveSet.erase(arg);

  // Delete ports from the module.
  module.erasePorts(deadPortIndexes);

  // Add arguments of the new module to liveSet.
  for (auto arg : module.getArguments())
    liveSet.insert(arg);

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = cast<InstanceOp>(*use->getInstance());
    ImplicitLocOpBuilder builder(instance.getLoc(), instance);
    // Since we will rewrite instance op, it is necessary to remove old instance
    // results from liveSet.
    for (auto oldResult : instance.getResults())
      liveSet.erase(oldResult);

    // Replace old instance results with dummy wires.
    for (auto index : deadPortIndexes) {
      auto result = instance.getResult(index);
      assert(isAssumedDead(result) &&
             "instance results of dead ports must be dead");
      WireOp wire = builder.create<WireOp>(result.getType());
      result.replaceAllUsesWith(wire);
    }

    // Create a new instance op without dead ports.
    auto newInstance = instance.erasePorts(builder, deadPortIndexes);

    // Mark new results as alive.
    for (auto newResult : newInstance.getResults())
      liveSet.insert(newResult);

    instanceGraph->replaceInstance(instance, newInstance);
    // Remove old one.
    instance.erase();
  }

  numRemovedPorts += deadPortIndexes.size();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createIMDeadCodeElimPass() {
  return std::make_unique<IMDeadCodeElimPass>();
}
