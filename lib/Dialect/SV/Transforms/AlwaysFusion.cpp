//===- AlwaysFusion.cpp - TODO --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "SVPassDetails.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/Visitors.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace sv;

namespace {

struct SimpleOperationInfo : public llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(const Operation *opC) {
    return OperationEquivalence::computeHash(const_cast<Operation *>(opC));
  }
  static bool isEqual(const Operation *lhsC, const Operation *rhsC) {
    auto *lhs = const_cast<Operation *>(lhsC);
    auto *rhs = const_cast<Operation *>(rhsC);
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return OperationEquivalence::isEquivalentTo(const_cast<Operation *>(lhsC),
                                                const_cast<Operation *>(rhsC));
  }
};

struct AlwaysFusionPass : public AlwaysFusionBase<AlwaysFusionPass> {
  void runOnOperation() override;
};

// Returns true if the similar operations of the same kind should be merged
// together.  The operations must only have one block per region, and return no
// values.
static bool mergable(Operation *op) {
  return TypeSwitch<Operation *, bool>(op)
      .Case<AlwaysFFOp, IfOp>([](Operation *) { return true; })
      .Default([](Operation *) { return false; });
}

// inline all regions from op2 to op1
static void merge(Operation *op1, Operation *op2) {
  assert(op1 != op2 && "Cannot merge the same op");
  assert(op1->getName() == op2->getName() && "Ops must be the same type");
  assert(op1->getNumRegions() == op2->getNumRegions() &&
         "Ops must have same number of regions");

  for (unsigned i = 0; i < op1->getNumRegions(); ++i) {
    auto &region1 = op1->getRegion(i);
    auto &region2 = op2->getRegion(i);
    if (region1.empty()) {
      // If both regions are empty, move on to the next pair of regions
      if (region2.empty())
        continue;
      // If the first region has no block, move the second region's block over.
      region1.getBlocks().splice(region1.end(), region2.getBlocks());
      continue;
    }
    // If the second region is not empty, splice its block into the end of
    // the first region.
    if (!region2.empty()) {
      auto *block1 = &region1.front();
      auto *block2 = &region2.front();
      // Remove the terminator from the first block before merging.
      block1->back().erase();
      block1->getOperations().splice(block1->end(), block2->getOperations());
    }
  }
} // namespace

void AlwaysFusionPass::runOnOperation() {
  ModuleOp module = getOperation();
  SmallVector<Block *> worklist;
  worklist.push_back(module.getBody());

  while (!worklist.empty()) {
    // Scan each block in the worklist for identical operations.  This will not
    // consider operations in different blocks.
    auto *block = worklist.pop_back_val();
    DenseSet<Operation *, SimpleOperationInfo> knownOps;
    for (auto &op : llvm::make_early_inc_range(*block)) {
      // If the operation is in our whitelist of mergable operations,
      // check if we have encountered an identical operation already.  If we
      // have, merge them together and delete the old operation.
      if (mergable(&op)) {
        auto it = knownOps.find(&op);
        if (it != knownOps.end()) {
          merge(*it, &op);
          op.erase();
          continue;
        }

        // If we haven't encountered this operation before, store it in a set
        // of our known operations.
        knownOps.insert(&op);
      }

      // Add any regions belonging to the operation
      for (Region &region : op.getRegions()) {
        // TODO: verify that the region is not a CFG region.
        for (Block &block : region) {
          worklist.push_back(&block);
        }
      }
    }
  }
}

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> circt::sv::createAlwaysFusionPass() {
  return std::make_unique<AlwaysFusionPass>();
}
