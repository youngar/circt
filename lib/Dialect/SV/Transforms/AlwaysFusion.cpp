//===- AlwaysFusion.cpp - TODO --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass performs simple fusion of if and always_ff
// in the same region.
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

/// Check the equivalence of operations by doing a deep comparison of operands
/// and attributes, but does not compare the content of any regions attached to
/// each op.
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

static void simplifyOperation(Operation *op);

/// Returns true if the similar operations of the same kind can be merged
/// together, acting as a whitelist of operations kinds to consider. Currently,
/// all regions must be graph regions.
static bool isMergableOp(Operation *op) { return isa<AlwaysFFOp, IfOp>(op); }

/// Merge two SSACFG regions together.
static void mergeCFGRegions(Region *region1, Region *region2) {
  assert("Unable to merge SSACFG regions.");
}

// Merge two graph regions together.  Graph regions can only have a single
// block.
static void mergeGraphRegions(Region *region1, Region *region2) {
  assert(region1->getNumArguments() == 0 && region2->getNumArguments() == 0 &&
         "Region must not have any arguments.");
  if (region1->empty()) {
    // If both regions are empty, move on to the next pair of regions
    if (region2->empty())
      return;
    // If the first region has no block, move the second region's block over.
    region1->getBlocks().splice(region1->end(), region2->getBlocks());
    return;
  }
  // If the second region is not empty, splice its block into the end of
  // the first region.
  if (!region2->empty()) {
    auto &block1 = region1->front();
    auto &block2 = region2->front();
    // Remove the terminator from the first block before merging.
    block1.back().erase();
    block1.getOperations().splice(block1.end(), block2.getOperations());
  }
}

/// Inline all regions from the second operation into the first.
static void mergeOperations(Operation *op1, Operation *op2) {
  assert(op1 != op2 && "Cannot merge an op into itself");
  assert(op1->getName() == op2->getName() && "Ops must be the same type");
  assert(op1->getNumRegions() == op2->getNumRegions() &&
         "Ops must have same number of regions");
  // This could replace all uses of op2 with op1, but we don't have a use case
  // for this yet.  Adding support should be trivial as adding a
  // replaceAllUsesOf(..) at the end of merging.
  assert(op1->getNumResults() == 0 && "Can not merge operations with results");
  auto kindInterface = dyn_cast<mlir::RegionKindInterface>(op1);
  for (unsigned i = 0; i < op1->getNumRegions(); ++i) {
    if (kindInterface && kindInterface.getRegionKind(i) == RegionKind::Graph)
      mergeGraphRegions(&op1->getRegion(i), &op2->getRegion(i));
    else
      mergeCFGRegions(&op1->getRegion(i), &op2->getRegion(i));
  }
}

static bool simplifyBlock(Block *block) {
  return false;
}

// Simplify the operations in a regular region.
static bool simplifyCFGRegion(Region *region) {
  // In CFG regions, only merge operations which are adjacent.

  bool result = false;
  for (Block &block : *region) {
    for (Operation &op : block) {
      result |= simplifyOperation(&op);
    }
  }
  return result;
}

/// Simplify the operations in a graph region.  Combine all mergable operations
/// in the region without regard to the order of operations in the region.  The
/// difference between a graph region and a SSACFG region is that all values in
/// a block dominate all regions, and must be processed first.  Returns true if
/// any operations were simplified.
static bool simplifyGraphRegion(Region *region) {
  bool result = false;
  // A set of operations in the current block which are mergable. Any operation
  // in this set is a candidate for another similar operation to merge in to.
  DenseSet<Operation *, SimpleOperationInfo> mergableOps;
  for (Block &block : *region) {
    for (Operation &op : block) {
      // If the operation is in our whitelist of mergable operations, check if
      // we have encountered an equivalent operation already.  If we have,
      // merge them together and delete the old operation.
      if (isMergableOp(&op)) {
        auto it = mergableOps.find(&op);
        if (it != mergableOps.end()) {
          mergeOperations(*it, &op);
          op.erase();
          result = true;
          continue;
        }
        // Record new mergable ops as a candidate for merging
        mergableOps.insert(&op);
      }
    }
  }
  // Recursively simplify all regions
  for (Block &block : *region) {
    for (Operation &op : block) {
      result |= simplifyOperation(&op);
    }
  }
  return result;
}

/// Simplify an operation and all of its regions.  Returns true if the operation
/// was simplified.
bool simplifyOperation(Operation *op) {
  // Don't simplify terminator operations.
  if (op->isKnownTerminator())
    return false;

  bool result = false;
  auto kindInterface = dyn_cast<mlir::RegionKindInterface>(op);
  for (unsigned i = 0; i < op->getNumRegions(); ++i) {
    // If the operation does not implement the region kind interface, all of its
    // regions are implicitly regular SSACFG region.
    if (kindInterface && kindInterface.getRegionKind(i) == RegionKind::Graph)
      result |= simplifyGraphRegion(&op->getRegion(i));
    else
      result |= simplifyCFGRegion(&op->getRegion(i));
  }
  return result;
}

struct AlwaysFusionPass : public AlwaysFusionBase<AlwaysFusionPass> {
  void runOnOperation() override {
    // Simplify the root operation and all regions recursively.  If this returns
    // true, then the graph was changed and all analysis are removed.
    if (simplifyOperation(getOperation()))
      return;

    // If we did not change anything in the graph mark all analysis as
    // preserved.
    markAllAnalysesPreserved();
  }
};

} // end anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> circt::sv::createAlwaysFusionPass() {
  return std::make_unique<AlwaysFusionPass>();
}
