//===- LowerOpenAggs.cpp - Lower Open Aggregate Types -----------*- C++ -*-===//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerOpenAggs pass.  This pass replaces the open
// aggregate types with hardware aggregates, with non-hardware fields
// expanded out as with LowerTypes.
//
// This pass is ref-specific for now.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/FIRRTLVisitors.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/IR/Visitors.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"

#include <vector>

#define DEBUG_TYPE "firrtl-lower-open-aggs"

using namespace circt;
using namespace firrtl;

namespace {

/// Information on non-hw (ref) elements.
struct NonHWField {
  /// Type of the field, not a hardware type.
  FIRRTLType type;
  /// FieldID relative to base of converted type.
  uint64_t fieldID;
  /// Relative orientation.  False means aligned.
  bool isFlip;
  /// String suffix naming this field.
  SmallString<16> suffix;
  /// Print this structure to llvm::errs().
  void dump() const;
};

/// Mapped port info
struct PortMappingInfo {
  /// Preserve this port, map use of old directly to new.
  bool identity;

  // When not identity, the port will be split:

  /// Type of the hardware-only portion.  May be null, indicating all non-hw.
  Type hwType;
  /// List of the individual non-hw fields to be split out.
  SmallVector<NonHWField, 0> fields;

  /// Determine number of types this argument maps to.
  size_t count(bool includeErased = false) const {
    if (identity)
      return 1;
    return fields.size() + (hwType ? 1 : 0) + (includeErased ? 1 : 0);
  }
  void dump() const;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const NonHWField &field) {
  return os << llvm::formatv(
             "non-HW(type={0}, fieldID={1}, isFlip={2}, suffix={3})",
             field.type, field.fieldID, field.isFlip, field.suffix);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                              const PortMappingInfo &pmi) {
  if (pmi.identity)
    return os << "(identity)";

  os << "[[hw portion: ";
  if (pmi.hwType)
    os << pmi.hwType;
  else
    os << "(none)";
  os << ", fields: ";
  llvm::interleaveComma(pmi.fields, os);
  return os << "]]";
}

} // namespace

void NonHWField::dump() const { llvm::errs() << *this; }
void PortMappingInfo::dump() const { llvm::errs() << *this; }

template <typename Range>
LogicalResult walkPortMappings(
    Range &&range, bool includeErased,
    llvm::function_ref<LogicalResult(size_t, PortMappingInfo &, size_t)>
        callback) {
  size_t count = 0;
  for (const auto &[index, pmi] : llvm::enumerate(range)) {
    if (failed(callback(index, pmi, count)))
      return failure();
    count += pmi.count(includeErased);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Visitor
//===----------------------------------------------------------------------===//

namespace {
class Visitor : public FIRRTLVisitor<Visitor, LogicalResult> {
public:
  explicit Visitor(MLIRContext *context) : context(context){};

  /// Entrypoint.
  LogicalResult visit(FModuleLike mod);

  using FIRRTLVisitor<Visitor, LogicalResult>::visitDecl;
  using FIRRTLVisitor<Visitor, LogicalResult>::visitExpr;
  using FIRRTLVisitor<Visitor, LogicalResult>::visitStmt;

  LogicalResult visitDecl(InstanceOp op);

  LogicalResult visitExpr(OpenSubfieldOp op);
  LogicalResult visitExpr(OpenSubindexOp op);

  LogicalResult visitUnhandledOp(Operation *op) {
    auto notOpenAggType = [](auto type) {
      return !isa<OpenBundleType, OpenVectorType>(type);
    };
    if (!llvm::all_of(op->getOperandTypes(), notOpenAggType) ||
        !llvm::all_of(op->getResultTypes(), notOpenAggType))
      return op->emitOpError(
          "unhandled use or producer of types containing references");
    return success();
  }

  LogicalResult visitInvalidOp(Operation *op) { return visitUnhandledOp(op); }

private:
  /// Convert a type to its HW-only projection.,
  /// Gather non-hw elements encountered and their names / positions.
  /// Returns a PortMappingInfo with its findings.
  PortMappingInfo mapPortType(Type type);

  MLIRContext *context;

  /// Map non-HW fields to their new Value.
  DenseMap<FieldRef, Value> nonHWValues;

  /// Map from port to its hw-only aggregate equivalent.
  DenseMap<Value, std::optional<Value>> hwOnlyAggMap;

  /// List of operations to erase at the end.
  SmallVector<Operation *> opsToErase;
};
} // namespace

LogicalResult Visitor::visit(FModuleLike mod) {
  auto ports = mod.getPorts();

  SmallVector<PortMappingInfo, 16> portMappings(
      llvm::map_range(ports, [&](auto &p) { return mapPortType(p.type); }));

  /// Total number of types mapped to.
  /// Include erased ports.
  size_t countWithErased = 0;
  for (auto &pmi : portMappings)
    countWithErased += pmi.count(/*includeErased=*/true);

  /// Ports to add.
  SmallVector<std::pair<unsigned, PortInfo>> newPorts;

  /// Ports to remove.
  BitVector portsToErase(countWithErased);

  /// Go through each port mapping, gathering information about all new ports.
  LLVM_DEBUG(llvm::dbgs() << "Ports for "
                          << cast<mlir::SymbolOpInterface>(*mod).getName()
                          << ":\n");
  auto result = walkPortMappings(
      portMappings, /*includeErased=*/true,
      [&](auto index, auto &pmi, auto newIndex) -> LogicalResult {
        LLVM_DEBUG(llvm::dbgs() << "\t" << ports[index].name << " : "
                                << ports[index].type << " => " << pmi << "\n");
        // Index for inserting new points next to this point.
        // (Immediately after current port's index).
        auto idxOfInsertPoint = index + 1;

        if (pmi.identity)
          return success();

        auto &port = ports[index];

        // If not identity, mark this port for eventual removal.
        portsToErase.set(newIndex);

        // Create new hw-only port, this will generally replace this port.
        if (pmi.hwType) {
          auto newPort = port;
          newPort.type = pmi.hwType;
          newPorts.emplace_back(idxOfInsertPoint, newPort);

          // If want to run this pass later, need to fixup annotations.
          if (!port.annotations.empty())
            return mlir::emitError(port.loc)
                   << "annotations on open aggregates not handled yet";
        } else {
          if (port.sym)
            return mlir::emitError(port.loc)
                   << "symbol found on aggregate with no HW";
          if (!port.annotations.empty())
            return mlir::emitError(port.loc)
                   << "annotations found on aggregate with no HW";
        }

        // Create ports for each non-hw field.
        for (const auto &[findex, field] : llvm::enumerate(pmi.fields)) {
          auto name = StringAttr::get(context,
                                      Twine(port.name.strref()) + field.suffix);
          auto orientation =
              (Direction)((unsigned)port.direction ^ field.isFlip);
          PortInfo pi(name, field.type, orientation, /*symName=*/StringAttr{},
                      port.loc, std::nullopt);
          newPorts.emplace_back(idxOfInsertPoint, pi);
        }
        return success();
      });
  if (failed(result))
    return failure();

  // Insert the new ports!
  mod.insertPorts(newPorts);

  assert(mod->getNumRegions() == 1);

  // (helper to determine/get the body block if present)
  auto getBodyBlock = [](auto mod) {
    auto &blocks = mod->getRegion(0).getBlocks();
    return !blocks.empty() ? &blocks.front() : nullptr;
  };

  // Process body block.
  // Create mapping for ports, then visit all operations within.
  if (auto *block = getBodyBlock(mod)) {
    // Create mappings for split ports.
    auto result =
        walkPortMappings(portMappings, /*includeErased=*/true,
                         [&](auto index, PortMappingInfo &pmi, auto newIndex) {
                           // Nothing to do for identity.
                           if (pmi.identity)
                             return success();

                           // newIndex is index of this port after insertion.
                           // This will be removed.
                           assert(portsToErase.test(newIndex));
                           auto oldPort = block->getArgument(newIndex);
                           auto newPortIndex = newIndex;

                           // Create mappings for split ports.
                           if (pmi.hwType)
                             hwOnlyAggMap[oldPort] =
                                 block->getArgument(++newPortIndex);
                           else
                             hwOnlyAggMap[oldPort] = std::nullopt;

                           for (auto &field : pmi.fields) {
                             auto ref = FieldRef(oldPort, field.fieldID);
                             auto newVal = block->getArgument(++newPortIndex);
                             nonHWValues[ref] = newVal;
                           }
                           return success();
                         });
    if (failed(result))
      return failure();

    // Walk the module.
    if (block
            ->walk<mlir::WalkOrder::PreOrder>([&](Operation *op) -> WalkResult {
              return dispatchVisitor(op);
            })
            .wasInterrupted())
      return failure();

    // Cleanup dead operations.
    for (auto &op : llvm::reverse(opsToErase))
      op->erase();
  }

  // Drop dead ports.
  mod.erasePorts(portsToErase);

  return success();
}

LogicalResult Visitor::visitExpr(OpenSubfieldOp op) {
  // We're indexing into an OpenBundle, which contains some non-hw elements and
  // may contain hw elements.

  // By the time this is reached, the "root" storage for the input
  // has already been handled and mapped to its new location(s),
  // such that the hardware-only contents are split from non-hw.

  // If there is a hardware portion selected by this operation,
  // create a "closed" subfieldop using the hardware-only new storage,
  // and add an entry mapping our old (soon, dead) result to
  // this new hw-only result (of the subfieldop).

  // Downstream indexing operations will expect that they can
  // still chase up through this operation, and that they will find
  // the hw-only portion in the map.

  // If this operation selects a non-hw element (not mixed),
  // look up where that ref now lives and update all users to use that instead.
  // (This case falls under "this selects only non-hw", which means
  // that this operation is now dead).

  // In all cases, this operation will be dead and should be removed.
  opsToErase.push_back(op);

  // Chase this to its original root.
  // If the FieldRef for this selection has a new home,
  // RAUW to that value and this op is dead.
  auto resultRef = getFieldRefFromValue(op.getResult());
  auto nonHWForResult = nonHWValues.find(resultRef);
  if (nonHWForResult != nonHWValues.end()) {
    auto newResult = nonHWForResult->second;
    assert(op.getResult().getType() == newResult.getType());
    assert(!isa<FIRRTLBaseType>(newResult.getType()));
    op.getResult().replaceAllUsesWith(newResult);
    return success();
  }

  assert(hwOnlyAggMap.count(op.getInput()));

  auto newInput = hwOnlyAggMap[op.getInput()];
  // Skip if no hw-only portion.  This is dead.
  if (!newInput.has_value()) {
    hwOnlyAggMap[op.getResult()] = std::nullopt;
    return success();
  }

  auto bundleType = cast<BundleType>(newInput->getType());

  // Recompute the "actual" index for this field, it may have changed.
  auto fieldName = op.getFieldName();
  auto newFieldIndex = bundleType.getElementIndex(fieldName);
  assert(newFieldIndex.has_value());

  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto newOp = builder.create<SubfieldOp>(*newInput, *newFieldIndex);
  if (auto name = op->getAttrOfType<StringAttr>("name"))
    newOp->setAttr("name", name);

  hwOnlyAggMap[op.getResult()] = newOp;

  if (isa<FIRRTLBaseType>(op.getType()))
    op.getResult().replaceAllUsesWith(newOp.getResult());

  return success();
}

LogicalResult Visitor::visitExpr(OpenSubindexOp op) {

  // In all cases, this operation will be dead and should be removed.
  opsToErase.push_back(op);

  // Chase this to its original root.
  // If the FieldRef for this selection has a new home,
  // RAUW to that value and this op is dead.
  auto resultRef = getFieldRefFromValue(op.getResult());
  auto nonHWForResult = nonHWValues.find(resultRef);
  if (nonHWForResult != nonHWValues.end()) {
    auto newResult = nonHWForResult->second;
    assert(op.getResult().getType() == newResult.getType());
    assert(!isa<FIRRTLBaseType>(newResult.getType()));
    op.getResult().replaceAllUsesWith(newResult);
    return success();
  }

  assert(hwOnlyAggMap.count(op.getInput()));

  auto newInput = hwOnlyAggMap[op.getInput()];
  // Skip if no hw-only portion.  This is dead.
  if (!newInput.has_value()) {
    hwOnlyAggMap[op.getResult()] = std::nullopt;
    return success();
  }

  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto newOp = builder.create<SubindexOp>(*newInput, op.getIndex());
  if (auto name = op->getAttrOfType<StringAttr>("name"))
    newOp->setAttr("name", name);

  hwOnlyAggMap[op.getResult()] = newOp;

  if (isa<FIRRTLBaseType>(op.getType()))
    op.getResult().replaceAllUsesWith(newOp.getResult());
  return success();
}

LogicalResult Visitor::visitDecl(InstanceOp op) {
  // Rewrite ports same strategy as for modules.

  auto type = op.getType();
  SmallVector<PortMappingInfo, 16> portMappings(
      llvm::map_range(type.getElements(),
                      [&](auto element) { return mapPortType(element.type); }));

  /// Total number of types mapped to.
  size_t countWithErased = 0;
  for (auto &pmi : portMappings)
    countWithErased += pmi.count(/*includeErased=*/true);

  /// Ports to add.
  SmallVector<std::pair<unsigned, PortInfo>> newPorts;

  /// Ports to remove.
  BitVector portsToErase(countWithErased);

  // Mapping from old-port index to new-port indices(0+).
  SmallVector<SmallVector<size_t>> portMap;

  /// Go through each port mapping, gathering information about all new ports.
  LLVM_DEBUG(llvm::dbgs() << "Ports for " << op << ":\n");
  auto result = walkPortMappings(
      portMappings, /*includeErased=*/true,
      [&](auto index, auto &pmi, auto newIndex) -> LogicalResult {
        LLVM_DEBUG(llvm::dbgs()
                   << "\t" << op.getPortName(index) << " : "
                   << op.getElement(index).type << " => " << pmi << "\n");
        // Index for inserting new points next to this point.
        // (Immediately after current port's index).
        auto idxOfInsertPoint = index + 1;

        if (pmi.identity)
          return success();

        // If not identity, mark this port for eventual removal.
        portsToErase.set(newIndex);

        auto portName = op.getPortName(index);
        auto portDirection = op.getPortDirection(index);
        auto loc = op.getLoc();

        // Create new hw-only port, this will generally replace this port.
        if (pmi.hwType) {
          PortInfo hwPort(portName, pmi.hwType, portDirection,
                          /*symName=*/StringAttr{}, loc,
                          AnnotationSet(op.getPortAnnotation(index)));
          newPorts.emplace_back(idxOfInsertPoint, hwPort);

          // If want to run this pass later, need to fixup annotations.
          if (!op.getPortAnnotation(index).empty())
            return mlir::emitError(op.getLoc())
                   << "annotations on open aggregates not handled yet";
        } else {
          if (!op.getPortAnnotation(index).empty())
            return mlir::emitError(op.getLoc())
                   << "annotations found on aggregate with no HW";
        }

        // Create ports for each non-hw field.
        for (const auto &[findex, field] : llvm::enumerate(pmi.fields)) {
          auto name =
              StringAttr::get(context, Twine(portName.strref()) + field.suffix);
          auto orientation =
              (Direction)((unsigned)portDirection ^ field.isFlip);
          PortInfo pi(name, field.type, orientation, /*symName=*/StringAttr{},
                      loc, std::nullopt);
          newPorts.emplace_back(idxOfInsertPoint, pi);
        }
        return success();
      });
  if (failed(result))
    return failure();

  // If no new ports, we're done.
  if (newPorts.empty())
    return success();

  // Create new instance op with desired ports.

  // TODO: add and erase ports without intermediate + various array attributes.
  auto tempOp = op.cloneAndInsertPorts(newPorts);
  opsToErase.push_back(tempOp);
  ImplicitLocOpBuilder builder(op.getLoc(), op);
  auto newInst = tempOp.erasePorts(builder, portsToErase);

  // Update each InstanceSubOp to point to updated port index, fixing up
  // it's users.
  // Group subops by the port index they refer to.
  SmallVector<SmallVector<InstanceSubOp, 1>> accesses(op.getNumElements(), {});
  op.eachSubOp([&](auto sub) { accesses[sub.getIndex()].push_back(sub); });

  auto mappingResult = walkPortMappings(
      portMappings, /*includeErased=*/false,
      [&](auto index, PortMappingInfo &pmi, auto newIndex) {
        // no work to do if the port was unused.
        if (accesses[index].size() == 0)
          return success();

        // Identity means index -> newIndex.
        // Update the original subOps for this port in-place.
        if (pmi.identity) {
          for (auto sub : accesses[index]) {
            assert(sub.getType() == newInst.getElement(newIndex).type);
            sub.getInputMutable().assign(newInst);
            sub.setIndex(newIndex);
          }
          return success();
        }

        // Create mappings for updating open aggregate users.
        auto newPortIndex = newIndex;
        std::optional<Value> newHwResult;
        if (pmi.hwType)
          newHwResult = builder.create<InstanceSubOp>(newInst, newPortIndex++);
        for (auto oldResult : accesses[index]) {
          hwOnlyAggMap[oldResult] = newHwResult;
          opsToErase.push_back(oldResult);
        }

        for (auto &field : pmi.fields) {
          auto newVal = builder.create<InstanceSubOp>(newInst, newPortIndex++);
          for (auto oldResult : accesses[index]) {
            auto ref = FieldRef(oldResult, field.fieldID);
            assert(newVal.getType() == field.type);
            nonHWValues[ref] = newVal;
          }
        }
        return success();
      });
  if (failed(mappingResult))
    return failure();

  opsToErase.push_back(op);

  return success();
}

//===----------------------------------------------------------------------===//
// Type Conversion
//===----------------------------------------------------------------------===//

PortMappingInfo Visitor::mapPortType(Type type) {
  PortMappingInfo pi{false, {}, {}};
  auto ftype = dyn_cast<FIRRTLType>(type);
  // Ports that aren't open aggregates are left alone.
  if (!ftype || !isa<OpenBundleType, OpenVectorType>(ftype)) {
    pi.identity = true;
    return pi;
  }

  // NOLINTBEGIN(misc-no-recursion)
  auto recurse = [&](auto &&f, FIRRTLType type, const Twine &suffix = "",
                     bool flip = false,
                     uint64_t fieldID = 0) -> FIRRTLBaseType {
    return TypeSwitch<FIRRTLType, FIRRTLBaseType>(type)
        .Case<FIRRTLBaseType>([](auto base) { return base; })
        .template Case<OpenBundleType>(
            [&](OpenBundleType obTy) -> FIRRTLBaseType {
              SmallVector<BundleType::BundleElement> hwElements;
              for (const auto &[index, element] :
                   llvm::enumerate(obTy.getElements()))
                if (auto base =
                        f(f, element.type, suffix + "_" + element.name.strref(),
                          flip ^ element.isFlip,
                          fieldID + obTy.getFieldID(index)))
                  hwElements.emplace_back(element.name, element.isFlip, base);

              if (hwElements.empty())
                return FIRRTLBaseType{};

              return BundleType::get(context, hwElements, obTy.isConst());
            })
        .template Case<OpenVectorType>(
            [&](OpenVectorType ovTy) -> FIRRTLBaseType {
              FIRRTLBaseType convert;
              // Walk for each index to extract each leaf separately, but expect
              // same hw-only type for all.
              for (auto idx : llvm::seq<size_t>(0U, ovTy.getNumElements())) {
                auto hwElementType =
                    f(f, ovTy.getElementType(), suffix + "_" + Twine(idx), flip,
                      fieldID + ovTy.getFieldID(idx));
                assert((!convert || convert == hwElementType) &&
                       "expected same hw type for all elements");
                convert = hwElementType;
              }

              if (!convert)
                return FIRRTLBaseType{};

              return FVectorType::get(convert, ovTy.getNumElements(),
                                      ovTy.isConst());
            })
        .template Case<RefType>([&](auto ref) {
          // Do this better, don't re-serialize so much?
          auto f = NonHWField{ref, fieldID, flip, {}};
          suffix.toVector(f.suffix);
          pi.fields.emplace_back(std::move(f));
          return FIRRTLBaseType{};
        })
        .Default(FIRRTLBaseType{});
  };

  pi.hwType = recurse(recurse, ftype);
  assert(pi.hwType != type);
  // NOLINTEND(misc-no-recursion)

  return pi;
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct LowerOpenAggsPass : public LowerOpenAggsBase<LowerOpenAggsPass> {
  LowerOpenAggsPass() = default;
  void runOnOperation() override;
};
} // end anonymous namespace

// This is the main entrypoint for the lowering pass.
void LowerOpenAggsPass::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "===- Running Lower Open Aggregates Pass "
                      "------------------------------------------------===\n");
  SmallVector<Operation *, 0> ops(getOperation().getOps<FModuleLike>());

  auto result = failableParallelForEach(&getContext(), ops, [&](Operation *op) {
    Visitor visitor(&getContext());
    return visitor.visit(cast<FModuleLike>(op));
  });

  if (result.failed())
    signalPassFailure();
}

/// This is the pass constructor.
std::unique_ptr<mlir::Pass> circt::firrtl::createLowerOpenAggsPass() {
  return std::make_unique<LowerOpenAggsPass>();
}
